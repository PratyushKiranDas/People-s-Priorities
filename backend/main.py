from __future__ import annotations

import mimetypes
import json
import os
from threading import Lock
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ai_engine import analyze_submission, fetch_demographics_from_bigquery

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"
SUBMISSIONS_FILE = BASE_DIR / "data" / "submissions.json"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WORKFLOW_STATUSES = {
    "Noticed by Government",
    "Under Process",
    "Work Started",
    "Work Done",
}

SUBMISSION_DEFAULTS = {
    "status": "Noticed by Government",
    "mp_explanation": "",
    "citizen_review": None,
    "is_archived": False,
}


def _project_id() -> str:
    return (
        os.getenv("GCP_PROJECT_ID")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("FIREBASE_PROJECT_ID")
        or ""
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _extension_for(content_type: str, filename: Optional[str] = None) -> str:
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix[:16]
    return mimetypes.guess_extension((content_type or "").split(";")[0]) or ".bin"


def _with_submission_defaults(record: Dict[str, Any]) -> Dict[str, Any]:
    record.setdefault("status", SUBMISSION_DEFAULTS["status"])
    if record.get("status") == "new":
        record["status"] = SUBMISSION_DEFAULTS["status"]
    record.setdefault("mp_explanation", SUBMISSION_DEFAULTS["mp_explanation"])
    record.setdefault("citizen_review", SUBMISSION_DEFAULTS["citizen_review"])
    record.setdefault("is_archived", SUBMISSION_DEFAULTS["is_archived"])
    return record


def _is_active_submission(record: Dict[str, Any]) -> bool:
    return not bool(record.get("is_archived"))


class StorageBackend:
    """Offline-first JSON storage for hackathon demos without cloud database billing."""

    def __init__(self, path: Path = SUBMISSIONS_FILE) -> None:
        self.path = path
        self._lock = Lock()

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    def _read_rows(self) -> List[Dict[str, Any]]:
        self._ensure_file()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        return payload if isinstance(payload, list) else []

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        self._ensure_file()
        self.path.write_text(json.dumps(rows, ensure_ascii=True, indent=2), encoding="utf-8")

    async def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        def _append() -> Dict[str, Any]:
            with self._lock:
                record["id"] = record.get("id") or str(uuid.uuid4())
                _with_submission_defaults(record)
                rows = self._read_rows()
                rows.append(_json_safe(record))
                self._write_rows(rows)
            return record

        await run_in_threadpool(_append)
        return record

    async def load(self, limit: int = 250) -> List[Dict[str, Any]]:
        def _load() -> List[Dict[str, Any]]:
            with self._lock:
                rows = self._read_rows()
            rows = [_with_submission_defaults(row) for row in rows]
            rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
            return rows[:limit]

        return await run_in_threadpool(_load)

    async def update(self, submission_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        def _update() -> Optional[Dict[str, Any]]:
            with self._lock:
                rows = self._read_rows()
                for row in rows:
                    if str(row.get("id")) != submission_id:
                        continue

                    _with_submission_defaults(row)
                    row.update(updates)
                    if row.get("status") == "Work Done" and row.get("citizen_review") not in (None, ""):
                        row["is_archived"] = True
                    row["updated_at"] = _now_iso()
                    _with_submission_defaults(row)
                    self._write_rows(rows)
                    return row
            return None

        return await run_in_threadpool(_update)


storage_backend = StorageBackend()


def _local_media_record(media: Dict[str, Any]) -> Dict[str, Any]:
    """Return the JSON-safe media metadata saved in data/submissions.json."""

    return {
        key: value
        for key, value in media.items()
        if key != "bytes"
    }


async def save_bytes_to_uploads(
    content: bytes,
    content_type: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    if not content:
        raise HTTPException(status_code=400, detail="Cannot upload an empty media file.")

    extension = _extension_for(content_type, filename)
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / unique_filename
    await run_in_threadpool(destination.write_bytes, content)

    relative_path = (Path("data") / "uploads" / unique_filename).as_posix()
    return {
        "bytes": content,
        "mime_type": content_type or "application/octet-stream",
        "filename": unique_filename,
        "original_filename": filename,
        "path": relative_path,
    }


async def save_upload_file(upload: UploadFile) -> Optional[Dict[str, Any]]:
    if upload is None or not upload.filename:
        return None
    content = await upload.read()
    content_type = upload.content_type or "application/octet-stream"
    return await save_bytes_to_uploads(content, content_type, upload.filename)


async def download_external_media_to_local(media: Dict[str, Any]) -> Dict[str, Any]:
    url = media.get("url")
    provider = media.get("provider")
    if not url and media.get("id") and provider == "meta":
        url = await resolve_meta_media_url(str(media["id"]))
    if not url:
        raise HTTPException(status_code=400, detail="Webhook media needs a url or resolvable Meta media id.")

    headers: Dict[str, str] = {}
    auth = None
    parsed = urlparse(url)

    if "api.twilio.com" in parsed.netloc:
        sid = os.getenv("TWILIO_ACCOUNT_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        if sid and token:
            auth = (sid, token)

    if provider == "meta":
        token = os.getenv("META_WHATSAPP_ACCESS_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, auth=auth, headers=headers)
        response.raise_for_status()
        content_type = media.get("mime_type") or response.headers.get("content-type") or "application/octet-stream"
        filename = Path(parsed.path).name or None
        return await save_bytes_to_uploads(response.content, content_type, filename)


async def resolve_meta_media_url(media_id: str) -> str:
    token = os.getenv("META_WHATSAPP_ACCESS_TOKEN")
    if not token:
        raise HTTPException(status_code=400, detail="Set META_WHATSAPP_ACCESS_TOKEN to resolve Meta media ids.")

    graph_version = os.getenv("META_GRAPH_VERSION", "v21.0")
    url = f"https://graph.facebook.com/{graph_version}/{media_id}"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        payload = response.json()
        media_url = payload.get("url")
        if not media_url:
            raise HTTPException(status_code=400, detail="Meta media response did not include a download URL.")
        return media_url


app = FastAPI(title="People's Priorities API", version="2.0.0")

allowed_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_dashboard_user() -> Dict[str, Any]:
    """
    Offline demo auth shim.

    Cloud auth is intentionally bypassed for the hackathon demo so billing or
    network issues cannot block the MP dashboard. Protected API routes always
    receive this local mock user.
    """

    return {"uid": "local-dev"}


def public_dashboard_config() -> Dict[str, Any]:
    return {
        "mapsApiKey": os.getenv("GOOGLE_MAPS_API_KEY", ""),
        "requireAuth": False,
        "defaultMapCenter": {
            "lat": _env_float("DEFAULT_MAP_LAT", 28.6139),
            "lng": _env_float("DEFAULT_MAP_LNG", 77.2090),
        },
        "heatmapRadius": _parse_int(os.getenv("HEATMAP_RADIUS"), 38),
    }


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return FileResponse(INDEX_HTML)


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    return public_dashboard_config()


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "project": _project_id(),
        "storage_file": str(storage_backend.path),
        "upload_dir": str(UPLOAD_DIR),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    }


async def analyze_and_store(
    *,
    channel: str,
    text: str,
    media_refs: List[Dict[str, Any]],
    ward_id: Optional[str] = None,
    address: Optional[str] = None,
    sender: Optional[str] = None,
    raw_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    demographics = await run_in_threadpool(
        lambda: fetch_demographics_from_bigquery(constituency_id=os.getenv("CONSTITUENCY_ID"))
    )
    analysis = await run_in_threadpool(
        analyze_submission,
        text,
        media_refs,
        demographics.get("rows", []),
    )

    record = {
        "id": str(uuid.uuid4()),
        "channel": channel,
        "sender": sender,
        "text": text,
        "ward_id": ward_id,
        "address": address,
        "media": [_local_media_record(item) for item in media_refs],
        "analysis": analysis,
        "category": analysis.get("category"),
        "urgency_score": analysis.get("urgency_score"),
        "status": SUBMISSION_DEFAULTS["status"],
        "mp_explanation": SUBMISSION_DEFAULTS["mp_explanation"],
        "citizen_review": SUBMISSION_DEFAULTS["citizen_review"],
        "is_archived": SUBMISSION_DEFAULTS["is_archived"],
        "demographic_source": demographics.get("source"),
        "created_at": _now_iso(),
        "raw_metadata": raw_metadata or {},
    }
    return await storage_backend.append(record)


@app.post("/api/submissions")
@app.post("/api/submit")
async def create_submission(
    request: Request,
    text: str = Form(default=""),
    ward_id: Optional[str] = Form(default=None),
    address: str = Form(default=""),
    photo: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    legacy_file: Optional[UploadFile] = File(default=None, alias="file"),
) -> Dict[str, Any]:
    media_refs: List[Dict[str, Any]] = []

    image_upload = photo or image or legacy_file
    uploaded_image = await save_upload_file(image_upload) if image_upload else None
    if uploaded_image:
        media_refs.append(uploaded_image)

    if not text.strip() and not media_refs:
        raise HTTPException(status_code=400, detail="Submit text or a photo.")

    return await analyze_and_store(
        channel="web",
        text=text.strip(),
        media_refs=media_refs,
        ward_id=ward_id,
        address=address.strip(),
        raw_metadata={"client_host": request.client.host if request.client else None},
    )


@app.get("/api/submissions")
async def list_submissions(
    limit: int = Query(default=250, ge=1, le=1000),
    user: Dict[str, Any] = Depends(require_dashboard_user),
) -> Dict[str, Any]:
    del user
    rows = await storage_backend.load(limit=10000)
    active_rows = [row for row in rows if _is_active_submission(row)]
    return {"items": active_rows[:limit]}


@app.post("/api/submissions/{submission_id}/update")
@app.put("/api/submissions/{submission_id}/update")
async def update_submission(
    submission_id: str,
    request: Request,
    user: Dict[str, Any] = Depends(require_dashboard_user),
) -> Dict[str, Any]:
    del user

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        payload = dict(await request.form())

    updates: Dict[str, Any] = {}
    status_was_updated_to_done = False
    review_was_submitted = False

    if "status" in payload and payload.get("status") not in (None, ""):
        status = str(payload["status"]).strip()
        if status not in WORKFLOW_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Use one of: {', '.join(sorted(WORKFLOW_STATUSES))}.",
            )
        updates["status"] = status
        status_was_updated_to_done = status == "Work Done"

    if "mp_explanation" in payload:
        updates["mp_explanation"] = "" if payload.get("mp_explanation") is None else str(payload["mp_explanation"]).strip()

    if "citizen_review" in payload:
        raw_review = payload.get("citizen_review")
        updates["citizen_review"] = None if raw_review in (None, "") else raw_review
        review_was_submitted = updates["citizen_review"] is not None

    if not updates:
        raise HTTPException(status_code=400, detail="Submit status, mp_explanation, or citizen_review to update.")

    if status_was_updated_to_done and review_was_submitted:
        updates["is_archived"] = True

    updated = await storage_backend.update(submission_id, updates)
    if not updated:
        raise HTTPException(status_code=404, detail="Submission not found.")

    return {"ok": True, "item": updated}


def _demographic_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("ward_id", "")).lower(): row for row in rows if row.get("ward_id")}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.get("/api/priorities")
async def ranked_priorities(
    limit: int = Query(default=500, ge=1, le=1000),
    user: Dict[str, Any] = Depends(require_dashboard_user),
) -> Dict[str, Any]:
    del user
    submissions = [
        item
        for item in await storage_backend.load(limit=10000)
        if _is_active_submission(item)
    ][:limit]
    demographics = await run_in_threadpool(
        lambda: fetch_demographics_from_bigquery(constituency_id=os.getenv("CONSTITUENCY_ID"))
    )
    demo_by_ward = _demographic_index(demographics.get("rows", []))

    grouped: Dict[str, Dict[str, Any]] = {}
    for item in submissions:
        ward_id = str(item.get("ward_id") or "unknown")
        category = str(item.get("category") or (item.get("analysis") or {}).get("category") or "other")
        key = f"{ward_id}:{category}"
        urgency = _number(item.get("urgency_score") or (item.get("analysis") or {}).get("urgency_score"), 1)
        demo = demo_by_ward.get(ward_id.lower(), {})
        population_weight = min(_number(demo.get("population")) / 50000, 4)
        vulnerability_weight = min(_number(demo.get("vulnerability_index")) / 20, 4)
        weighted_score = urgency + population_weight + vulnerability_weight

        bucket = grouped.setdefault(
            key,
            {
                "ward_id": ward_id,
                "ward_name": demo.get("ward_name") or ward_id,
                "category": category,
                "count": 0,
                "max_urgency": 0,
                "demand_score": 0.0,
                "demographics": demo,
                "latest_summary": "",
            },
        )
        bucket["count"] += 1
        bucket["max_urgency"] = max(bucket["max_urgency"], urgency)
        bucket["demand_score"] += weighted_score
        bucket["latest_summary"] = (item.get("analysis") or {}).get("summary") or item.get("text") or ""

    priorities = sorted(grouped.values(), key=lambda row: row["demand_score"], reverse=True)
    return {
        "items": priorities,
        "demographics_source": demographics.get("source"),
        "demographics_query": demographics.get("query"),
    }


async def _read_webhook_payload(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        return await request.json()
    form = await request.form()
    return dict(form)


def _extract_twilio_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    media: List[Dict[str, Any]] = []
    for index in range(_parse_int(payload.get("NumMedia"))):
        url = payload.get(f"MediaUrl{index}")
        mime_type = payload.get(f"MediaContentType{index}") or "application/octet-stream"
        if url and not mime_type.startswith("audio/"):
            media.append({"provider": "twilio", "url": url, "mime_type": mime_type})

    return {
        "sender": payload.get("From") or payload.get("WaId"),
        "text": payload.get("Body", ""),
        "address": payload.get("Address", ""),
        "media": media,
        "raw": payload,
    }


def _media_from_meta(message: Dict[str, Any], msg_type: str) -> Optional[Dict[str, Any]]:
    media_obj = message.get(msg_type) or {}
    if not media_obj:
        return None
    return {
        "provider": "meta",
        "id": media_obj.get("id"),
        "url": media_obj.get("url"),
        "mime_type": media_obj.get("mime_type", "application/octet-stream"),
    }


def _extract_meta_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for message in value.get("messages", []):
                msg_type = message.get("type")
                text = ""
                media: List[Dict[str, Any]] = []

                if msg_type == "text":
                    text = (message.get("text") or {}).get("body", "")
                elif msg_type in {"image", "document"}:
                    media_item = _media_from_meta(message, msg_type)
                    if media_item:
                        media.append(media_item)

                location = message.get("location") or {}
                extracted.append(
                    {
                        "sender": message.get("from"),
                        "text": text,
                        "address": location.get("address") or location.get("name") or "",
                        "media": media,
                        "raw": message,
                    }
                )
    return extracted


def _extract_mock_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "messages" in payload and isinstance(payload["messages"], list):
        messages = payload["messages"]
    else:
        messages = [payload]

    extracted = []
    for message in messages:
        media = message.get("media", [])
        if isinstance(media, dict):
            media = [media]
        extracted.append(
            {
                "sender": message.get("from") or message.get("sender"),
                "text": message.get("text") or message.get("body") or "",
                "address": message.get("address", ""),
                "media": media,
                "raw": message,
            }
        )
    return extracted


def extract_whatsapp_messages(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "Body" in payload or "NumMedia" in payload:
        return [_extract_twilio_message(payload)]
    if "entry" in payload:
        return _extract_meta_messages(payload)
    return _extract_mock_messages(payload)


async def load_existing_local_media(media: Dict[str, Any]) -> Dict[str, Any]:
    raw_path = Path(str(media["path"]))
    file_path = raw_path if raw_path.is_absolute() else BASE_DIR / raw_path
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"Local media path not found: {media['path']}")

    content = await run_in_threadpool(file_path.read_bytes)
    mime_type = media.get("mime_type") or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    try:
        relative_path = file_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        relative_path = file_path.name
    return {
        "bytes": content,
        "mime_type": mime_type,
        "filename": file_path.name,
        "original_filename": media.get("filename") or file_path.name,
        "path": relative_path,
    }


async def _materialize_webhook_media(media_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for item in media_items:
        if item.get("path"):
            refs.append(await load_existing_local_media(item))
            continue
        refs.append(await download_external_media_to_local(item))
    return refs


@app.post("/api/webhook/whatsapp")
async def whatsapp_webhook(request: Request) -> Dict[str, Any]:
    payload = await _read_webhook_payload(request)
    messages = extract_whatsapp_messages(payload)
    saved: List[Dict[str, Any]] = []

    for message in messages:
        media_refs = await _materialize_webhook_media(message.get("media", []))
        if not str(message.get("text") or "").strip() and not media_refs:
            continue
        saved.append(
            await analyze_and_store(
                channel="whatsapp",
                text=str(message.get("text") or "").strip(),
                media_refs=media_refs,
                address=str(message.get("address") or "").strip(),
                sender=message.get("sender"),
                raw_metadata=message.get("raw"),
            )
        )

    return {"ok": True, "saved": saved}


# Register static frontend serving after API routes so /api/* endpoints keep
# priority, while /config.js, assets, and other frontend files resolve normally.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import socket

    import uvicorn

    def _find_available_port(bind_host: str, preferred_port: int, attempts: int = 25) -> int:
        for candidate in range(preferred_port, preferred_port + attempts):
            family = socket.AF_INET6 if ":" in bind_host and bind_host != "0.0.0.0" else socket.AF_INET
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind((bind_host, candidate))
                    return candidate
                except OSError:
                    continue
        return preferred_port

    host = os.getenv("HOST", "127.0.0.1")
    requested_port = _parse_int(os.getenv("PORT"), 8000)
    port = _find_available_port(host, requested_port)
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if port != requested_port:
        print(f"Port {requested_port} is busy; using {port} instead.")
    print(f"People's Priorities API running at http://{display_host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info"))
