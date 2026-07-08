from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LOCAL_DEMOGRAPHICS_CSV = BASE_DIR / "data" / "demographics_sample.csv"

Category = Literal[
    "roads", "water", "sanitation", "health", "education",
    "safety", "electricity", "housing", "environment", "other",
]

TriageCategory = Literal["quick_fix", "urgent_infrastructure", "long_term_planning"]


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class GeoPoint(BaseModel):
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lng: Optional[float] = Field(default=None, ge=-180, le=180)


class FormalizationResult(BaseModel):
    """
    Step 1 output: a cleaned, translated, profanity-free, slang-free version
    of the raw citizen complaint, suitable for a government record.
    """
    formal_description: str = Field(
        description=(
            "A professional, formal English description of the civic issue. "
            "ALL profanity and offensive language must be removed and replaced with "
            "neutral factual language. ALL slang, colloquialisms, abbreviations, and "
            "emotional overstatements must be converted to standard English. "
            "ALL regional Indian languages must be translated to English. "
            "The factual civic complaint — location, problem type, severity — must be fully preserved."
        )
    )
    detected_language: str = Field(
        description=(
            "ISO 639-1 language code of the primary language used in the original submission. "
            "Examples: 'en' (English), 'hi' (Hindi), 'bn' (Bengali), 'ta' (Tamil), "
            "'te' (Telugu), 'mr' (Marathi), 'pa' (Punjabi), 'ml' (Malayalam), "
            "'gu' (Gujarati), 'kn' (Kannada), 'ur' (Urdu), 'or' (Odia)."
        )
    )
    profanity_detected: bool = Field(
        description=(
            "Set to true if the original text contained ANY profanity, swear words, "
            "derogatory terms, or offensive language in any language or script."
        )
    )
    cleaned_summary: str = Field(
        description=(
            "A single clear sentence summarizing the civic issue in professional English, "
            "suitable as a headline for a government record."
        )
    )


class DeduplicationResult(BaseModel):
    """
    Step 2 output: whether this new submission is a duplicate of an existing active issue.
    Only mark as duplicate when you are highly confident (>= 0.85).
    """
    is_duplicate: bool = Field(
        description=(
            "True ONLY if the new submission describes the EXACT SAME real-world problem "
            "at the SAME physical location as an existing active issue. "
            "Set to false if the location differs, the problem type differs, or you are uncertain. "
            "When in doubt, set is_duplicate=false to avoid false merges."
        )
    )
    master_id: Optional[str] = Field(
        default=None,
        description=(
            "The ID of the matching existing master issue. "
            "Required (non-null) when is_duplicate is true. Must be null when is_duplicate is false."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Your confidence in this decision, from 0.0 (completely uncertain) to 1.0 (certain). "
            "Only set is_duplicate=true if confidence >= 0.85."
        ),
    )
    reasoning: str = Field(
        description="A brief, specific explanation of why this is or is not a duplicate.",
    )


class SubmissionAnalysis(BaseModel):
    """Step 3 output: full structured civic issue analysis for a unique new issue."""

    summary: str = Field(description="One sentence citizen-friendly issue summary.")
    category: Category
    triage_category: TriageCategory = Field(
        description=(
            "Classify into exactly one of three triage buckets:\n"
            "• quick_fix — Minor, fast-resolvable issues needing minimal budget and time "
            "(potholes, broken streetlights, garbage collection, broken benches, "
            "minor drainage blocks, missing road signs, damaged footpaths).\n"
            "• urgent_infrastructure — Safety-critical or health-critical issues requiring "
            "immediate escalation (bridge damage or collapse risk, hospital/clinic failures, "
            "major power grid outages, sewage overflow into streets, severe flooding, "
            "structural building collapse, water supply contamination).\n"
            "• long_term_planning — Issues requiring formal planning, budget approval, "
            "and multi-month execution (building new roads, schools, parks, community centers, "
            "major drainage networks, new highways, public transit infrastructure, "
            "large-scale electrification projects)."
        )
    )
    urgency_score: int = Field(ge=1, le=10, description="10 = imminent danger requiring immediate action.")
    sentiment: Literal["negative", "neutral", "positive"]
    suggested_department: str
    constituency_priority: str = Field(
        description="Why this issue matters for the constituency development plan."
    )
    keywords: List[str] = Field(default_factory=list, max_length=8)
    location_hint: Optional[str] = Field(
        default=None,
        description="Named place or landmark mentioned by the citizen. Leave null if not mentioned.",
    )
    geocode: Optional[GeoPoint] = None
    rationale: str = Field(
        description="Brief evidence for the category, triage_category, and urgency_score assignments."
    )


# ─── System Prompts ───────────────────────────────────────────────────────────

FORMALIZATION_SYSTEM = """\
You are a civic issue formalization engine for an Indian constituency management platform.

Your SOLE job is to convert raw citizen complaints into clean, formal, government-record-ready English.

The raw input may contain ANY combination of the following — handle ALL of them:

═══ LANGUAGE ISSUES ════════════════════════════════════════════════════════════
• Regional Indian languages in native script:
  Hindi (Devanagari), Bengali, Tamil, Telugu, Marathi, Punjabi (Gurmukhi),
  Malayalam, Gujarati, Odia, Kannada, Assamese, Urdu, Rajasthani, Bhojpuri, etc.
• Transliteration and code-switching:
  Hinglish, Banglish, Tanglish, Manglish, mixed-script text, Roman-script regional languages
• Mixed multilingual sentences (e.g., English structure with Hindi words)
→ TRANSLATE ALL non-English content to formal English.

═══ LANGUAGE QUALITY ISSUES ════════════════════════════════════════════════════
• Slang, colloquial expressions, street language (e.g., "jugaad", "bakwaas", "ekdum kharab")
• Informal abbreviations (ASAP, FYI, etc. in unprofessional context)
• Emotional overstatements ("This road is literally killing us!", "Worst thing ever!")
• Repetition and filler words ("yaar", "bhai", "abey", etc.)
→ Convert to factual, professional, civic complaint language suitable for government records.

═══ PROFANITY AND OFFENSIVE CONTENT ════════════════════════════════════════════
• Swear words and profanity in ANY language or script
• Derogatory or abusive terms directed at officials, groups, or infrastructure
• Vulgar or sexually explicit language
• Hate speech or discriminatory content
→ REMOVE completely. Replace with neutral factual descriptions of the civic issue.

EXAMPLES OF PROFANITY HANDLING:
  Input:  "Ye haramkhor neta ka road bilkul ch**iya bana rakha hai"
  Output: "The road in this area has been severely neglected and requires urgent repair."

  Input:  "This f***ing pothole broke my bike yesterday near the market!!"
  Output: "A large pothole near the local market caused vehicle damage and requires immediate repair."

  Input:  "ভাঙা রাস্তা, শালা সরকার কিচ্ছু করে না, গাড়ি চলতেই পারে না"
  Output: "The road is severely damaged, preventing vehicles from passing. Government action is urgently needed."

═══ PRESERVATION RULES ══════════════════════════════════════════════════════════
• ALWAYS preserve the factual civic details: location, problem type, severity, impact
• DO NOT add information not present in the original
• DO NOT minimize or exaggerate the severity
• Set profanity_detected=true if ANY profanity or offensive language was present in the original
• Detect and report the primary language code used

Output ONLY the structured JSON. No preamble, no explanation.
"""

ANALYSIS_SYSTEM = """\
You are an AI assistant for constituency development planning in India.

Analyze citizen civic issue submissions (text and/or attached photos) and produce a complete structured analysis.

═══ TRIAGE CATEGORIES — Assign exactly one ═══════════════════════════════════
• quick_fix: Issues resolvable within days to a few weeks with routine budget allocation.
  Examples: potholes, broken streetlights, overflowing garbage bins, broken benches,
  blocked roadside drains, missing road signs, damaged footpaths, waterlogging on minor roads.

• urgent_infrastructure: Safety-critical or health-critical issues requiring immediate escalation.
  Examples: damaged or unstable bridge, hospital/PHC closure or critical shortage,
  major power grid failure (entire colony/ward), raw sewage overflow on streets,
  severe flooding blocking access routes, structural building collapse risk,
  drinking water contamination.

• long_term_planning: Issues requiring formal planning, budget sanction, and multi-month execution.
  Examples: constructing a new road or highway, building a school or community center,
  new public park, major drainage network installation, electrification of new areas,
  public transit infrastructure, large water supply projects.

═══ URGENCY SCORING (1–10) ═══════════════════════════════════════════════════
• 9–10: Imminent danger to human life or health. Hours matter.
• 7–8:  Significant disruption, active safety risk. Days matter.
• 5–6:  Moderate impact on daily life, needs action within weeks.
• 3–4:  Low impact, inconvenience, can wait months.
• 1–2:  Minor cosmetic or convenience issue with minimal impact.

═══ LOCATION ═════════════════════════════════════════════════════════════════
• If the citizen mentions a specific location, capture it in location_hint.
• Only populate geocode coordinates if you have strong geographic knowledge.
• Otherwise leave geocode null.

Output ONLY the structured JSON. No preamble.
"""


# ─── Gemini Client ────────────────────────────────────────────────────────────

def get_client() -> genai.Client:
    """Local hackathon demo client using only the Gemini Developer API key from .env."""
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _media_part(media: Dict[str, Any]) -> types.Part:
    mime_type = media.get("mime_type") or "application/octet-stream"
    return types.Part.from_bytes(data=media["bytes"], mime_type=mime_type)


# ─── Response Coercers ────────────────────────────────────────────────────────

def _coerce_formalization(response: Any) -> FormalizationResult:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, FormalizationResult):
        return parsed
    if isinstance(parsed, dict):
        return FormalizationResult.model_validate(parsed)
    text = getattr(response, "text", None) or ""
    try:
        return FormalizationResult.model_validate_json(text)
    except (ValidationError, ValueError):
        return FormalizationResult.model_validate(json.loads(text))


def _coerce_deduplication(response: Any) -> DeduplicationResult:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DeduplicationResult):
        return parsed
    if isinstance(parsed, dict):
        return DeduplicationResult.model_validate(parsed)
    text = getattr(response, "text", None) or ""
    try:
        return DeduplicationResult.model_validate_json(text)
    except (ValidationError, ValueError):
        return DeduplicationResult.model_validate(json.loads(text))


def _coerce_analysis(response: Any) -> Dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, SubmissionAnalysis):
        return parsed.model_dump(mode="json")
    if isinstance(parsed, dict):
        return SubmissionAnalysis.model_validate(parsed).model_dump(mode="json")
    text = getattr(response, "text", None) or ""
    try:
        return SubmissionAnalysis.model_validate_json(text).model_dump(mode="json")
    except (ValidationError, ValueError):
        return SubmissionAnalysis.model_validate(json.loads(text)).model_dump(mode="json")


# ─── Step 1: Formalization ────────────────────────────────────────────────────

def formalize_submission(
    raw_text: str,
    media: Optional[List[Dict[str, Any]]] = None,
) -> FormalizationResult:
    """
    Step 1 of the V2.0 intake pipeline.

    Converts raw citizen input into clean formal English by:
    - Translating regional Indian languages (Hindi, Bengali, Tamil, etc.)
    - Removing profanity and offensive language, replacing with neutral factual text
    - Scrubbing slang, transliteration, colloquialisms, and emotional overstatements
    - Detecting the source language

    Never raises — falls back to a safe default on any Gemini failure so the
    intake pipeline never halts and no citizen report is ever lost.
    """
    media = media or []
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = (
        "Process the following raw citizen complaint and return a formalized version.\n\n"
        "RAW CITIZEN INPUT:\n"
        "'''\n"
        f"{raw_text or '[No text provided. Analyze the civic issue shown in the attached photo/media and describe it formally.]'}\n"
        "'''\n\n"
        "Apply ALL formalization rules:\n"
        "1. Translate to English if in a regional language\n"
        "2. Remove all profanity and offensive language (replace with neutral civic language)\n"
        "3. Convert slang and transliteration to professional English\n"
        "4. Preserve all factual civic details (location, problem, severity)\n"
        "5. Detect and report the original language code\n"
        "6. Set profanity_detected appropriately\n"
    )

    parts: List[types.Part] = [types.Part.from_text(text=prompt)]
    parts.extend(_media_part(m) for m in media if m.get("bytes"))

    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=FORMALIZATION_SYSTEM,
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=FormalizationResult,
            ),
        )
        return _coerce_formalization(response)
    except Exception:
        # Graceful fallback: preserve raw text, mark as unprocessed
        summary = (raw_text.strip()[:160] if raw_text.strip()
                   else "A civic issue has been reported via submitted media.")
        return FormalizationResult(
            formal_description=raw_text.strip() or "A civic issue has been reported via submitted media.",
            detected_language="en",
            profanity_detected=False,
            cleaned_summary=summary,
        )


# ─── Step 2: Deduplication ────────────────────────────────────────────────────

def check_deduplication(
    formal_description: str,
    address: str,
    existing_issues: List[Dict[str, Any]],
) -> DeduplicationResult:
    """
    Step 2 of the V2.0 intake pipeline.

    Compares the new issue's formal_description and address against up to 20 existing
    active issues using Gemini. Returns a structured duplicate decision.

    Confidence gate: only accepts duplicate decisions with confidence >= 0.85 to
    prevent false merges. Any confidence below this threshold forces a new issue.

    Never raises — always returns DeduplicationResult(is_duplicate=False) on any
    failure so no citizen report is silently dropped.
    """
    if not existing_issues:
        return DeduplicationResult(
            is_duplicate=False,
            master_id=None,
            confidence=1.0,
            reasoning="No existing active issues to compare against — creating a new issue.",
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Build a compact comparison list (only fields Gemini needs, capped at 20)
    comparison_list = [
        {
            "id": str(issue.get("id", "")),
            "description": (
                issue.get("formal_description") or issue.get("text", "")
            )[:250],
            "address": str(issue.get("address") or "")[:120],
            "category": str(issue.get("category", "")),
            "status": str(issue.get("status", "")),
        }
        for issue in existing_issues
        if not issue.get("is_archived")
    ][:20]

    prompt = (
        "Determine whether the following NEW civic issue submission is a duplicate of any existing active issue.\n\n"
        f"NEW SUBMISSION:\n"
        f"  Description: {formal_description[:350]}\n"
        f"  Address: {address or 'Not specified'}\n\n"
        f"EXISTING ACTIVE ISSUES:\n{json.dumps(comparison_list, ensure_ascii=True, indent=2)}\n\n"
        "DEDUPLICATION RULES:\n"
        "• Mark is_duplicate=true ONLY if the new submission describes the EXACT SAME real-world "
        "problem at the SAME physical location as an existing issue.\n"
        "• The same type of problem at a DIFFERENT location is NOT a duplicate.\n"
        "• A broad category match (e.g., both about potholes) at different streets is NOT a duplicate.\n"
        "• If both the problem type AND location strongly overlap, and you are >= 85% confident, "
        "set is_duplicate=true and provide the matching master_id.\n"
        "• When in doubt, set is_duplicate=false — it is safer to create a new issue than to "
        "incorrectly merge two different real-world problems.\n"
    )

    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=DeduplicationResult,
            ),
        )
        result = _coerce_deduplication(response)

        # Enforce minimum confidence threshold — reject ambiguous duplicates
        if result.is_duplicate and result.confidence < 0.85:
            return DeduplicationResult(
                is_duplicate=False,
                master_id=None,
                confidence=result.confidence,
                reasoning=(
                    f"Possible match found but confidence {result.confidence:.2f} is below "
                    "the required 0.85 threshold. Creating a new issue to prevent false merges."
                ),
            )
        return result

    except Exception as exc:
        # Safety fallback: always treat as new issue — never silently drop a report
        return DeduplicationResult(
            is_duplicate=False,
            master_id=None,
            confidence=0.0,
            reasoning=f"Deduplication check failed ({exc!s}). Treating as new issue to prevent any data loss.",
        )


# ─── Step 3: Full Analysis ────────────────────────────────────────────────────

def _local_fallback_analysis(text: str) -> Dict[str, Any]:
    """
    Keyword-based triage fallback when Gemini is unavailable.
    Activated only when ENABLE_LOCAL_AI_FALLBACK=true is set in .env.
    """
    lowered = text.lower()
    category: Category
    triage: TriageCategory

    if any(w in lowered for w in ["sewage", "sewer", "sewage overflow", "open drain overflow"]):
        category, triage = "sanitation", "urgent_infrastructure"
    elif any(w in lowered for w in ["water", "pipeline", "tap", "leakage", "water supply"]):
        category, triage = "water", "quick_fix"
    elif any(w in lowered for w in ["garbage", "waste", "litter", "trash", "bins", "dump"]):
        category, triage = "sanitation", "quick_fix"
    elif any(w in lowered for w in ["pothole", "road surface", "broken road", "damaged road"]):
        category, triage = "roads", "quick_fix"
    elif any(w in lowered for w in ["bridge", "overpass", "flyover", "underpass"]):
        category, triage = "roads", "urgent_infrastructure"
    elif any(w in lowered for w in ["new road", "new highway", "road construction", "road building"]):
        category, triage = "roads", "long_term_planning"
    elif any(w in lowered for w in ["road", "traffic", "street", "footpath", "pavement"]):
        category, triage = "roads", "quick_fix"
    elif any(w in lowered for w in ["streetlight", "street light", "lamp post"]):
        category, triage = "electricity", "quick_fix"
    elif any(w in lowered for w in ["power cut", "blackout", "power failure", "electricity", "grid"]):
        category, triage = "electricity", "urgent_infrastructure"
    elif any(w in lowered for w in ["hospital", "clinic", "phc", "doctor", "health centre", "ambulance"]):
        category, triage = "health", "urgent_infrastructure"
    elif any(w in lowered for w in ["new school", "build school", "school construction"]):
        category, triage = "education", "long_term_planning"
    elif any(w in lowered for w in ["school", "teacher", "classroom", "college", "education"]):
        category, triage = "education", "quick_fix"
    elif any(w in lowered for w in ["park", "garden", "playground", "community centre"]):
        category, triage = "environment", "long_term_planning"
    elif any(w in lowered for w in ["flood", "waterlogging", "drainage"]):
        category, triage = "water", "urgent_infrastructure"
    else:
        category, triage = "other", "quick_fix"

    urgent_words = ["urgent", "danger", "accident", "emergency", "collapse", "fire", "death",
                    "bleeding", "critical", "explosion", "gas leak"]
    urgency = 8 if any(w in lowered for w in urgent_words) else 5
    if urgency >= 7 and triage == "quick_fix":
        triage = "urgent_infrastructure"

    summary = text.strip()[:180] or "Citizen submitted a media-only civic issue."
    return SubmissionAnalysis(
        summary=summary,
        category=category,
        triage_category=triage,
        urgency_score=urgency,
        sentiment="negative" if urgency >= 7 else "neutral",
        suggested_department="Constituency development office",
        constituency_priority="Requires triage against demographic vulnerability and citizen demand.",
        keywords=[category, triage, "citizen-report"],
        location_hint=None,
        geocode=None,
        rationale="Generated by local keyword fallback because Gemini was unavailable.",
    ).model_dump(mode="json")


def analyze_submission(
    text: str = "",
    media: Optional[List[Dict[str, Any]]] = None,
    demographics_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Step 3 of the V2.0 intake pipeline (unique new issues only).

    Performs full AI analysis: category, triage_category, urgency_score, department,
    constituency_priority, keywords, geocode, and all other structured fields.

    `text` should be the already-formalized formal_description from Step 1.
    Media entries: {"bytes": b"...", "mime_type": "image/jpeg", "path": "static/uploads/f.jpg"}
    """
    media = media or []
    demographics_context = demographics_context or []
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = (
        "Analyze this civic issue submission and return a complete structured analysis.\n\n"
        f"ISSUE DESCRIPTION (formal English):\n{text or '[No text — analyze the attached photo/media.]'}\n\n"
        f"CONSTITUENCY DEMOGRAPHIC CONTEXT:\n{json.dumps(demographics_context[:5], ensure_ascii=True)}\n\n"
        "Assign the correct category, triage_category, urgency_score, and all other required fields."
    )

    parts: List[types.Part] = [types.Part.from_text(text=prompt)]
    parts.extend(_media_part(m) for m in media if m.get("bytes"))

    try:
        client = get_client()
        response = client.models.generate_content(
            model=model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=ANALYSIS_SYSTEM,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=SubmissionAnalysis,
            ),
        )
        return _coerce_analysis(response)
    except Exception:
        if os.getenv("ENABLE_LOCAL_AI_FALLBACK", "false").lower() == "true":
            return _local_fallback_analysis(text)
        raise


# ─── Demographics ─────────────────────────────────────────────────────────────

DEFAULT_DEMOGRAPHICS: List[Dict[str, Any]] = [
    {
        "ward_id": "unknown",
        "ward_name": "Unknown Ward",
        "population": 0,
        "households": 0,
        "vulnerability_index": 0,
        "water_access_pct": None,
        "sanitation_access_pct": None,
        "health_facility_count": None,
    }
]


def _load_demographics_csv(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return DEFAULT_DEMOGRAPHICS
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_demographics() -> Dict[str, Any]:
    """
    Load constituency demographics strictly from the local CSV.
    Never attempts BigQuery or any network connection.
    """
    return {
        "source": "csv_local",
        "path": str(LOCAL_DEMOGRAPHICS_CSV),
        "rows": _load_demographics_csv(LOCAL_DEMOGRAPHICS_CSV),
    }


def fetch_demographics_from_bigquery(
    constituency_id: Optional[str] = None,
    fallback_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compatibility wrapper for the legacy function name.
    All demographic data comes strictly from the local CSV — no cloud calls.
    """
    del constituency_id, fallback_csv_path
    return load_demographics()
