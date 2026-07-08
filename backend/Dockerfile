FROM python:3.11-slim

# Set the main working directory inside the server
WORKDIR /app

# Copy your ENTIRE GitHub repository (both backend and frontend) into the server
COPY . .

# Install your Python packages (pointing it to where requirements.txt actually lives)
RUN pip install --no-cache-dir -r backend/requirements.txt

# Shift the server's focus into the backend folder so it can find main.py
WORKDIR /app/backend

# Set up the network ports for Render
ENV PORT=8080
EXPOSE 8080

# Start the application
CMD ["python", "main.py"]