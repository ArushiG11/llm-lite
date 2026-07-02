FROM python:3.11-slim

WORKDIR /app

# Install deps first so Docker layer cache survives code-only changes.
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Model code lives at repo root; API code under api/.
COPY model_causal.py .
COPY src/ src/
COPY api/ api/

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
