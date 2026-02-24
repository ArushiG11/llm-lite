from pydantic import BaseModel
import os

class Settings(BaseModel):
    # runtime
    env: str = os.getenv("ENV", "dev")
    service_name: str = os.getenv("SERVICE_NAME", "llm-gateway")
    port: int = int(os.getenv("PORT", "8080"))

    # providers
    provider: str = os.getenv("PROVIDER", "openai")  # openai|vertex
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # redis (Memorystore)
    redis_host: str | None = os.getenv("REDIS_HOST")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))

    # cloud sql
    db_dsn: str | None = os.getenv("DB_DSN")  # e.g. postgresql+psycopg://...

    # otel
    otel_exporter_otlp_endpoint: str | None = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

settings = Settings()