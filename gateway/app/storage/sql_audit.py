from sqlalchemy import create_engine, text
from ..config import settings

_engine = None

def engine():
    global _engine
    if _engine is None:
        if not settings.db_dsn:
            return None
        _engine = create_engine(settings.db_dsn, pool_pre_ping=True)
    return _engine

def log_request(request_id: str, model: str, status: str):
    eng = engine()
    if not eng:
        return
    with eng.begin() as conn:
        conn.execute(
            text("insert into inference_audit(request_id, model, status) values (:r,:m,:s)"),
            {"r": request_id, "m": model, "s": status},
        )