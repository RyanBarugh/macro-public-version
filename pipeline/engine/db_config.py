from __future__ import annotations

from typing import Optional

import psycopg2

from .secrets import get_secret
from .logger import get_logger

logger = get_logger(__name__)


def get_db_config(secret_name: str, region: str) -> dict:
    """Fetch DB credentials from Secrets Manager. Never logs passwords."""
    logger.info("Fetching DB config from secret '%s' in region '%s'…", secret_name, region)
    try:
        secret = get_secret(secret_name, region_name=region)

        def pick(*keys: str) -> Optional[str]:
            for k in keys:
                v = secret.get(k)
                if v is not None and v != "":
                    return str(v)
            return None

        host     = pick("host", "hostname", "db_host")
        port     = int(pick("port", "db_port") or 5432)
        dbname   = pick("dbname", "database", "db_name")
        user     = pick("username", "user", "db_user")
        password = pick("password", "db_password")
        sslmode  = pick("sslmode") or "require"

        missing = [k for k, v in [("host", host), ("dbname", dbname),
                                   ("user", user), ("password", password)] if not v]
        if missing:
            raise ValueError(f"DB secret missing keys: {missing}")

        logger.info("DB config loaded for %s@%s:%s/%s", user, host, port, dbname)
        return {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
            "sslmode": sslmode,
        }
    except Exception as e:
        logger.error("Failed to load DB config: %s", e, exc_info=True)
        raise


def open_db_connection(
    db_config,
    timeout_ms: int = 300_000,
    schema_path: str = "macro,equity,prices,public",
) -> psycopg2.extensions.connection:
    """Open a hardened PostgreSQL connection.

    TCP keepalive ensures the connection survives NAT Gateway idle
    timeouts (~350 s) and RDS default tcp_keepalives_idle (300 s).

    Matches TFF/Oanda pipeline pattern exactly.
    """
    # Support both dict (from get_db_config) and DbConfig dataclass
    if hasattr(db_config, "host"):
        host = db_config.host
        port = db_config.port
        dbname = db_config.dbname
        user = db_config.user
        password = db_config.password
    else:
        host = db_config["host"]
        port = db_config["port"]
        dbname = db_config["dbname"]
        user = db_config["user"]
        password = db_config["password"]

    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=10,
        # ── TCP keepalive — prevents silent drops ──────────────────
        keepalives=1,
        keepalives_idle=30,       # first probe after 30s idle
        keepalives_interval=10,   # probe every 10s after that
        keepalives_count=5,       # give up after 5 failed probes
        # ── Session settings ──────────────────────────────────────
        options=f"-c search_path={schema_path} -c statement_timeout={timeout_ms}",
        application_name="edgeflow-macro-pipeline",
    )
    logger.info("DB connection opened (statement_timeout=%ds, keepalive=30s).",
                timeout_ms // 1000)
    return conn