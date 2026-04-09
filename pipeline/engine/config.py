from __future__ import annotations

"""
config.py
=========
Pipeline configuration via frozen dataclass.

Conditional dotenv at module level — Lambda uses native env vars,
local dev loads from .env file.

No S3 configuration — raw storage removed.
Matches TFF/Oanda pipeline pattern.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Only load .env in local dev — Lambda uses native env vars
if not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


@dataclass(frozen=True)
class DbConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str
    sslmode: str = "require"


@dataclass(frozen=True)
class MacroConfig:
    env: str
    region: str
    secrets_manager_secret_id: str
    api_secrets: Dict[str, str]
    db_schema: str = "macro"
    strict_db: bool = True
    lookback_months: int = 24

    def build_db_config(self, secret: Dict[str, Any]) -> DbConfig:
        def pick(*keys: str) -> Optional[str]:
            for k in keys:
                v = secret.get(k)
                if v is not None and v != "":
                    return str(v)
            return None

        host = pick("host", "hostname", "db_host")
        port = int(pick("port", "db_port") or 5432)
        dbname = pick("dbname", "database", "db_name")
        user = pick("username", "user", "db_user")
        password = pick("password", "db_password")
        sslmode = pick("sslmode") or "require"

        missing = [k for k, v in [("host", host), ("dbname", dbname), ("user", user), ("password", password)] if not v]
        if missing:
            raise ValueError(f"DB secret missing keys: {missing}")

        return DbConfig(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
            sslmode=sslmode,
        )


class ConfigError(Exception):
    pass


def load_macro_config() -> MacroConfig:
    api_secrets: Dict[str, str] = {}
    prefix = "API_SECRET__"

    for key, value in os.environ.items():
        if key.startswith(prefix) and value.strip():
            provider = key[len(prefix):].lower()
            api_secrets[provider] = value.strip()

    missing = []
    db_secret_id = os.getenv("DB_SECRET_ID")
    if not db_secret_id:
        missing.append("DB_SECRET_ID")

    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    return MacroConfig(
        env=os.getenv("ENV", "local"),
        region=os.getenv("AWS_REGION", "eu-west-2"),
        secrets_manager_secret_id=db_secret_id,
        api_secrets=api_secrets,
        db_schema=os.getenv("DB_SCHEMA", "macro"),
        strict_db=os.getenv("STRICT_DB", "true").lower() == "true",
        lookback_months=int(os.getenv("LOOKBACK_MONTHS", "24")),
    )