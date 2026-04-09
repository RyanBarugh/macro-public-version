from __future__ import annotations

from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any

from .logger import get_logger

logger = get_logger(__name__)


class SeriesLoadError(Exception):
    pass


@dataclass(frozen=True)
class SeriesDef:
    series_id: str
    provider: str | None = None
    dataset: str | None = None
    api_id: str | None = None
    meta: dict[str, Any] | None = None


def _default_series_dir() -> Path:
    return Path(__file__).parent.parent / "series"


def _parse_entries(raw_list: list[dict[str, Any]]) -> list[SeriesDef]:
    out: list[SeriesDef] = []
    for item in raw_list:
        if not isinstance(item, dict):
            raise SeriesLoadError("series entries must be objects")
        series_id = item.get("series_id") or item.get("id") or item.get("name")
        if not series_id or not isinstance(series_id, str):
            raise SeriesLoadError("series entry missing 'series_id' (string)")

        out.append(
            SeriesDef(
                series_id=series_id,
                provider=item.get("provider"),
                dataset=item.get("dataset"),
                api_id=item.get("api_id") or item.get("provider_series_id"),
                meta=item,
            )
        )
    return out


def load_series(
    path: str | None = None,
    currencies: list[str] | None = None,
) -> list[SeriesDef]:
    """
    Load series definitions.

    If `path` is given, load from that single file (legacy behaviour).
    Otherwise, load .json files from the series/ directory.
    If `currencies` is given (e.g. ["usd", "eur"]), only those files are loaded.
    """
    try:
        # ── Legacy single-file mode ──────────────────────────────────
        if path is not None:
            series_path = Path(__file__).parent / path
            logger.info("Loading series catalog from %s", series_path)
            with series_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if isinstance(payload, dict):
                raw_list: list[dict[str, Any]] = []
                for v in payload.values():
                    if not isinstance(v, list):
                        raise SeriesLoadError("series.json dict values must be lists")
                    raw_list.extend(v)
            elif isinstance(payload, list):
                raw_list = payload
            else:
                raise SeriesLoadError("series.json must be a dict or a list")

            out = _parse_entries(raw_list)
            logger.info("Loaded %d series definitions from %s", len(out), series_path)
            return out

        # ── Directory mode (default) ─────────────────────────────────
        series_dir = _default_series_dir()
        if not series_dir.is_dir():
            raise SeriesLoadError(
                f"Series directory not found: {series_dir}. "
                f"Expected a 'series/' folder with per-currency .json files."
            )

        json_files = sorted(series_dir.glob("*.json"))
        if currencies:
            allowed = {c.lower() for c in currencies}
            json_files = [f for f in json_files if f.stem.lower() in allowed]
            if not json_files:
                raise SeriesLoadError(
                    f"No matching files for currencies={currencies} in {series_dir}"
                )
        if not json_files:
            raise SeriesLoadError(f"No .json files found in {series_dir}")

        logger.info(
            "Loading series catalog from %s (%d files)",
            series_dir, len(json_files),
        )

        all_defs: list[SeriesDef] = []
        seen_ids: set[str] = set()

        for fp in json_files:
            with fp.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if not isinstance(payload, list):
                raise SeriesLoadError(f"{fp.name}: expected a JSON array")

            defs = _parse_entries(payload)

            # Deduplicate across files
            for d in defs:
                if d.series_id in seen_ids:
                    logger.warning(
                        "Duplicate series_id=%s in %s — skipping",
                        d.series_id, fp.name,
                    )
                    continue
                seen_ids.add(d.series_id)
                all_defs.append(d)

            logger.info("  %s: %d series", fp.name, len(defs))

        logger.info("Loaded %d series definitions total", len(all_defs))
        return all_defs

    except SeriesLoadError:
        raise
    except Exception as e:
        logger.error("Failed to load series: %s", e, exc_info=True)
        raise SeriesLoadError(f"Could not load series: {e}") from e