
from typing import List
import pandas as pd

from .logger import get_logger

logger = get_logger(__name__)

MAX_DROP_PCT = 5.0 


def census_table_to_dataframe(
    table: List[List[str]],
    series_id: str,
    strict: bool = True,
) -> pd.DataFrame:
    logger.info("Cleaning series_id=%s strict=%s", series_id, strict)

    if not table or len(table) < 2:
        logger.error("Census payload empty or malformed for series_id=%s", series_id)
        raise ValueError("Census payload is empty or malformed.")

    header = table[0]
    rows = table[1:]

    logger.info("Raw rows (excluding header) series_id=%s rows_in=%d", series_id, len(rows))

    df = pd.DataFrame(rows, columns=header)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["value"] = pd.to_numeric(df["cell_value"], errors="coerce")

    out = df[["time", "value"]].copy()
    out.insert(0, "series_id", series_id)

    before = len(out)
    out = out.dropna(subset=["time", "value"]).sort_values("time")
    dropped = before - len(out)

    if dropped:
        drop_pct = (dropped / before) * 100
        bad = df[df["time"].isna() | df["value"].isna()].head(5)
        logger.warning(
            "Dropped %d/%d rows (%.1f%%) for series_id=%s. Sample bad rows:\n%s",
            dropped, before, drop_pct, series_id, bad.to_string()
        )
        if strict and drop_pct > MAX_DROP_PCT:
            raise ValueError(
                f"series_id={series_id}: dropped {drop_pct:.1f}% of rows "
                f"(threshold={MAX_DROP_PCT}%). Possible format change. "
                f"Set strict=False to suppress."
            )

    dupes = out[out["time"].duplicated(keep=False)]
    if not dupes.empty:
        dupe_times = dupes["time"].unique().tolist()
        raise ValueError(
            f"series_id={series_id}: duplicate time values detected: {dupe_times}. "
            f"Census returned overlapping periods."
        )

    logger.info("Cleaned rows series_id=%s rows_out=%d", series_id, len(out))

    out["time"] = out["time"].dt.strftime("%Y-%m-%d")

    return out