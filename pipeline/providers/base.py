from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from ..engine.series import SeriesDef


class BaseProvider(ABC):
    """
    Abstract base class for all data providers.
    Each provider implements fetch() and clean().
    The core loop calls these without knowing which provider it's talking to.
    """

    @abstractmethod
    def fetch(
        self,
        series_def: SeriesDef,
        session: requests.Session,
        timeout: Tuple[float, float],
        api_key: Optional[str],
        start: str,
    ) -> Any:
        """
        Fetch raw data from the provider API.
        Returns the raw payload exactly as received — no transformation.
        """
        ...

    @abstractmethod
    def clean(
        self,
        raw_payload: Any,
        series_id: str,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        Transform raw payload into a clean DataFrame.
        Must return columns: series_id, time, value.
        """
        ...