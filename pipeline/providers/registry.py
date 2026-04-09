from __future__ import annotations

from .base     import BaseProvider
from .bls      import BlsProvider
from .fred     import FredProvider
from .census   import CensusProvider
from .eurostat import EurostatProvider
from .statcan  import StatCanProvider
from .abs      import AbsProvider
from .estat    import EStatProvider
from .fso      import FsoProvider
from .oecd     import OecdProvider
from .scb      import ScbProvider
from .ssb      import SsbProvider
from .ons      import ONSProvider
from .ecb      import EcbProvider
from .snb      import SnbProvider
from .bea      import BeaProvider
from .meti     import MetiProvider
from .meti_iip import MetiIipProvider
from .ecbcs    import EcBcsProvider
from .boe      import BoeProvider
from .rba      import RbaProvider
from .boc      import BocProvider
from .boj      import BojProvider
from .mof      import MofProvider
from .statsnz  import StatsNzCsvProvider
from .rbnz     import RbnzProvider
from .fso_csv  import FsoCsvProvider
from .eodhd    import EodhdProvider
from .bis      import BisProvider

_REGISTRY: dict[str, BaseProvider] = {
    "bls": BlsProvider(),
    "fred": FredProvider(),
    "census": CensusProvider(),
    "ecb": EcbProvider(),
    "eurostat": EurostatProvider(),
    "statcan": StatCanProvider(),
    "abs": AbsProvider(),
    "estat": EStatProvider(),
    "fso": FsoProvider(),
    "oecd": OecdProvider(),
    "scb": ScbProvider(),
    "ssb": SsbProvider(),
    "ons": ONSProvider(),
    "snb": SnbProvider(),
    "bea": BeaProvider(),
    "meti": MetiProvider(),
    "meti_iip": MetiIipProvider(),
    "ecbcs": EcBcsProvider(),
    "boe": BoeProvider(),
    "rba": RbaProvider(),
    "boc": BocProvider(),
    "boj": BojProvider(),
    "mof": MofProvider(),
    "statsnz_csv": StatsNzCsvProvider(),
    "rbnz": RbnzProvider(),
    "fso_csv": FsoCsvProvider(),
    "eodhd": EodhdProvider(),
    "bis": BisProvider(),
}

def get_provider(name: str) -> BaseProvider:
    provider = _REGISTRY.get(name)
    if provider is None:
        raise RuntimeError(
            f"Unknown provider='{name}'. "
            f"Registered providers: {list(_REGISTRY.keys())}"
        )
    return provider