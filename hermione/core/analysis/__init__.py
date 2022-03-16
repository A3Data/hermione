from ._cluster import Clusterer
from ._dimensionality_reduction import DimensionalityReducer
from ._feature_selection import FeatureSelector
from ._hypothesis_autopilot import HTestAutoPilot
from ._hypothesis_testing import HypothesisTester
from ._pca import PCAReducer
from ._vif import VIF


__all__ = [
    "Clusterer",
    "DimensionalityReducer",
    "FeatureSelector",
    "HTestAutoPilot",
    "HypothesisTester",
    "PCAReducer",
    "VIF",
]


try:
    from ._autoenconder import Autoencoder

    __all__.append("Autoencoder")
except (ModuleNotFoundError, ImportError):
    pass
