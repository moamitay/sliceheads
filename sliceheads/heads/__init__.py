from sliceheads.heads.base import BaseHead
from sliceheads.heads.pooling import MeanPoolClassifier, MaxPoolClassifier, GeMPoolClassifier
from sliceheads.heads.mil import ABMILClassifier, GatedABMILClassifier, DSMILClassifier
from sliceheads.heads.transformer import TransformerMILClassifier
from sliceheads.heads.timeseries import (
    MultiRocketClassifier,
    InceptionTimeClassifier,
    ALSTMFCNClassifier,
)

__all__ = [
    "BaseHead",
    "MeanPoolClassifier",
    "MaxPoolClassifier",
    "GeMPoolClassifier",
    "ABMILClassifier",
    "GatedABMILClassifier",
    "DSMILClassifier",
    "TransformerMILClassifier",
    "MultiRocketClassifier",
    "InceptionTimeClassifier",
    "ALSTMFCNClassifier",
]
