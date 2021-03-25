from __future__ import absolute_import

from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .triplet import TripletLoss, SoftTripletLoss, NNLoss

__all__ = [
    'CrossEntropyLabelSmooth',
    'SoftEntropy',
    'TripletLoss',
    'SoftTripletLoss',
    'NNLoss'
]
