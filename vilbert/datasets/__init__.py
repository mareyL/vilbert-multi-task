# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .concept_cap_dataset import (
    ConceptCapLoaderTrain,
    ConceptCapLoaderVal,
    ConceptCapLoaderRetrieval,
)
from .me_dataset import MERegressionDataset

# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal
__all__ = [
    "MERegressionDataset"
]

DatasetMapTrain ={
    "ME": MERegressionDataset,
}


DatasetMapEval = {
    "ME": MERegressionDataset,
}
