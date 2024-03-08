# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import itertools
import logging
import pickle
import random

import numpy as np
import torch.utils.data as data
from detectron2.utils.serialize import PicklableWrapper
from torch.utils.data.sampler import Sampler
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset

__all__ = [
    "ClassAspectRatioGroupedDataset",
    "AspectRatioGroupedMixedDatasets",
]


class ClassAspectRatioGroupedDataset(data.IterableDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2 * 2000)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d["width"], d["height"]
            bucket_id = 0 if w > h else 1

            classes = list(set(d["instances"].gt_classes.tolist()))

            # for c in classes:
            #     bucket = self._buckets[2 * c + bucket_id]
            #     bucket.append(copy.deepcopy(d))
            #     if len(bucket) == self.batch_size:
            #         yield bucket[:]
            #         del bucket[:]
            # continue

            if len(classes) > 0:
                c = random.choice(classes)
            else:
                c = 0

            bucket_id = bucket_id + c * 2

            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

