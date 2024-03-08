# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.data.datasets.coco_panoptic import (
    register_coco_panoptic, register_coco_panoptic_separated)
from detectron2.data.datasets.lvis import (get_lvis_instances_meta,
                                           register_lvis_instances)
from detectron2.data.datasets.register_coco import register_coco_instances

from .builtin_meta import _get_builtin_metadata
from .pascal_voc import register_pascal_voc


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
        ("voc_2012_test", "VOC2012", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


# ==== Predefined datasets and splits for ImageNet ==========

_PREDEFINED_SPLITS_ImageNet = {}
_PREDEFINED_SPLITS_ImageNet["imagenet"] = {
    "ilsvrc_2012_val": (
        "ILSVRC2012/val/",
        "ILSVRC2012/ILSVRC2012_img_val_converted.json",
    ),
    "ilsvrc_2012_train": (
        "ILSVRC2012/train/",
        "ILSVRC2012/ILSVRC2012_img_train_converted.json",
    ),
}

def register_all_imagenet(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_ImageNet.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("WSOVOD_DATASETS", "datasets")
    register_all_pascal_voc(_root)
    register_all_imagenet(_root)