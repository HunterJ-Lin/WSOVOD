# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import operator
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torchdata
from detectron2.config import configurable
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import (AspectRatioGroupedDataset, DatasetFromList,
                                    MapDataset, ToIterableDataset)
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import (InferenceSampler,
                                      RepeatFactorTrainingSampler,
                                      TrainingSampler)
from detectron2.structures import BoxMode
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import _log_api_usage, log_first_n
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size
from tabulate import tabulate
from termcolor import colored

from .common import ClassAspectRatioGroupedDataset
from .dataset_mapper import DatasetMapper

"""
This file contains the default logic to build a dataloader for training or testing.
"""

__all__ = [
    "build_batch_data_loader",
    "build_detection_train_loader",
    "build_detection_test_loader",
    "get_detection_dataset_dicts",
    "load_proposals_into_dataset",
    "print_instances_class_histogram",
]


def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get("iscrowd", 0) == 0:
                return True
        return False

    dataset_dicts = [x for x in dataset_dicts if valid(x["annotations"])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with no usable annotations. {} images left.".format(
            num_before - num_after, num_after
        )
    )
    return dataset_dicts


def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
        annotations = dic["annotations"]
        return sum(
            (np.array(ann["keypoints"][2::3]) > 0).sum()
            for ann in annotations
            if "keypoints" in ann
        )

    dataset_dicts = [
        x for x in dataset_dicts if visible_keypoints_in_image(x) >= min_keypoints_per_image
    ]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info(
        "Removed {} images with fewer than {} keypoints.".format(
            num_before - num_after, min_keypoints_per_image
        )
    )
    return dataset_dicts


def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info("Loading proposals from: {}".format(proposal_file))

    if proposal_file=='':
        return dataset_dicts

    if Path(proposal_file).is_dir():
        for record in dataset_dicts:
            image_id = str(record["image_id"])
            record["proposal_file"] = proposal_file + "/" + image_id + ".pkl"

        return dataset_dicts

    with PathManager.open(proposal_file, "rb") as f:
        proposals = pickle.load(f, encoding="latin1")

    # Rename the key names in D1 proposal files
    rename_keys = {"indexes": "ids", "scores": "objectness_logits"}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)

    # Fetch the indexes of all proposals that are in the dataset
    # Convert image_id to str since they could be int.
    img_ids = set({str(record["image_id"]) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals["ids"]) if str(id) in img_ids}

    # Assuming default bbox_mode of precomputed proposals are 'XYXY_ABS'
    bbox_mode = BoxMode(proposals["bbox_mode"]) if "bbox_mode" in proposals else BoxMode.XYXY_ABS

    for record in dataset_dicts:
        # Get the index of the proposal
        i = id_to_index[str(record["image_id"])]

        boxes = proposals["boxes"][i]
        objectness_logits = proposals["objectness_logits"][i]
        # Sort the proposals in descending order of the scores
        inds = objectness_logits.argsort()[::-1]
        record["proposal_boxes"] = boxes[inds]
        record["proposal_objectness_logits"] = objectness_logits[inds]
        record["proposal_bbox_mode"] = bbox_mode

    return dataset_dicts


def print_instances_class_histogram(dataset_dicts, class_names):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    log_first_n(
        logging.INFO,
        "Distribution of instances among all {} categories:\n".format(num_classes)
        + colored(table, "cyan"),
        key="message",
    )


def get_detection_dataset_dicts(
    names,
    filter_empty=True,
    min_keypoints=0,
    proposal_files=None,
    check_consistency=True,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None and len(proposal_files) > 0:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        return torchdata.ConcatDataset(dataset_dicts)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    has_instances = "annotations" in dataset_dicts[0]
    if filter_empty and has_instances:
        dataset_dicts = filter_images_with_only_crowd_annotations(dataset_dicts)
    if min_keypoints > 0 and has_instances:
        dataset_dicts = filter_images_with_few_keypoints(dataset_dicts, min_keypoints)

    if check_consistency and has_instances:
        try:
            class_names = MetadataCatalog.get(names[0]).thing_classes
            check_metadata_consistency("thing_classes", names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    aspect_ratio_grouping=False,
    class_aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    if aspect_ratio_grouping:
        data_loader = torchdata.DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=operator.itemgetter(0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
            persistent_workers=True,
        )  # yield individual mapped dict
        if class_aspect_ratio_grouping:
            data_loader = ClassAspectRatioGroupedDataset(data_loader, batch_size)
            if collate_fn is None:
                return data_loader
            return MapDataset(data_loader, collate_fn)
        data_loader = AspectRatioGroupedDataset(data_loader, batch_size)
        if collate_fn is None:
            return data_loader
        return MapDataset(data_loader, collate_fn)
    else:
        return torchdata.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
            worker_init_fn=worker_init_reset_seed,
            persistent_workers=True,
        )


def _train_loader_from_config(cfg, mapper=None, *, dataset=None, sampler=None):
    if dataset is None:
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "RepeatFactorTrainingSampler":
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset, cfg.DATALOADER.REPEAT_THRESHOLD
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "sampler": sampler,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "class_aspect_ratio_grouping": cfg.DATALOADER.CLASS_ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
    }


@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset,
    *,
    mapper,
    sampler=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    class_aspect_ratio_grouping=False,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a dataloader for object detection with some default features.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers. Batching
            simply puts data into a list.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.
            No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        aspect_ratio_grouping=aspect_ratio_grouping,
        class_aspect_ratio_grouping=class_aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _test_loader_from_config(cfg, dataset_name, mapper=None):
    """
    Uses the given `dataset_name` argument (instead of the names in cfg), because the
    standard practice is to evaluate each test set individually (not combining them).
    """
    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    dataset = get_detection_dataset_dicts(
        dataset_name,
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    return {"dataset": dataset, "mapper": mapper, "num_workers": cfg.DATALOADER.NUM_WORKERS}


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(dataset, *, mapper, sampler=None, num_workers=0, collate_fn=None):
    """
    Similar to `build_detection_train_loader`, but uses a batch size of 1,
    and :class:`InferenceSampler`. This sampler coordinates all workers to
    produce the exact set of all samples.
    This interface is experimental.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). They can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset
           and returns the format to be consumed by the model.
           When using cfg, the default choice is ``DatasetMapper(cfg, is_train=False)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``. Default to :class:`InferenceSampler`,
            which splits the dataset across all workers. Sampler must be None
            if `dataset` is iterable.
        num_workers (int): number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.

    Examples:
    ::
        data_loader = build_detection_test_loader(
            DatasetRegistry.get("my_test"),
            mapper=DatasetMapper(...))

        # or, instantiate with a CfgNode:
        data_loader = build_detection_test_loader(cfg, "my_test")
    """
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    return torchdata.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        persistent_workers=True,
    )



## --------------------------------- Multi Dataset --------------------------------- ##
def DatasetCatalog_get(dataset_name, reduce_memory, reduce_memory_size):
    import os, psutil

    logger = logging.getLogger(__name__)
    logger.info(
        "Current memory usage: {} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )

    dataset_dicts = DatasetCatalog.get(dataset_name)

    # logger.info(
    #     "Current memory usage: {} GB".format(
    #         psutil.Process(os.getpid()).memory_info().rss / 1024**3
    #     )
    # )
    # logger.info("Reducing memory usage...")

    # for d in dataset_dicts:
    #     # LVIS
    #     if "not_exhaustive_category_ids" in d.keys():
    #         del d["not_exhaustive_category_ids"]
    #     if "neg_category_ids" in d.keys():
    #         del d["neg_category_ids"]
    #     if "pos_category_ids" in d.keys():
    #         del d["pos_category_ids"]

    #     if "annotations" not in d.keys():
    #         continue
    #     for anno in d["annotations"]:
    #         if "iscrowd" in anno.keys():
    #             if anno["iscrowd"] == 0:
    #                 del anno["iscrowd"]

    logger.info(
        "Current memory usage: {} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )

    if not reduce_memory:
        return dataset_dicts
    if len(dataset_dicts) < reduce_memory_size:
        return dataset_dicts

    logger.info("Reducing memory usage further...")

    for d in dataset_dicts:
        if "annotations" not in d.keys():
            continue

        for anno in d["annotations"]:

            if "bbox" in anno.keys():
                del anno["bbox"]

            if "bbox_mode" in anno.keys():
                del anno["bbox_mode"]

            if "segmentation" in anno.keys():
                del anno["segmentation"]

            if "phrase" in anno.keys():
                del anno["phrase"]

    logger.info(
        "Current memory usage: {} GB".format(
            psutil.Process(os.getpid()).memory_info().rss / 1024**3
        )
    )

    return dataset_dicts


def get_detection_dataset_dicts_multi_dataset(
    names,
    proposal_files=None,
    check_consistency=True,
    filter_emptys=[True],
    dataloader_id=None,
    reduce_memory=False,
    reduce_memory_size=1e6,
):
    """
    Load and prepare dataset dicts for instance detection/segmentation and semantic segmentation.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        filter_empty (bool): whether to filter out images without instance annotations
        min_keypoints (int): filter out images with fewer keypoints than
            `min_keypoints`. Set to 0 to do nothing.
        proposal_files (list[str]): if given, a list of object proposal files
            that match each dataset in `names`.
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    # dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]
    dataset_dicts = [
        DatasetCatalog_get(dataset_name, reduce_memory, reduce_memory_size)
        for dataset_name in names
    ]

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    if proposal_files is not None and len(proposal_files) > 0:
        assert len(names) == len(proposal_files)
        # load precomputed proposals from proposal files
        dataset_dicts = [
            load_proposals_into_dataset(dataset_i_dicts, proposal_file)
            for dataset_i_dicts, proposal_file in zip(dataset_dicts, proposal_files)
        ]

    for dataset_id, (dataset_name, dicts) in enumerate(
        zip(names, dataset_dicts)
    ):
        for d in dicts:
            d["dataset_id"] = dataset_id
            if dataloader_id is not None:
                d["dataloader_id"] = dataloader_id

        has_instances = "annotations" in dicts[0]
        if not check_consistency or not has_instances:
            continue
        try:
            class_names = MetadataCatalog.get(dataset_name).thing_classes
            check_metadata_consistency("thing_classes", [dataset_name])
            print_instances_class_histogram(dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass

    dataset_dicts = [
        filter_images_with_only_crowd_annotations(dicts)
        if flag and "annotations" in dicts[0]
        else dicts
        for dicts, flag in zip(dataset_dicts, filter_emptys)
    ]

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    return dataset_dicts


def _train_loader_from_config_multi_dataset(cfg, mapper=None, *, dataset=None, sampler=None):
    seed1 = comm.shared_random_seed()
    seed2 = comm.shared_random_seed()
    seed3 = comm.shared_random_seed()
    seed4 = comm.shared_random_seed()
    logger = logging.getLogger(__name__)
    logger.info("rank {} seed1 {} seed2 {}".format(comm.get_local_rank(), seed1, seed2))
    logger.info("rank {} seed3 {} seed4 {}".format(comm.get_local_rank(), seed3, seed4))

    # Hard-coded 2 sequent group and 1200s time wait.
    wait_group = 2
    wait_time = cfg.DATALOADER.GROUP_WAIT
    wait = comm.get_local_rank() % wait_group * wait_time
    logger.info("rank {} _train_loader_from_config sleep {}".format(comm.get_local_rank(), wait))
    time.sleep(wait)

    if dataset is None:
        dataset = get_detection_dataset_dicts_multi_dataset(
            cfg.DATASETS.TRAIN,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            filter_emptys=cfg.MULTI_DATASET.FILTER_EMPTY_ANNOTATIONS,
        )
        _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])

    if mapper is None:
       mapper = DatasetMapper(cfg, True)

    if sampler is None:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        if isinstance(dataset, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler = TrainingSampler(len(dataset), seed=seed1)
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                )
                sampler = RepeatFactorTrainingSampler(repeat_factors, seed=seed1)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler = RandomSubsetTrainingSampler(
                    len(dataset),
                    cfg.DATALOADER.RANDOM_SUBSET_RATIO,
                    seed_shuffle=seed1,
                    seed_subset=seed2,
                )
            elif sampler_name == "MultiDatasetSampler":
                raise ValueError("Despreted training sampler: {}".format(sampler_name))
                sizes = [0 for _ in range(len(cfg.DATASETS.TRAIN))]
                for d in dataset:
                    sizes[d["dataset_id"]] += 1
                sampler = MultiDatasetSampler(cfg, dataset, sizes, seed=seed1)
            elif sampler_name == "MultiDatasetTrainingSampler":
                # sampler = MultiDatasetTrainingSampler(cfg, dataset, seed=seed1)
                repeat_factors = MultiDatasetTrainingSampler.get_repeat_factors(
                    dataset,
                    len(cfg.DATASETS.TRAIN),
                    cfg.MULTI_DATASET.RATIOS,
                    cfg.MULTI_DATASET.USE_RFS,
                    cfg.MULTI_DATASET.USE_CAS,
                    cfg.MULTI_DATASET.REPEAT_THRESHOLD,
                    cfg.MULTI_DATASET.CAS_LAMBDA,
                )
                sampler = MultiDatasetTrainingSampler(repeat_factors, seed=seed1)
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    if True:
        sampler_name = cfg.DATALOADER.COPYPASTE.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        if isinstance(dataset_bg, torchdata.IterableDataset):
            logger.info("Not using any sampler since the dataset is IterableDataset.")
            sampler = None
        else:
            logger.info("Using training sampler {}".format(sampler_name))
            if sampler_name == "TrainingSampler":
                sampler_bg = TrainingSampler(len(dataset_bg), seed=seed3)
            elif sampler_name == "RepeatFactorTrainingSampler":
                repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                    dataset_bg, cfg.DATALOADER.COPYPASTE.REPEAT_THRESHOLD
                )
                sampler_bg = RepeatFactorTrainingSampler(repeat_factors, seed=seed3)
            elif sampler_name == "RandomSubsetTrainingSampler":
                sampler_bg = RandomSubsetTrainingSampler(
                    len(dataset_bg),
                    cfg.DATALOADER.COPYPASTE.RANDOM_SUBSET_RATIO,
                    seed_shuffle=seed3,
                    seed_subset=seed4,
                )
            else:
                raise ValueError("Unknown training sampler: {}".format(sampler_name))

    return {
        "dataset": dataset,
        "dataset_bg": dataset_bg,
        "sampler": sampler,
        "sampler_bg": sampler_bg,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "total_batch_size_list": cfg.SOLVER.IMS_PER_BATCH_LIST,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "num_datasets": len(cfg.DATASETS.TRAIN),
    }


@configurable(from_config=_train_loader_from_config_multi_dataset)
def build_detection_train_loader_multi_dataset(
    dataset,
    dataset_bg,
    *,
    mapper,
    sampler=None,
    sampler_bg=None,
    total_batch_size,
    total_batch_size_list,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    num_datasets=1,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    # wait = round(comm.get_local_rank() * 1.0 * len(dataset) / 60000)
    # logger = logging.getLogger(__name__)
    # logger.info("get_detection_dataset_dicts_multi_dataset sleep {}".format(wait))
    # time.sleep(wait)

    if isinstance(sampler_bg, Callable):
        sampler_bg = sampler_bg(dataset_bg)
    if isinstance(sampler, Callable):
        sampler = sampler(dataset)

    if isinstance(dataset_bg, list):
        dataset_bg = DatasetFromList(dataset_bg, copy=False)

    if isinstance(dataset_bg, torchdata.IterableDataset):
        assert sampler_bg is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler_bg is None:
            sampler_bg = TrainingSampler(len(dataset_bg))
        assert isinstance(
            sampler_bg, torchdata.Sampler
        ), f"Expect a Sampler but got {type(sampler)}"

    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset_coppaste(dataset, mapper, dataset_bg, sampler_bg)

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        if sampler is None:
            sampler = TrainingSampler(len(dataset))
        assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
    return build_batch_data_loader_multi_dataset(
        dataset,
        sampler,
        total_batch_size,
        total_batch_size_list,
        aspect_ratio_grouping=aspect_ratio_grouping,
        num_workers=num_workers,
        collate_fn=collate_fn,
        num_datasets=num_datasets,
    )

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)
