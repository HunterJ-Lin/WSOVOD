# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Note:
For your custom dataset, there is no need to hard-code metadata anywhere in the code.
For example, for COCO-format dataset, metadata will be obtained automatically
when calling `load_coco_json`. For other dataset, metadata may also be obtained in other ways
during loading.

However, we hard-coded metadata for a few common dataset here.
The only goal is to allow users who don't have these dataset to use pre-trained models.
Users don't have to download a COCO json (which contains metadata), in order to visualize a
COCO model (with correct class names and colors).
"""

import numpy as np

# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {"color": [220, 20, 60], "in_voc": 1, "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "in_voc": 1, "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "in_voc": 1, "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "in_voc": 1, "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "in_voc": 1, "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "in_voc": 1, "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "in_voc": 1, "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "in_voc": 0, "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "in_voc": 1, "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "in_voc": 0, "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "in_voc": 0, "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "in_voc": 0, "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "in_voc": 0, "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "in_voc": 0, "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "in_voc": 1, "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "in_voc": 1, "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "in_voc": 1, "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "in_voc": 1, "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "in_voc": 1, "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "in_voc": 1, "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "in_voc": 0, "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "in_voc": 0, "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "in_voc": 0, "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "in_voc": 0, "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "in_voc": 0, "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "in_voc": 0, "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "in_voc": 0, "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "in_voc": 0, "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "in_voc": 0, "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "in_voc": 0, "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "in_voc": 0, "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "in_voc": 0, "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "in_voc": 0, "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "in_voc": 0, "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "in_voc": 0, "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "in_voc": 0, "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "in_voc": 0, "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "in_voc": 0, "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "in_voc": 0, "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "in_voc": 1, "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "in_voc": 0, "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "in_voc": 0, "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "in_voc": 0, "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "in_voc": 0, "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "in_voc": 0, "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "in_voc": 0, "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "in_voc": 0, "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "in_voc": 0, "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "in_voc": 0, "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "in_voc": 0, "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "in_voc": 0, "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "in_voc": 0, "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "in_voc": 0, "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "in_voc": 0, "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "in_voc": 0, "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "in_voc": 0, "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1],"in_voc": 1, "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "in_voc": 1, "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0],"in_voc": 1, "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "in_voc": 0, "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "in_voc": 1, "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "in_voc": 0, "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "in_voc": 1, "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "in_voc": 0, "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "in_voc": 0, "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "in_voc": 0, "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "in_voc": 0, "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "in_voc": 0, "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "in_voc": 0, "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "in_voc": 0, "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "in_voc": 0, "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "in_voc": 0, "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "in_voc": 0, "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "in_voc": 0, "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "in_voc": 0, "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "in_voc": 0, "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "in_voc": 0, "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "in_voc": 0, "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "in_voc": 0, "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "in_voc": 0, "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "in_voc": 0, "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "in_voc": 0, "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "in_voc": 0, "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "in_voc": 0, "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "in_voc": 0, "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "in_voc": 0, "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "in_voc": 0, "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "in_voc": 0, "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "in_voc": 0, "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "in_voc": 0, "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "in_voc": 0, "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "in_voc": 0, "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "in_voc": 0, "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "in_voc": 0, "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "in_voc": 0, "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "in_voc": 0, "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "in_voc": 0, "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "in_voc": 0, "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "in_voc": 0, "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "in_voc": 0, "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "in_voc": 0, "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "in_voc": 0, "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "in_voc": 0, "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "in_voc": 0, "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "in_voc": 0, "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "in_voc": 0, "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "in_voc": 0, "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "in_voc": 0, "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "in_voc": 0, "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "in_voc": 0, "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "in_voc": 0, "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "in_voc": 0, "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "in_voc": 0, "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "in_voc": 0, "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "in_voc": 0, "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "in_voc": 0, "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "in_voc": 0, "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "in_voc": 0, "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "in_voc": 0, "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "in_voc": 0, "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "in_voc": 0, "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "in_voc": 0, "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "in_voc": 0, "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "in_voc": 0, "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "in_voc": 0, "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "in_voc": 0, "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "in_voc": 0, "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "in_voc": 0, "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "in_voc": 0, "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "in_voc": 0, "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "in_voc": 0, "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "in_voc": 0, "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "in_voc": 0, "isthing": 0, "id": 200, "name": "rug-merged"},
]


VOC_C = 21


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits""" ""
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


voc_cmaps = labelcolormap(VOC_C).tolist()
VOC_CATEGORIES = [
    {"id": 1, "name": "aeroplane", "isthing": 1, "color": voc_cmaps[1]},
    {"id": 2, "name": "bicycle", "isthing": 1, "color": voc_cmaps[2]},
    {"id": 3, "name": "bird", "isthing": 1, "color": voc_cmaps[3]},
    {"id": 4, "name": "boat", "isthing": 1, "color": voc_cmaps[4]},
    {"id": 5, "name": "bottle", "isthing": 1, "color": voc_cmaps[5]},
    {"id": 6, "name": "bus", "isthing": 1, "color": voc_cmaps[6]},
    {"id": 7, "name": "car", "isthing": 1, "color": voc_cmaps[7]},
    {"id": 8, "name": "cat", "isthing": 1, "color": voc_cmaps[8]},
    {"id": 9, "name": "chair", "isthing": 1, "color": voc_cmaps[9]},
    {"id": 10, "name": "cow", "isthing": 1, "color": voc_cmaps[10]},
    {"id": 11, "name": "diningtable", "isthing": 1, "color": voc_cmaps[11]},
    {"id": 12, "name": "dog", "isthing": 1, "color": voc_cmaps[12]},
    {"id": 13, "name": "horse", "isthing": 1, "color": voc_cmaps[13]},
    {"id": 14, "name": "motorbike", "isthing": 1, "color": voc_cmaps[14]},
    {"id": 15, "name": "person", "isthing": 1, "color": voc_cmaps[15]},
    {"id": 16, "name": "pottedplant", "isthing": 1, "color": voc_cmaps[16]},
    {"id": 17, "name": "sheep", "isthing": 1, "color": voc_cmaps[17]},
    {"id": 18, "name": "sofa", "isthing": 1, "color": voc_cmaps[18]},
    {"id": 19, "name": "train", "isthing": 1, "color": voc_cmaps[19]},
    {"id": 20, "name": "tvmonitor", "isthing": 1, "color": voc_cmaps[20]},
    {"id": 21, "name": "background", "isthing": 0, "color": [255, 255, 255]},
]


def _get_voc_instances_meta():
    thing_ids = [k["id"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous VOC category id to an id in [0, 79]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def _get_builtin_metadata(dataset_name):
    if dataset_name == "voc_json":
        return _get_voc_instances_meta()
    elif dataset_name == "imagenet":
        meta = {}
        return meta
    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))
