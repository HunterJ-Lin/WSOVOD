import argparse
import json
import os
import os.path as osp
import pathlib
import random
import xml.dom.minidom
from distutils.util import strtobool

import numpy as np
from detectron2.data.detection_utils import read_image
# from nltk.corpus import wordnet as wn
from PIL import Image
from tqdm import tqdm

# import nltk
# nltk.download('wordnet')

miss_xml = []
folder_classes = []

def parse_xml(file_path):
    print("processing: ", file_path)

    dom = xml.dom.minidom.parse(file_path)
    document = dom.documentElement
    
    size = document.getElementsByTagName('size')
    bboxes = []
    labels = []
    for item in size:
        width = item.getElementsByTagName('width')
        width = int(width[0].firstChild.data)
        height = item.getElementsByTagName('height')
        height = int(height[0].firstChild.data)

    # try:
    if True:
        object = document.getElementsByTagName('object')
        for item in object:
            label = item.getElementsByTagName('name')
            label = str(label[0].firstChild.data)

            xmin = item.getElementsByTagName('xmin')
            xmin = float(xmin[0].firstChild.data)
            ymin = item.getElementsByTagName('ymin')
            ymin = float(ymin[0].firstChild.data)
            xmax = item.getElementsByTagName('xmax')
            xmax = float(xmax[0].firstChild.data)
            ymax = item.getElementsByTagName('ymax')
            ymax = float(ymax[0].firstChild.data)

            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
    # except:
    #     continue

    return bboxes, labels


def cvt_annotations(path_h_w, out_file, has_instance, has_segmentation, ann_root):
    label_ids = {name: i for i, name in enumerate(folder_classes)}
    print('cvt annotations')
    print("label_ids: ", label_ids)

    annotations = []
    pbar = enumerate(path_h_w)
    pbar = tqdm(pbar)
    for i, [img_path, height, width] in pbar:
        # if i > 10000:
        #     break
        # print(i, img_path)

        wnid = pathlib.PurePath(img_path).parent.name

        if has_instance:
            xml_path = ann_root + img_path[:-5] + ".xml"
            if os.path.exists(xml_path):
                bboxes, wnids = parse_xml(xml_path)
                ## check error wnid
                # for wnid_ in wnids:
                #     assert wnid_ == wnid, [wnids, wnid]
                # labels = [label_ids[wnid_] for wnid_ in wnids]
                labels = [label_ids[wnid] for wnid_ in wnids]
            else:
                print("xml file not found", xml_path)
                miss_xml.append(xml_path)
                # continue
                print("genenrate pseudo annotation for", img_path)
                bboxes = [[1, 1, width, height]]
                labels = [label_ids[wnid]]

            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        else:
            bboxes = np.array(np.zeros((0, 4), dtype=np.float32))
            labels = np.array(np.zeros((0,), dtype=np.int64))

        annotation = {
            "filename": img_path,
            "width": width,
            "height": height,
            "ann": {
                "bboxes": bboxes.astype(np.float32),
                "labels": labels.astype(np.int64),
                "bboxes_ignore": np.zeros((0, 4), dtype=np.float32),
                "labels_ignore": np.zeros((0,), dtype=np.int64),
            },
        }

        annotations.append(annotation)
    annotations = cvt_to_coco_json(annotations, has_segmentation)

    with open(out_file, "w") as f:
        json.dump(annotations, f)

    return annotations


def cvt_to_coco_json(annotations, has_segmentation):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco["images"] = []
    coco["type"] = "instance"
    coco["categories"] = []
    coco["annotations"] = []
    image_set = set()
    print('cvt to coco json')
    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        if has_segmentation:
            annotation_item["segmentation"] = []

            seg = []
            # bbox[] is x1,y1,x2,y2
            # left_top
            seg.append(int(bbox[0]))
            seg.append(int(bbox[1]))
            # left_bottom
            seg.append(int(bbox[0]))
            seg.append(int(bbox[3]))
            # right_bottom
            seg.append(int(bbox[2]))
            seg.append(int(bbox[3]))
            # right_top
            seg.append(int(bbox[2]))
            seg.append(int(bbox[1]))

            annotation_item["segmentation"].append(seg)

        xywh = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item["area"] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item["ignore"] = 0
            annotation_item["iscrowd"] = 1
        else:
            annotation_item["ignore"] = 0
            annotation_item["iscrowd"] = 0
        annotation_item["image_id"] = int(image_id)
        annotation_item["bbox"] = xywh.astype(int).tolist()
        annotation_item["category_id"] = int(category_id)
        annotation_item["id"] = int(annotation_id)
        coco["annotations"].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(folder_classes):
        category_item = dict()
        category_item["supercategory"] = str("none")
        category_item["id"] = int(category_id)
        category_item["name"] = str(name)
        coco["categories"].append(category_item)

    pbar = tqdm(annotations)
    for ann_dict in pbar:
        file_name = ann_dict["filename"]
        ann = ann_dict["ann"]
        assert file_name not in image_set
        image_item = dict()
        image_item["id"] = int(image_id)
        image_item["file_name"] = str(file_name)
        image_item["height"] = int(ann_dict["height"])
        image_item["width"] = int(ann_dict["width"])
        coco["images"].append(image_item)
        image_set.add(file_name)

        bboxes = ann["bboxes"][:, :4]
        labels = ann["labels"]
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann["bboxes_ignore"][:, :4]
        labels_ignore = ann["labels_ignore"]
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def _strtobool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert image list to coco format")
    parser.add_argument("--ann-root", help="ann root", type=str, default='')
    parser.add_argument("--out-file", help="output path", required=True)
    parser.add_argument("--info-json", help="output path", required=True)
    parser.add_argument('--has-instance', nargs='?', const=True, type=_strtobool, default=True, help="has instance")
    parser.add_argument('--has-segmentation', nargs='?', const=True, type=_strtobool, default=True, help="has segmentation")
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    print(args)
    ann_root = args.ann_root
    out_file = args.out_file

    has_instance = args.has_instance
    has_segmentation = args.has_segmentation
 
    global folder_classes
    with open(args.info_json,'r') as f:
        d = json.load(f)
        folder_classes = d['categories']
        path_h_w = d['path_h_w']
    annotations = cvt_annotations(path_h_w, out_file, has_instance, has_segmentation, ann_root)
    print("Done!")
    print('miss xml file:{}'.format(len(miss_xml)))
    print(miss_xml)

    with open(out_file, "w") as f:
        json.dump(annotations, f)

    print(args)


if __name__ == "__main__":
    main()