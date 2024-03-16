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


def get_filename_key(x):
    basename = os.path.basename(x)
    if basename[:-4].isdigit():
        return int(basename[:-4])

    return basename

def _strtobool(x):
    return bool(strtobool(x))

def parse_args():
    parser = argparse.ArgumentParser(description="Statistical ILSVRC")
    parser.add_argument("--img-root", help="img root", required=True)
    parser.add_argument("--out-file", help="output path", required=True)
    parser.add_argument('--max-per-dir', nargs='?', const=1e10, type=int, default=1e10, help="max per dir")
    parser.add_argument('--min-per-dir', nargs='?', const=0, type=int, default=0, help="min per dir")
    parser.add_argument('--has-shuffle', nargs='?', const=False, type=_strtobool, default=False, help="shuffle or sort")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    img_root = args.img_root
    out_file = args.out_file

    max_per_dir = args.max_per_dir
    min_per_dir = args.min_per_dir

    has_shuffle = args.has_shuffle

    folder_classes = []
    path_h_w = []
    pbar = tqdm(os.walk(img_root))
    for root, dirs, files in pbar:
        if not os.path.basename(root).startswith("n"):
            print("\tskip folder: ", root)
            continue

        files = [f for f in files if f.endswith(('.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG'))]

        if min_per_dir > 0 and len(files) < min_per_dir:
            print("\tskip folder: ", root, " #image: ", len(files))
            continue

        if has_shuffle:
            random.shuffle(files)
        else:
            files = sorted(files, key=lambda x: get_filename_key(x), reverse=False)


        folder_classes_this = []
        path_h_w_this = []
        for name in files:
            if max_per_dir > 0 and len(path_h_w_this) >= max_per_dir:
                break

            path = os.path.join(root, name)
            print('parsing:',path)
            rpath = path.replace(img_root, "")

            wnid = pathlib.PurePath(path).parent.name

            if wnid not in folder_classes_this:
                folder_classes_this.append(wnid)

            try:
                img = read_image(path, format="BGR")
                height, width, _ = img.shape
            except Exception as e:
                print("*" * 60)
                print("fail to open image: ", e)
                print("*" * 60)
                continue

            # if width < 300 or height < 300:
            #     continue

            # if width < 224 or height < 224:
            #     continue

            # if width >  1000 or height > 1000:
            #     continue

            path_h_w_this.append([rpath, height, width])

        if len(path_h_w_this) >= min_per_dir:
            pass
        else:
            print("\tskip folder: ", root, " #image: ", len(path_h_w_this))
            continue

        folder_classes.extend(folder_classes_this)
        path_h_w.extend(path_h_w_this)

        print("folder: ", root, " #image: ", len(path_h_w_this))
        print("#folder: ", len(folder_classes), " #image: ", len(path_h_w))

    folder_classes = list(set(folder_classes))
    folder_classes.sort()

    print("#category: ", len(folder_classes), " categories: ", folder_classes)
    print("#image: ", len(path_h_w))
    print("")
    with open(out_file,'w') as f:
        d = {}
        d['categories'] = folder_classes
        d['path_h_w'] = path_h_w
        json.dump(d,f)
    print(args)

if __name__ == "__main__":
    main()