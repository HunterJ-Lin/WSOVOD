import argparse
import json
import cv2
from six.moves import cPickle as pickle
from tqdm import tqdm
import numpy as np
from detectron2.data.catalog import DatasetCatalog
from detectron2.utils.file_io import PathManager
import os
import clip
from detectron2.data import MetadataCatalog
import torch

PROMPTS = [
    'There is a {category} in the scene.',
    'There is my {category} in the scene.',
    'There is the {category} in the scene.',
    'There is one {category} in the scene.',
    'a photo of a {category} in the scene.',
    'a photo of my {category} in the scene.',
    'a photo of the {category} in the scene.',
    'a photo of one {category} in the scene.',
    'itap of a {category}.',
    'itap of my {category}.',
    'itap of the {category}.',
    'itap of one {category}.',
    'a photo of a {category}.',
    'a photo of my {category}.',
    'a photo of the {category}.',
    'a photo of one {category}.',
    'a good photo of a {category}.',
    'a good photo of the {category}.',
    'a bad photo of a {category}.',
    'a bad photo of the {category}.',
    'a photo of a nice {category}.',
    'a photo of the nice {category}.',
    'a photo of a cool {category}.',
    'a photo of the cool {category}.',
    'a photo of a weird {category}.',
    'a photo of the weird {category}.',
    'a photo of a small {category}.',
    'a photo of the small {category}.',
    'a photo of a large {category}.',
    'a photo of the large {category}.',
    'a photo of a clean {category}.',
    'a photo of the clean {category}.',
    'a photo of a dirty {category}.',
    'a photo of the dirty {category}.',
    'a bright photo of a {category}.',
    'a bright photo of the {category}.',
    'a dark photo of a {category}.',
    'a dark photo of the {category}.',
    'a photo of a hard to see {category}.',
    'a photo of the hard to see {category}.',
    'a low resolution photo of a {category}.',
    'a low resolution photo of the {category}.',
    'a cropped photo of a {category}.',
    'a cropped photo of the {category}.',
    'a close-up photo of a {category}.',
    'a close-up photo of the {category}.',
    'a jpeg corrupted photo of a {category}.',
    'a jpeg corrupted photo of the {category}.',
    'a blurry photo of a {category}.',
    'a blurry photo of the {category}.',
    'a pixelated photo of a {category}.',
    'a pixelated photo of the {category}.',
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='coco_2017_val')
    parser.add_argument('--categories', type=str, default='')
    # parser.add_argument('--mode-type', type=str, default='ViT-L/14/32')
    parser.add_argument('--model-type', type=str, default='ViT-B/32')
    parser.add_argument('--prompt-type', type=str, default='single')
    parser.add_argument('--output', type=str, default='models/coco_text_embedding_single_prompt.pkl')
    args = parser.parse_args()
    # load clip model
    clip_model, clip_preprocess = clip.load(args.model_type, 'cpu', jit=False)
    clip_model = clip_model.eval()
    d = []

    thing_classes = MetadataCatalog.get(args.dataset_name).thing_classes
    if thing_classes is None:
        thing_classes = args.categories.split(',')
        assert len(thing_classes)>0

    if args.prompt_type == 'mutiple':
        for category in thing_classes:
            text_inputs = torch.cat([clip.tokenize(prompt.format(**{'category':category})) for prompt in PROMPTS]).to('cpu')
            text_embedding = clip_model.encode_text(text_inputs).float()
            d[category] = text_embedding.mean(0)
        text_embeddings = torch.cat(d)
    else:
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in thing_classes]).to('cpu')
        text_embeddings = clip_model.encode_text(text_inputs).float()

    print(text_embeddings.shape)
    # print('save to '+args.output)
    # with open(args.output, "wb") as f:
    #     pickle.dump(text_embeddings, f, pickle.HIGHEST_PROTOCOL)