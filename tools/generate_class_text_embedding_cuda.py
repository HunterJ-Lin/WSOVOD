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
from detectron2.utils.comm import is_main_process, get_world_size, get_local_rank
import torch
import torch.distributed as dist

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
    parser.add_argument('--dataset-name', type=str, default='voc_2007_val')
    parser.add_argument('--categories', type=str, default='')
    parser.add_argument('--bs', type=int, default=32)
    # parser.add_argument('--mode-type', type=str, default='ViT-L/14/32')
    parser.add_argument('--model-type', type=str, default='ViT-B/32')
    parser.add_argument('--prompt-type', type=str, default='single')
    parser.add_argument('--output', type=str, default='models/voc_text_embedding_single_prompt_cuda.pkl')
    args = parser.parse_args()

    # load clip model
    clip_model, clip_preprocess = clip.load(args.model_type, 'cuda', jit=False)
    clip_model = clip_model.eval()

    thing_classes = MetadataCatalog.get(args.dataset_name).thing_classes
    if thing_classes is None:
        thing_classes = args.categories.split(',')
        assert len(thing_classes)>0

    descriptions = []
    candidates = []
    for cls_name in thing_classes:
        if args.prompt_type == 'mutiple':
            candidates.append(len(PROMPTS))
            for template in PROMPTS:
                description = template.format(**{'category':cls_name})
                descriptions.append(description)
        else:
            candidates.append(1)
            descriptions.append(f"a photo of a {cls_name}.")

    with torch.no_grad():
        tot = len(descriptions)
        bs = args.bs
        nb = tot // bs
        if tot % bs != 0:
            nb += 1
        text_embeddings_list = []
        for i in range(nb):
            local_descriptions = descriptions[i * bs: (i + 1) * bs]
            text_inputs = torch.cat([clip.tokenize(ds) for ds in local_descriptions]).to('cuda')
            local_text_embeddings = clip_model.encode_text(text_inputs).to(device='cpu').float()
            text_embeddings_list.append(local_text_embeddings)
        text_embeddings = torch.cat(text_embeddings_list)

    dim = text_embeddings.shape[-1]
    candidate_tot = sum(candidates)
    text_embeddings = text_embeddings.split(candidates, dim=0)
    if args.prompt_type == 'mutiple':
        text_embeddings = [text_embedding.mean(0).unsqueeze(0) for text_embedding in text_embeddings]
    
    text_embeddings = torch.cat(text_embeddings)
    print('save to '+args.output)
    with open(args.output, "wb") as f:
        pickle.dump(text_embeddings, f, pickle.HIGHEST_PROTOCOL)