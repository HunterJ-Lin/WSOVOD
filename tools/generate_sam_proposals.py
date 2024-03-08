import cv2
import os
import argparse
import numpy as np
import pickle
from detectron2.data.catalog import DatasetCatalog
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import torch

def process_image(mask_generator, image_info, device):
    torch.cuda.set_device(device)
    image_path = image_info['file_name']
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    proposals = []
    scores = []
    for instance in masks:
        score = instance['predicted_iou'] * instance['stability_score']
        if score > 1.0:
            score = 1.0
        bbox = instance['bbox']
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
        scores.append(score)
        proposals.append(bbox)
    proposals = np.array(proposals)
    scores = np.array(scores)
    return proposals, scores, image_info['image_id']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='tools/sam_checkpoints/sam_vit_h_4b8939.pth')
    parser.add_argument('--mode-type', type=str, default='vit_h')
    parser.add_argument('--dataset-name', type=str, default='coco_2017_val')
    parser.add_argument('--output', type=str, default='datasets/proposals/sam_coco_2017_val_d2.pkl')    
    args = parser.parse_args()

    dataset_dicts = DatasetCatalog.get(args.dataset_name)
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    num_gpus = torch.cuda.device_count()
    executors = [ThreadPoolExecutor(max_workers=1) for _ in range(num_gpus)]
    mask_generators = []
    sam_checkpoint = args.checkpoint
    model_type = args.model_type
    for i in range(num_gpus):
        gpu_index = i
        device = f'cuda:{gpu_index}'
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=20,  # Requires open-cv to run post-processing
        )
        mask_generators.append(mask_generator)

    all_boxes = []
    all_scores = []
    all_indexes = []

    future_to_image_id = []
    for i, image_info in enumerate(dataset_dicts):
        gpu_index = i % num_gpus
        device = f'cuda:{gpu_index}'
        mask_generator = mask_generators[gpu_index]
        future = executors[gpu_index].submit(process_image, mask_generator, image_info, device)
        future_to_image_id.append((future, image_info['image_id']))

    for future, _ in tqdm(future_to_image_id, total=len(dataset_dicts), desc="Processing images"):
        boxes, scores, index = future.result()
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_indexes.append(index)

    output_file = args.output
    with open(output_file, 'wb') as f:
        pickle.dump(dict(boxes=all_boxes, scores=all_scores, indexes=all_indexes), f)

    print("Proposal generation and saving completed.")

if __name__ == '__main__':
    main()
