from math import e
import cv2
import os
import argparse
import numpy as np
import pickle
from detectron2.data.catalog import DatasetCatalog
import torch
import torch.distributed as dist
from detectron2.utils.comm import synchronize, get_world_size, get_rank
from wsovod.data.datasets import builtin

# Import DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

def process_image(mask_generator, image_info):
    image_path = image_info['file_name']
    print("processing ", image_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        masks = mask_generator.generate(image)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        mask_generator.predictor.model.to('cpu')
        masks = mask_generator.generate(image)
        mask_generator.predictor.model.to(f'cuda:{get_rank()}')
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
    parser.add_argument('--model-type', type=str, default='vit_h')
    parser.add_argument('--dataset-name', type=str, default='coco_2017_val')
    parser.add_argument('--output', type=str, default='datasets/proposals/sam_coco_2017_val_d2.pkl')
    parser.add_argument('--points-per-side', type=int, default=32)
    parser.add_argument('--pred-iou-thresh', type=float, default=0.86)        
    parser.add_argument('--stability-score-thresh', type=float, default=0.92)
    parser.add_argument('--crop-n-layers', type=int, default=1)  
    parser.add_argument('--crop-n-points-downscale-factor', type=int, default=2)
    parser.add_argument('--min-mask-region-area', type=float, default=20.0)
    # Add an argument for the local rank that will be set by torchrun or the PyTorch launcher
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    import os
    print("Local rank (from environment):", os.environ.get("LOCAL_RANK"))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dataset_dicts = DatasetCatalog.get(args.dataset_name)
    # dataset_dicts = dataset_dicts[:100] for debugging
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f'cuda:{rank}')
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        crop_n_points_downscale_factor=args.crop_n_points_downscale_factor,
        min_mask_region_area=args.min_mask_region_area,  # Requires open-cv to run post-processing
    )
    
    # Now, adjust the data loading and processing to only work on a subset of the data based on the rank of the process
    # This is a simple way to split the dataset, more sophisticated methods might be needed for your use case
    subset_size = len(dataset_dicts) // world_size
    start_idx = rank * subset_size
    end_idx = start_idx + subset_size if rank < world_size - 1 else len(dataset_dicts)
    local_dataset_dicts = dataset_dicts[start_idx:end_idx]

    all_boxes = []
    all_scores = []
    all_indexes = []
    if rank == 0:
        from tqdm import tqdm
        local_dataset_dicts = tqdm(local_dataset_dicts)
    with torch.no_grad():
        for image_info in local_dataset_dicts:
            boxes, scores, indexes = process_image(mask_generator, image_info)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_indexes.append(indexes)

    # Gather results from all processes
    gathered_boxes = [None] * world_size
    gathered_scores = [None] * world_size
    gathered_indexes = [None] * world_size
    torch.cuda.set_device(rank)
    dist.barrier()
    print(f"Rank: {rank} gathering boxes results...")
    dist.all_gather_object(gathered_boxes, all_boxes)
    print(f"Rank: {rank} gathered boxes.")
    dist.barrier()
    print(f"Rank: {rank} gathering scores results...")
    dist.all_gather_object(gathered_scores, all_scores)
    print(f"Rank: {rank} gathered scores.")
    dist.barrier()
    print(f"Rank: {rank} gathering indexes results...")
    dist.all_gather_object(gathered_indexes, all_indexes)
    print(f"Rank: {rank} gathered indexes.")
    dist.barrier()
    # Collecting and saving the results should be done by one process
    if rank == 0:
        print("All results gathered.")
        # Flatten lists from all processes
        all_boxes_flat = [item for sublist in gathered_boxes for item in sublist]
        all_scores_flat = [item for sublist in gathered_scores for item in sublist]
        all_indexes_flat = [item for sublist in gathered_indexes for item in sublist]
        assert len(all_boxes_flat) == len(all_scores_flat) == len(all_indexes_flat) == len(dataset_dicts)
        assert set(all_indexes_flat) == set([image_info['image_id'] for image_info in dataset_dicts])
        # Save only in the master process
        output_file = args.output
        with open(output_file, 'wb') as f:
            pickle.dump({'boxes': all_boxes_flat, 'scores': all_scores_flat, 'indexes': all_indexes_flat}, f)
        print("Proposal generation and saving completed.")
    # Ensure all processes have finished before exiting
    dist.barrier()

if __name__ == '__main__':
    main()
