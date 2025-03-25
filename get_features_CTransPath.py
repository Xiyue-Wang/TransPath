import argparse
import os
from pathlib import Path

import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from ctran import ctranspath
import torch.distributed as dist
import tqdm

from datasets.patch_dataset import PatchDataset

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
        default="./ctranspath.pth"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to write results and logs",
        required=True,
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Path to the dataset directory",
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for feature extraction",
        default=1024
    )
    return parser

def main(args):
    # setup distributed training if necessary
    is_distributed = int(os.environ.get("RANK", -1)) != -1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device_id = rank % torch.cuda.device_count()
        device = f"cuda:{device_id}"
    else:
        rank = 0
        world_size = 1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load model
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(args.pretrained_weights)
    model.load_state_dict(td['model'], strict=True)
    model = model.to(device)
    model.eval()

    # create output_dir if necessary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # load h5 files
    patch_files = list(Path(args.dataset_dir).rglob("*.h5"))
    # split files between processes
    file_list = patch_files[rank:: world_size]
    if rank == 0:
        # use tqdm only in the master process
        file_list = tqdm.tqdm(file_list)

    for patch_file in file_list:
        # skip files that have already been processed
        output_path = output_dir / f"{patch_file.stem}.pth"
        if output_path.exists():
            continue

        # extract features
        feature_list = []
        dataset = PatchDataset(patch_file, test_transforms)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=32, pin_memory=True)
        if rank == 0:
            loader = tqdm.tqdm(loader, total=len(loader))

        for batch in loader:
            with torch.inference_mode():
                features = model(batch.to(device, non_blocking=True))
                feature_list.append(features.cpu())
        patch_features = torch.cat(feature_list, dim=0)

        # save features
        torch.save(patch_features, output_path)

    if is_distributed:
        # synchronize processes
        dist.barrier()
        dist.destroy_process_group()




if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    main(args)
