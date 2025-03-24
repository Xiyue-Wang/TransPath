import argparse

import pandas as pd
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean, std = std)
    ]
)
class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        path = self.images_lst.filename[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)


        return image

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
    return parser

def main(args):

    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(args.pretrained_weights)
    model.load_state_dict(td['model'], strict=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        # run inference on random batch
        x = torch.randn(1, 3, 224, 224).cuda()
        features = model(x)
        print(features.shape)
        # for batch in database_loader:
        #     features = model(batch)
        #     features = features.cpu().numpy()


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    main(args)
