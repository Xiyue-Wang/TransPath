import pandas as pd
import numpy as np
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import argparse
from functools import partial
import moco.builder_infence
import moco.loader
import moco.optimizer
import torchvision.models as torchvision_models
import vits
torchvision_model_names = sorted(name for name in torchvision_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(torchvision_models.__dict__[name]))
model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names
parser = argparse.ArgumentParser(description='MoCov3 TCGA get_feature')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: vit_small)')
args = parser.parse_args()
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

img_csv=pd.read_csv(r'./test_list.csv')
test_datat=roi_dataset(img_csv)
database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

if args.arch.startswith('vit'):
    model = moco.builder_infence.MoCo_ViT(
        partial(vits.__dict__[args.arch], stop_grad_conv1=True))

    pretext_model = torch.load(r'./vit_small.pth.tar')['state_dict']
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(pretext_model, strict=True)

    model.eval()
    with torch.no_grad():
        for batch in database_loader:
            features = model(batch)
            features = features.cpu().numpy()


#
