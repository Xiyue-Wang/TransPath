
from numpy.lib.function_base import append
from torch.autograd import Variable
import torch, torchvision
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from torchvision.models import resnet50
from byol_pytorch.byol_pytorch_get_feature import BYOL

from torch.utils.data import Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize(256),
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
model = BYOL(
    image_size=256,
    hidden_layer='to_latent'
)

img_csv=pd.read_csv(r'./test_list.csv')
test_datat=roi_dataset(img_csv)
database_loader = torch.utils.data.DataLoader(test_datat, batch_size=1, shuffle=False)

pretext_model = torch.load(r'./checkpoint.pth')
model = nn.DataParallel(model).cuda()
model.load_state_dict(pretext_model, strict=True)

model.module.online_encoder.net.head = nn.Identity()

model.eval()
with torch.no_grad():
    for batch in database_loader:
        _, embedding = model(batch.cuda(),return_embedding = True)

