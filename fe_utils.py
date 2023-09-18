import typing
import os

import h5py
import torch
from torchvision.transforms.transforms import Resize, ConvertImageDtype, Normalize, Compose
import numpy as np


TRANSFORM = Compose([
    Resize(size=(224, 224)),
    ConvertImageDtype(torch.float32),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_h5_list(dir_name: str) -> typing.List[str]:
    h5_files: list[str] = []
    for root, _, files in os.walk(dir_name):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.endswith("h5") and not file[0] == ".":
                h5_files.append(file)
    h5_files.sort()

    return h5_files


def extract_features_from_h5_wsi(h5_wsi: h5py.File, model: torch.nn.Module, batch_size: int,
                                 device: str) -> torch.Tensor:
    patches = np.array(h5_wsi["imgs"])
    print("Patches have shape: ", patches.shape)
    patches = torch.from_numpy(patches).to(device)

    # swap axis to make color channels the first axis
    patches = torch.einsum('nhwc->nchw', patches)
    patches = TRANSFORM(patches)

    features = []
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        with torch.no_grad():
            batch_features = model.forward(batch)
        features.append(batch_features)

    features = torch.cat(features, dim=0)

    return features.to("cpu")
