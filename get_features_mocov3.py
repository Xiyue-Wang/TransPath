import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.models as torchvision_models
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import moco.builder_infence
import moco.loader
import moco.optimizer
import vits
from fe_utils import get_h5_list, extract_features_from_h5_wsi, TRANSFORM


TV_MODELS_NAMES = sorted(name for name in torchvision_models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(torchvision_models.__dict__[name]))
MODELS_NAMES = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + TV_MODELS_NAMES


def load_model(arch: str, checkpoint: str):
    assert arch.startswith('vit'), "This case was not handled in the original implementation"

    model = moco.builder_infence.MoCo_ViT(partial(vits.__dict__[arch], stop_grad_conv1=True))
    pretext_model = torch.load(checkpoint)['state_dict']
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(pretext_model, strict=True)

    return model.eval()


@click.command()
@click.option("--output-features-path", type=click.Path(exists=False),
              required=True, help="The directory inside which the extracted features should be saved",
              default="./features")
@click.option("--h5-wsi-patches-filepath", type=click.Path(exists=True), required=True,
              help="The directory inside which you can find the h5 WSIs patches datasets", default='./h5_wsi_patches')
@click.option("--model-checkpoint", type=click.Path(exists=True), required=True, help="Checkpoint of the MoCoV3 model")
@click.option("--arch", type=click.Choice(MODELS_NAMES), required=False, default="vit_small",
              help='model architecture: ' + ' | '.join(MODELS_NAMES) + ' (default: vit_small)')
@click.option("--batch-size", type=int, required=False, default=1024, help="The number of patches processed at once")
@click.option("--device", type=str, required=False, default="cuda:0", help="What device to use")
def main(output_features_path: str, h5_wsi_patches_filepath: str, model_checkpoint: str, arch: str, batch_size: int,
         device: str):
    # Check whether the output features directory exists and in case not create it
    if not os.path.exists(output_features_path):
        print("The output directory for the features %s does not exists. Creating it...")
        os.makedirs(output_features_path)

    # Load the model
    model = load_model(arch=arch, checkpoint=model_checkpoint)

    # Get a list of all the WSI in the dataset folder provided
    h5_wsi_files_list = get_h5_list(h5_wsi_patches_filepath)

    for h5_wsi_filename in tqdm.tqdm(iterable=h5_wsi_files_list, desc="Processing H5 WSI patches datasets...",
                                     total=len(h5_wsi_files_list)):
        print("Now processing %s..." % h5_wsi_filename)
        h5_wsi_filepath = os.path.join(h5_wsi_patches_filepath, h5_wsi_filename)
        h5_wsi = h5py.File(name=h5_wsi_filepath)  # load the h5 file
        features = extract_features_from_h5_wsi(h5_wsi=h5_wsi, model=model, batch_size=batch_size, device=device)
        with h5py.File(name=os.path.join(output_features_path, os.path.splitext(h5_wsi_filename)[0] + ".h5"),
                       mode="w") as features_h5:
            _ = features_h5.create_dataset("features", data=features,
                                           compression="gzip")  # no chunking, we want a single file
            _ = features_h5.create_dataset(name="coords", data=h5_wsi["coords"], compression="gzip")
