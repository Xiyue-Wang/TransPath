import os

import click
import h5py
import torch
import torch.nn as nn
import tqdm

from ctran import ctranspath
from fe_utils import get_h5_list, extract_features_from_h5_wsi, TRANSFORM


@click.command()
@click.option("--output-features-path", type=click.Path(exists=False),
              required=True, help="The directory inside which the extracted features should be saved",
              default="./features")
@click.option("--h5-wsi-patches-filepath", type=click.Path(exists=True), required=True,
              help="The directory inside which you can find the h5 WSIs patches datasets",
              default='./h5_wsi_patches')
@click.option("--model-checkpoint", type=click.Path(exists=True), required=True,
              help="Checkpoint of the CTransPath model")
@click.option("--batch-size", type=int, required=False, default=1024, help="The number of patches processed at once")
@click.option("--device", type=str, required=False, default="cuda:0", help="What device to use")
def main(output_features_path: str, h5_wsi_patches_filepath: str, model_checkpoint: str, batch_size: int,
         device: str):
    # Check whether the output features directory exists and in case not create it
    if not os.path.exists(output_features_path):
        print("The output directory for the features %s does not exists. Creating it...")
        os.makedirs(output_features_path)

    # Load the model
    model = ctranspath()
    model.head = nn.Identity()

    # Load the checkpoint
    td = torch.load(r'./ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)

    model.eval()

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


if __name__ == "__main__":
    main()
