import click
import h5py
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.transforms import Resize, ConvertImageDtype, Normalize, Compose

from ctran import ctranspath

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
