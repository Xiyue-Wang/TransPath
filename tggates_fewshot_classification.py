import argparse
from multiprocessing.sharedctypes import class_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from ctran import ctranspath
from datasets.io import load_patches

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
test_transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)


class PatchDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return test_transform(self.patches[idx])


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
        "--train-set-path",
        type=str,
        help="Path to the train set pth file",
        required=True,
    )
    parser.add_argument(
        "--test-set-path",
        type=str,
        help="Path to the test set pth file",
        required=True,
    )
    return parser


def compute_embeddings(model: nn.Module, files_to_encode: Iterable[str], dataset_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_paths = [dataset_dir / f"{patch_file}.h5" for patch_file in files_to_encode]
    print(f"Computing embeddings for {len(file_paths)} files")
    for file_path in tqdm(file_paths):
        patches = load_patches(file_path, group_suffix="256")

        patch_dataset = PatchDataset(patches)
        # create dataloader
        patch_loader = DataLoader(
            patch_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )
        embeddings_list = []
        with torch.inference_mode():
            for samples in tqdm(patch_loader):
                samples = samples.cuda(non_blocking=True)
                embeddings = model(samples)
                embeddings_list.append(embeddings.cpu())

        # concatenate embeddings
        embeddings = torch.cat(embeddings_list, dim=0)
        # save embeddings
        output_path = output_dir / f"{file_path.stem}.pth"
        torch.save(embeddings, output_path)
        print(f"Saved embeddings for {file_path} to {output_path}")


def main(args):
    # load the model
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(args.pretrained_weights)
    model.load_state_dict(td['model'], strict=True)
    model.cuda()
    model.eval()
    # load training and test set
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    class_train_set = torch.load(args.train_set_path)
    class_test_set = torch.load(args.test_set_path)
    # compute embeddings if necessary
    train_files_to_encode = set()
    for class_dict in class_train_set.values():
        # add files only if they don't exist in the output directory
        for filename in class_dict.keys():
            if not (output_dir / "train" / f"{filename}.pth").exists():
                train_files_to_encode.add(filename)
    if len(train_files_to_encode) > 0:
        compute_embeddings(model, train_files_to_encode, dataset_dir / "train", output_dir / "train")
    test_files_to_encode = set()
    for class_dict in class_test_set.values():
        # add files only if they don't exist in the output directory
        for filename in class_dict.keys():
            if not (output_dir / "test" / f"{filename}.pth").exists():
                test_files_to_encode.add(filename)
    if len(test_files_to_encode) > 0:
        compute_embeddings(model, test_files_to_encode, dataset_dir / "test", output_dir / "test")

    avg_auc = []
    per_class_auc = {}
    for lesion_class in class_train_set.keys():
        # load positive and negative examples for training
        training_set = class_train_set[lesion_class]
        # load training dataset
        train_x, train_y = load_input_and_labels(training_set, output_dir / "train")
        print(f"Loading training dataset of size {train_x.shape[0]} for class '{lesion_class}'")
        # train logistic regression model
        print("Training logistic regression model...")
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000)
        clf.fit(train_x, train_y)

        # load positive and negative examples for testing
        test_set = class_test_set[lesion_class]
        # load test dataset
        test_x, test_y = load_input_and_labels(test_set, output_dir / "test")
        print(f"Loading test dataset of size {test_x.shape[0]} for class '{lesion_class}'")
        # predict using the logistic regression model
        print("Predicting using logistic regression model...")
        test_probs = clf.predict_proba(test_x)[:, 1]
        # compute roc auc
        roc_auc = roc_auc_score(test_y, test_probs)
        per_class_auc[lesion_class] = roc_auc
        avg_auc.append(roc_auc)
    # save per_class_auc
    torch.save(per_class_auc, str(output_dir / f"per_class_auc_{Path(args.test_set_path).stem}.pth"))

    print("Per class AUC:")
    for lesion_class, auc in per_class_auc.items():
        print(f"{lesion_class}: {auc}")
    print(f"Average ROC AUC score: {np.mean(avg_auc)}")


def load_input_and_labels(training_set: dict[str, int], embedding_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for filename, label in training_set.items():
        # load embeddings and compute the mean
        embeddings = torch.load(embedding_dir / f"{filename}.pth")
        mean_emb = embeddings.mean(dim=0).numpy()
        X.append(mean_emb)
        y.append(label)
    # concatenate the np arrays
    X = np.stack(X)
    y = np.stack(y)
    assert X.shape[0] == y.shape[0]
    return X, y


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    main(args)
