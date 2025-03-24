import io
from typing import List, Any

import h5py
from PIL import Image


def load_patches(f: Any, group_suffix: str = None) -> List[Image]:
    """
    Load patches and from the provided file like object

    Args:
        f: path to the file or file-like object
        group_suffix: suffix for the group name

    Returns:
        list of patches (PIL images)
    """
    with h5py.File(f, "r") as file:
        if group_suffix is None:
            group = "patches"
        else:
            group = f"patches_{group_suffix}"
            if group not in file:
                group = "patches"
        assert group in file, f"Group {group} not found in the file"
        patch_count = len(file[group])
        patches = []
        for i in range(patch_count):
            img_bytes = file[f"{group}/{i}"][()]
            img = Image.open(io.BytesIO(img_bytes))
            patches.append(img)

    return patches
