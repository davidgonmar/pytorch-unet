from torch.utils.data import Dataset
import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm
from PIL import Image
import torch
import logging
import numpy as np
from typing import Dict, Literal, Callable


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class OxfordPetsDataset(Dataset):
    IMAGES_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
    ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"

    def __init__(
        self,
        root: str,
        mode: str = "train",
        transform: Dict[Literal["image", "mask"], Callable] = None,
    ):
        super().__init__()

        self.root = Path(root) / "oxford_pets"
        self.raw_folder = self.root / "raw"
        self.mode = mode
        self._download()
        self.split_filenames = self._read_split()
        self.img_transform = (
            transform["image"]
            if transform is not None and "image" in transform
            else lambda x: x
        )
        self.mask_transform = (
            transform["mask"]
            if transform is not None and "mask" in transform
            else lambda x: x
        )

    def _check_downloaded(self) -> bool:
        if (
            not os.path.exists(self.root)
            or not os.path.exists(self.raw_folder)
            or not os.path.exists(self.raw_folder / "images")
            or not os.path.exists(self.raw_folder / "annotations")
            or not os.path.exists(self.raw_folder / "images.tar.gz")
            or not os.path.exists(self.raw_folder / "annotations.tar.gz")
            or not os.path.exists(self.raw_folder / "annotations" / "trimaps")
        ):
            logging.info("OxfordPetsDataset not downloaded")
            return False
        logging.info("OxfordPetsDataset already downloaded")
        return True

    def _download(self):
        def extract_archive(filepath, into_dir=None):
            logging.info(f"Extracting {filepath}")
            if into_dir is None:
                into_dir = filepath.parent
            shutil.unpack_archive(filepath, into_dir)

        def download_url(url, path):
            if os.path.exists(path):
                print("File already exists")
            else:
                with TqdmUpTo(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=url.split("/")[-1],
                ) as t:
                    urlretrieve(url, filename=path, reporthook=t.update_to)

        if not self._check_downloaded():
            logging.info("Downloading OxfordPetsDataset")
            # make dirs
            os.makedirs(self.root, exist_ok=True)
            os.makedirs(self.raw_folder, exist_ok=True)
            # download
            download_url(self.IMAGES_URL, self.raw_folder / "images.tar.gz")
            download_url(self.ANNOTATIONS_URL, self.raw_folder / "annotations.tar.gz")
            extract_archive(self.raw_folder / "images.tar.gz", self.raw_folder)
            extract_archive(self.raw_folder / "annotations.tar.gz", self.raw_folder)

    def _read_split(self):
        filename = "trainval.txt" if self.mode == "train" else "test.txt"
        p = self.raw_folder / "annotations" / filename

        with open(p, "r") as f:
            lines = f.readlines()

        return [l.split(" ")[0] for l in lines]

    def __len__(self):
        return len(self.split_filenames)

    def __getitem__(self, idx):
        filename = self.split_filenames[idx]
        image = Image.open(self.raw_folder / "images" / (filename + ".jpg"))
        image = np.array(image)[:, :, :3]
        trimap = Image.open(
            self.raw_folder / "annotations" / "trimaps" / (filename + ".png")
        )
        mask = torch.tensor(np.array(trimap)).float()
        mask[mask == 2] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        mask = mask.unsqueeze(0)  # add channel dim
        # images in PIL are H x W x C, we need C x H x W and normalize to [0, 1]
        img = torch.tensor(image).permute(2, 0, 1).to(torch.float32) / 255.0
        assert img.shape[0] == 3, f"got {img.shape[0]} channels"
        assert mask.shape[0] == 1, f"got {mask.shape[0]} channels"
        return self.img_transform(img), self.mask_transform(mask)
