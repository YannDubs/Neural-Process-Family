import glob
import logging
import os
import pathlib

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

DIR_DATA = os.path.abspath(os.path.join(__file__, "../../data/"))
logger = logging.getLogger(__name__)


class DatasetMerger(Dataset):
    """
    Helper which merges an iterable of datasets. Assume that they all have the same attributes 
    (redirect to the first one).
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.cumul_len = np.cumsum([len(d) for d in self.datasets])

    def __getitem__(self, index):
        idx_dataset = self.cumul_len.searchsorted(index + 1)  # + 1 because of 0 index
        idx_in_dataset = index
        if idx_dataset > 0:
            idx_in_dataset -= self.cumul_len[idx_dataset - 1]  # - 1 because rm previous
        return self.datasets[idx_dataset][idx_in_dataset]

    def __len__(self):
        return self.cumul_len[-1]

    def __getattr__(self, attr):
        return getattr(self.datasets[0], attr)


class _DatasetSubset(Dataset):
    """Helper to split train dataset into train and dev dataset.

    Parameters
    ----------
    to_split: Dataset
        Dataset to subset.

    idx_mapping: array-like
        Indices of the subset.

    Notes
    -----
    - Modified from: https: // gist.github.com / Fuchai / 12f2321e6c8fa53058f5eb23aeddb6ab
    - Cannot modify the length and targets with indexing anymore! I.e.
    `d.targets[1]=-1` doesn't work because np.array doesn't allow `arr[i][j]=-1`
    but you can do `d.targets=targets`
    """

    def __init__(self, to_split, idx_mapping):
        self.idx_mapping = idx_mapping
        self.length = len(idx_mapping)
        self.to_split = to_split

    def __getitem__(self, index):
        return self.to_split[self.idx_mapping[index]]

    def __len__(self):
        return self.length

    @property
    def targets(self):
        return self.to_split.targets[self.idx_mapping]

    @targets.setter
    def targets(self, values):
        self.to_split.targets[self.idx_mapping] = values

    @property
    def data(self):
        return self.to_split.data[self.idx_mapping]

    def __getattr__(self, attr):
        return getattr(self.to_split, attr)


def train_dev_split(to_split, dev_size=0.1, seed=123, is_stratify=True):
    """Split a training dataset into a training and validation one.

    Parameters
    ----------
    dev_size: float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of
        the dataset to include in the dev split. If int, represents the absolute
        number of dev samples.

    seed: int
        Random seed.

    is_stratify: bool
        Whether to stratify splits based on class label.
    """
    n_all = len(to_split)
    idcs_all = list(range(n_all))
    stratify = to_split.targets if is_stratify else None
    idcs_train, indcs_val = train_test_split(
        idcs_all, stratify=stratify, test_size=dev_size, random_state=seed
    )
    train = _DatasetSubset(to_split, idcs_train)
    valid = _DatasetSubset(to_split, indcs_val)

    return train, valid


def preprocess(root, size=(64, 64), img_format="JPEG", center_crop=None):
    """Preprocess a folder of images.

    Parameters
    ----------
    root : string
        Root directory of all images.

    size : tuple of int
        Size (width, height) to rescale the images. If `None` don't rescale.

    img_format : string
        Format to save the image in. Possible formats:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html.

    center_crop : tuple of int
        Size (width, height) to center-crop the images. If `None` don't center-crop.
    """
    imgs = []
    for ext in [".png", ".jpg", ".jpeg"]:
        imgs += glob.glob(os.path.join(root, "*" + ext))

    for img_path in tqdm(imgs):
        img = Image.open(img_path)
        width, height = img.size

        if size is not None and width != size[1] or height != size[0]:
            img = img.resize(size, Image.ANTIALIAS)

        if center_crop is not None:
            new_width, new_height = center_crop
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = (width + new_width) // 2
            bottom = (height + new_height) // 2

            img.crop((left, top, right, bottom))

        img.save(img_path, img_format)


def random_translation(img, max_pix):
    """
    Random translations of 0 to max_pix given np.ndarray or PIL.Image(H W C).
    """
    is_pil = not isinstance(img, np.ndarray)
    if is_pil:
        img = np.atleast_3d(np.asarray(img))
    idx_h, idx_w = 0, 1
    img = np.pad(img, [[max_pix, max_pix], [max_pix, max_pix], [0, 0]], mode="reflect")
    shifts = np.random.randint(-max_pix, max_pix + 1, size=[2])  # H and W
    processed_data = np.roll(img, shifts, (idx_h, idx_w))
    cropped_data = processed_data[max_pix:-max_pix, max_pix:-max_pix, :]
    if is_pil:
        img = Image.fromarray(img.squeeze())
    return cropped_data


def _parse_save_file_chunk(save_file, idx_chunk):
    if save_file is None:
        save_file, save_group = None, None
    elif isinstance(save_file, tuple):
        save_file, save_group = save_file[0], save_file[1] + "/"
    elif isinstance(save_file, str):
        save_file, save_group = save_file, ""
    else:
        raise ValueError("Unsupported type of save_file={}.".format(save_file))

    if idx_chunk is not None:
        chunk_suffix = "_chunk_{}".format(idx_chunk)
    else:
        chunk_suffix = ""

    return save_file, save_group, chunk_suffix


class NotLoadedError(Exception):
    pass


def load_chunk(keys, save_file, idx_chunk):
    items = dict()
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None or not os.path.exists(save_file):
        raise NotLoadedError()

    try:
        with h5py.File(save_file, "r") as hf:
            for k in keys:
                items[k] = torch.from_numpy(
                    hf["{}{}{}".format(save_group, k, chunk_suffix)][:]
                )
    except KeyError:
        raise NotLoadedError()

    return items


def save_chunk(to_save, save_file, idx_chunk, logger=None):
    save_file, save_group, chunk_suffix = _parse_save_file_chunk(save_file, idx_chunk)

    if save_file is None:
        return  # don't save

    if logger is not None:
        logger.info(
            "Saving group {} chunk {} for future use ...".format(save_group, idx_chunk)
        )

    with h5py.File(save_file, "a") as hf:
        for k, v in to_save.items():
            hf.create_dataset(
                "{}{}{}".format(save_group, k, chunk_suffix), data=v.numpy()
            )
