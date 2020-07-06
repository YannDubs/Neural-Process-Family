import abc
import glob
import hashlib
import logging
import os
import subprocess
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skorch.utils import to_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.helpers import set_seed

from .helpers import DIR_DATA, preprocess, random_translation, train_dev_split

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = torch.tensor([0.0, 0.0, 0.0])
COLOUR_WHITE = torch.tensor([1.0, 1.0, 1.0])
COLOUR_BLUE = torch.tensor([0.0, 0.0, 1.0])
DATASETS_DICT = {
    "mnist": "MNIST",
    "svhn": "SVHN",
    "celeba32": "CelebA32",
    "celeba64": "CelebA64",
    "zs-multi-mnist": "ZeroShotMultiMNIST",
    "zsmm": "ZeroShotMultiMNIST",  # shorthand
    "zsmmt": "ZeroShotMultiMNISTtrnslt",
    "zsmms": "ZeroShotMultiMNISTscale",
    "zs-mnist": "ZeroShotMNIST",
    "celeba": "CelebA",
}
DATASETS = list(DATASETS_DICT.keys())


# HELPERS
def get_train_test_img_dataset(dataset):
    """Return the correct instantiated train and test datasets."""
    try:
        train_dataset = get_dataset(dataset)(split="train")
        test_dataset = get_dataset(dataset)(split="test")
    except TypeError:
        train_dataset, test_dataset = train_dev_split(
            get_dataset(dataset)(), dev_size=0.1, is_stratify=False
        )

    return train_dataset, test_dataset


def get_dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).shape


# TORCHVISION DATASETS
class SVHN(datasets.SVHN):
    """SVHN wrapper. Docs: `datasets.SVHN.`

    Notes
    -----
    - Transformations (and their order) follow [1] besides the fact that we scale
    the images to be in [0,1] isntead of [-1,1] to make it easier to use
    probabilistic generative models.

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. 

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.CIFAR10`.

    References
    ----------
    [1] Oliver, A., Odena, A., Raffel, C. A., Cubuk, E. D., & Goodfellow, I.
        (2018). Realistic evaluation of deep semi-supervised learning algorithms.
        In Advances in Neural Information Processing Systems (pp. 3235-3246).
    """

    shape = (3, 32, 32)
    missing_px_color = COLOUR_BLACK
    n_classes = 10
    name = "SVHN"

    def __init__(
        self, root=DIR_DATA, split="train", logger=logging.getLogger(__name__), **kwargs
    ):

        if split == "train":
            transforms_list = [  # transforms.Lambda(lambda x: random_translation(x, 2)),
                transforms.ToTensor()
            ]
        elif split == "test":
            transforms_list = [transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(
            root,
            split=split,
            download=True,
            transform=transforms.Compose(transforms_list),
            **kwargs
        )

        self.labels = to_numpy(self.labels)

    @property
    def targets(self):
        # make compatible with CIFAR10 dataset
        return self.labels

    @targets.setter
    def targets(self, value):
        self.labels = value


class MNIST(datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST`.
    """

    shape = (1, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLUE
    name = "MNIST"

    def __init__(
        self, root=DIR_DATA, split="train", logger=logging.getLogger(__name__), **kwargs
    ):

        if split == "train":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(
            root,
            train=split == "train",
            download=True,
            transform=transforms.Compose(transforms_list),
            **kwargs
        )

        self.targets = to_numpy(self.targets)


# GENERATED DATASETS
class ZeroShotMultiMNIST(Dataset):
    """ZeroShotMultiMNIST dataset. The test set consists of multiple digits (by default 2).
    The training set consists of mnist digits with added black borders such that the image
    size is the same as in the test set, but the digits are of the same scale.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    split : {'train', 'test'}, optional
        According dataset is selected.

    n_test_digits : int, optional
        Number of digits per test image.

    final_size : int, optional
        Final size of the images (square of that shape). If `None` uses `n_test_digits*2`.

    seed : int, optional

    logger : logging.Logger

    kwargs:
        Additional arguments to the dataset data generation process `make_multi_mnist_*`.
    """

    missing_px_color = COLOUR_BLUE
    n_classes = 0
    shape = (1, 56, 56)
    files = {"train": "train", "test": "test"}
    name = "ZeroShotMultiMNIST"

    def __init__(
        self,
        root=DIR_DATA,
        split="train",
        n_test_digits=2,
        final_size=None,
        seed=123,
        logger=logging.getLogger(__name__),
        translation=0,
        **kwargs
    ):
        self.translation = translation
        self.split = split

        if split == "train":
            if self.translation:
                transforms_list = [
                    transforms.ToPILImage(),
                    transforms.RandomCrop(
                        (self.shape[1], self.shape[2]), padding=self.translation
                    ),
                    transforms.ToTensor(),
                ]
            else:
                transforms_list = [transforms.ToPILImage(), transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.ToPILImage(), transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        self.dir = os.path.join(root, self.name)
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger
        self.n_test_digits = n_test_digits
        self.seed = seed
        self.final_size = final_size
        self._init_size = 28

        saved_data = os.path.join(
            self.dir,
            "{}_seed{}_digits{}.pt".format(self.files[split], seed, n_test_digits),
        )

        try:
            self.data = torch.load(saved_data)
        except FileNotFoundError:
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)
            mnist = datasets.MNIST(root=root, train=split == "train", download=True)
            self.logger.info("Generating ZeroShotMultiMNIST {} split.".format(split))
            if split == "train":
                self.data = self.make_multi_mnist_train(mnist.data, **kwargs)
            elif split == "test":
                self.data = self.make_multi_mnist_test(mnist.data, **kwargs)
            torch.save(self.data, saved_data)
            self.logger.info("Finished Generating.")

        self.logger.info("Resizing ZeroShotMultiMNIST ...")
        self.data = self.data.float() / 255
        if self.final_size is not None:
            self.data = torch.nn.functional.interpolate(
                self.data.unsqueeze(1).float(),
                size=self.final_size,
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

    def __len__(self):
        return self.data.size(0)

    def make_multi_mnist_train(self, train_dataset):
        """Train set of multi mnist by taking mnist and adding borders to be the correct scale."""
        set_seed(self.seed)
        fin_img_size = self._init_size * self.n_test_digits
        init_img_size = train_dataset.shape[1:]
        background = np.zeros(
            (train_dataset.size(0), fin_img_size, fin_img_size)
        ).astype(np.uint8)
        borders = (np.array((fin_img_size, fin_img_size)) - init_img_size) // 2
        background[
            :, borders[0] : -borders[0], borders[1] : -borders[1]
        ] = train_dataset
        return torch.from_numpy(background)

    def make_multi_mnist_test(
        self, test_dataset, varying_axis=None, n_test_digits=None
    ):
        """
        Test set of multi mnist by concatenating moving digits around `varying_axis`
        (both axis if `None`) and concatenating them over the other. `n_test_digits` is th enumber
        of digits per test image (default `self.n_test_digits`).
        """
        set_seed(self.seed)

        n_test = test_dataset.size(0)

        if n_test_digits is None:
            n_test_digits = self.n_test_digits

        if varying_axis is None:
            out_axis0 = self.make_multi_mnist_test(
                test_dataset[: n_test // 2], varying_axis=0, n_test_digits=n_test_digits
            )
            out_axis1 = self.make_multi_mnist_test(
                test_dataset[: n_test // 2], varying_axis=1, n_test_digits=n_test_digits
            )
            return torch.cat((out_axis0, out_axis1), dim=0)[torch.randperm(n_test)]

        fin_img_size = self._init_size * self.n_test_digits
        n_tmp = self.n_test_digits * n_test
        init_img_size = test_dataset.shape[1:]

        tmp_img_size = list(test_dataset.shape[1:])
        tmp_img_size[varying_axis] = fin_img_size
        tmp_background = torch.from_numpy(
            np.zeros((n_tmp, *tmp_img_size)).astype(np.uint8)
        )

        max_shift = fin_img_size - init_img_size[varying_axis]
        shifts = np.random.randint(max_shift, size=n_test_digits * n_test)

        test_dataset = test_dataset.repeat(self.n_test_digits, 1, 1)[
            torch.randperm(n_tmp)
        ]

        for i, shift in enumerate(shifts):
            slices = [slice(None), slice(None)]
            slices[varying_axis] = slice(shift, shift + self._init_size)
            tmp_background[i, slices[0], slices[1]] = test_dataset[i, ...]

        out = torch.cat(tmp_background.split(n_test, 0), dim=1 + 1 - varying_axis)
        return out

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(self.data[idx]).float()

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class ZeroShotMultiMNISTtrnslt(ZeroShotMultiMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, translation=14, **kwargs)


class ZeroShotMultiMNISTscale(ZeroShotMultiMNIST):
    name = "ZeroShotMultiMNISTscale"
    shape = (1, 32, 32)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, translation=5, **kwargs)

        if self.split == "test":
            self.shape = (1, 56, 56)

    def make_multi_mnist_train(self, train_dataset):
        """Train set of multi mnist by taking mnist and adding borders to be the correct scale."""
        return train_dataset


class ZeroShotMNIST(ZeroShotMultiMNIST):
    """ZeroShotMNIST dataset. The test set consists of a single translated digit in a larger canvas.
    The training set consists of mnist digits with added black borders such that the image
    size is the same as in the test set, but the digits are of the same scale.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.

    split : {'train', 'test'}, optional
        According dataset is selected.

    final_size : int, optional
        Final size of the images (square of that shape). If `None` uses `n_test_digits*2`.

    seed : int, optional

    logger : logging.Logger

    kwargs:
        Additional arguments to the dataset data generation process `make_multi_mnist_*`.
    """

    missing_px_color = COLOUR_BLUE
    n_classes = 0
    shape = (1, 56, 56)
    files = {"train": "train", "test": "test"}
    name = "ZeroShotMNIST"

    def make_multi_mnist_test(
        self, test_dataset, varying_axis=None, n_test_digits=None
    ):
        """
        Like multi mnist but only shows a single digit.
        """
        return super().make_multi_mnist_test(
            test_dataset, varying_axis=varying_axis, n_test_digits=1
        )


# EXTERNAL DATASETS


class ExternalDataset(Dataset, abc.ABC):
    """Base Class for external datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.dir = os.path.join(root, self.name)
        self.train_data = os.path.join(self.dir, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(self.dir):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass


class CelebA64(ExternalDataset):
    """CelebA Dataset from [1].

    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
    with more than 200K celebrity images, each with 40 attribute annotations.
    The images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    10,177 number of identities, and 202,599 number of face images.

    Notes
    -----
    - Link : http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face
        attributes in the wild. In Proceedings of the IEEE international conference
        on computer vision (pp. 3730-3738).

    """

    urls = {
        "train": "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
    }
    files = {"train": "img_align_celeba"}
    shape = (3, 64, 64)
    missing_px_color = COLOUR_BLACK
    n_classes = 0  # not classification
    name = "celeba64"

    def __init__(self, root=DIR_DATA, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + "/*")

    def download(self):
        """Download the dataset."""
        save_path = os.path.join(self.dir, "celeba.zip")
        os.makedirs(self.dir)

        try:
            subprocess.check_call(
                ["curl", "-L", type(self).urls["train"], "--output", save_path]
            )
        except FileNotFoundError as e:
            raise Exception(e + " Please instal curl with `apt-get install curl`...")

        hash_code = "00d2c5bc6d35e252742224ab0c1e8fcb"
        assert (
            hashlib.md5(open(save_path, "rb").read()).hexdigest() == hash_code
        ), "{} file is corrupted.  Remove the file and try again.".format(save_path)

        with zipfile.ZipFile(save_path) as zf:
            self.logger.info("Extracting CelebA ...")
            zf.extractall(self.dir)

        os.remove(save_path)

        self.preprocess()

    def preprocess(self):
        self.logger.info("Resizing CelebA ...")
        preprocess(self.train_data, size=type(self).shape[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = plt.imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0


class CelebA32(CelebA64):
    shape = (3, 32, 32)
    name = "celeba32"


class CelebA(CelebA64):
    shape = (3, 218, 178)
    name = "celeba128"

    # use the default ones
    def preprocess(self):
        pass
