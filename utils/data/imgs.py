from .datasets import *
from .helpers import train_dev_split

DATASETS_DICT = {
    "mnist": MNIST,
    "svhn": SVHN,
    "celeba32": CelebA32,
    "celeba64": CelebA64,
    "zs-multi-mnist": ZeroShotMultiMNIST,
    "zsmm": ZeroShotMultiMNIST,  # shorthand
    "zsmmt": ZeroShotMultiMNISTtrnslt,
    "zsmms": ZeroShotMultiMNISTscale,
    "zs-mnist": ZeroShotMNIST,
    "celeba": CelebA,
    "celeba128": CelebA128,
}

DATASETS = list(DATASETS_DICT.keys())


# HELPERS
def get_train_test_img_dataset(dataset):
    """Return the correct instantiated train and test datasets."""
    try:
        train_dataset = get_dataset(dataset)(split="train")
        test_dataset = get_dataset(dataset)(split="test")
    except TypeError as e:
        train_dataset, test_dataset = train_dev_split(
            get_dataset(dataset)(), dev_size=0.1, is_stratify=False
        )

    return train_dataset, test_dataset


def get_dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        return DATASETS_DICT[dataset]
    except KeyError:
        raise ValueError("Unknown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).shape


def get_test_upscale_factor(dataset):
    """Return the correct image size."""
    try:
        dataset = get_dataset(dataset)
        return dataset.shape_test[-1] / dataset.shape[-1]
    except (AttributeError, ValueError):
        return 1
