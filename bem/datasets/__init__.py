import json
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10, CelebA, MNIST, ImageNet, ImageFolder
from torch.utils.data import TensorDataset
import torch.utils.data
from torch.utils.data import Subset, Dataset
import pickle
import numpy as np
from PIL import Image

from .Data import Generator

import argparse
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


image_datasets = [
    "CIFAR10",
    "MINI_CIFAR10",
    "CIFAR10_LT",
    "MNIST",
    "CELEBA",
    "CELEBA_HQ",
]

# Tensor-based datasets that share the (N, C, H, W) layout of images but whose
# values are raw continuous floats (no [-1, 1] affine, no PNG dumps).
# HRRR fields are z-scored per channel from the training years (see hrrr.py).
tensor_datasets = [
    'HRRR_VIL',
    'HRRR_W20',
]

def is_image_dataset(name):
    return name.upper() in [x.upper() for x in image_datasets]

def affine_transform(x):
    return 2*x -1

def inverse_affine_transform(x):
    return (x + 1) / 2

DATA_PATH = './data'


def _default_long_tail_config(config):
    num_classes = getattr(config.data, 'num_classes', 10)
    return {
        'enabled': False,
        'imbalance_ratio': 0.01,
        'max_count': 5000,
        'num_classes': num_classes,
        'seed': 42,
        'metadata_dir': os.path.join(DATA_PATH, 'splits'),
    }


def _get_long_tail_config(config):
    lt_config = _default_long_tail_config(config)
    if hasattr(config.data, 'long_tail'):
        lt_config.update(vars(config.data.long_tail))

    lt_config['num_classes'] = int(lt_config['num_classes'])
    lt_config['max_count'] = int(lt_config['max_count'])
    lt_config['seed'] = int(lt_config['seed'])
    lt_config['imbalance_ratio'] = float(lt_config['imbalance_ratio'])

    assert lt_config['num_classes'] > 1, 'long_tail.num_classes must be greater than 1'
    assert lt_config['max_count'] > 0, 'long_tail.max_count must be positive'
    assert 0 < lt_config['imbalance_ratio'] <= 1, 'long_tail.imbalance_ratio must be in (0, 1]'
    return lt_config


def _compute_long_tail_counts(long_tail_config):
    max_count = long_tail_config['max_count']
    imbalance_ratio = long_tail_config['imbalance_ratio']
    num_classes = long_tail_config['num_classes']
    denominator = num_classes - 1
    return [
        int(np.floor(max_count * (imbalance_ratio ** (class_id / denominator))))
        for class_id in range(num_classes)
    ]


def _long_tail_metadata_path(dataset_name, long_tail_config):
    dataset_prefix = dataset_name.lower()
    if not dataset_prefix.endswith('_lt'):
        dataset_prefix = f'{dataset_prefix}_lt'
    filename = (
        f"{dataset_prefix}_"
        f"max{long_tail_config['max_count']}_"
        f"ratio{long_tail_config['imbalance_ratio']}_"
        f"classes{long_tail_config['num_classes']}_"
        f"seed{long_tail_config['seed']}.json"
    )
    return os.path.join(long_tail_config['metadata_dir'], filename)


def _validate_long_tail_indices(targets, indices, expected_counts):
    index_array = np.array(indices, dtype=np.int64)
    selected_targets = targets[index_array]
    final_counts = np.bincount(selected_targets, minlength=len(expected_counts))
    assert final_counts.tolist() == expected_counts, (
        f'Long-tail split counts mismatch. Expected {expected_counts}, got {final_counts.tolist()}'
    )
    return final_counts.tolist()


def _build_or_load_long_tail_indices(targets, dataset_name, long_tail_config):
    metadata_path = _long_tail_metadata_path(dataset_name, long_tail_config)
    expected_counts = _compute_long_tail_counts(long_tail_config)

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        selected_indices = metadata['selected_indices']
        final_counts = _validate_long_tail_indices(targets, selected_indices, expected_counts)
        print(f'Loaded long-tail split metadata from {metadata_path}')
        print(f'Long-tail per-class counts: {final_counts}')
        return selected_indices

    rng = np.random.default_rng(long_tail_config['seed'])
    selected_indices = []
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    for class_id, class_count in enumerate(expected_counts):
        class_indices = np.flatnonzero(targets == class_id)
        assert len(class_indices) >= class_count, (
            f'Class {class_id} only has {len(class_indices)} samples, cannot keep {class_count}'
        )
        sampled_indices = rng.choice(class_indices, size=class_count, replace=False)
        sampled_indices = np.sort(sampled_indices)
        selected_indices.extend(sampled_indices.tolist())
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        print(f'Long-tail target count for class {class_id} ({class_name}): {class_count}')

    selected_indices = sorted(selected_indices)
    final_counts = _validate_long_tail_indices(targets, selected_indices, expected_counts)
    os.makedirs(long_tail_config['metadata_dir'], exist_ok=True)
    metadata = {
        'dataset': dataset_name,
        'seed': long_tail_config['seed'],
        'imbalance_ratio': long_tail_config['imbalance_ratio'],
        'max_count': long_tail_config['max_count'],
        'num_classes': long_tail_config['num_classes'],
        'selected_indices': selected_indices,
        'expected_counts': expected_counts,
        'final_counts': final_counts,
        'class_order': {
            '0': 'airplane',
            '1': 'automobile',
            '2': 'bird',
            '3': 'cat',
            '4': 'deer',
            '5': 'dog',
            '6': 'frog',
            '7': 'horse',
            '8': 'ship',
            '9': 'truck',
        },
        'formula': 'floor(max_count * (imbalance_ratio ** (class_id / (num_classes - 1))))',
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Saved long-tail split metadata to {metadata_path}')
    print(f'Long-tail per-class counts: {final_counts}')
    return selected_indices


def _build_cifar10_datasets(config):
    if config.data.random_flip is False:
        train_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor(), transforms.Lambda(affine_transform)]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Lambda(affine_transform),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor(), transforms.Lambda(affine_transform)]
        )

    dataset = CIFAR10(
        os.path.join(DATA_PATH, 'cifar10'),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = CIFAR10(
        os.path.join(DATA_PATH, 'cifar10_test'),
        train=False,
        download=True,
        transform=test_transform,
    )
    return dataset, test_dataset

def get_dataset(p):

    # retrocompatibility with LIM original code
    config = dict2namespace(p)

    # Tensor-based datasets (HRRR) are handled before the image / 2D assertion below.
    if config.data.dataset.upper() in [x.upper() for x in tensor_datasets]:
        name = config.data.dataset.upper()
        if name == 'HRRR_VIL':
            from .hrrr import get_hrrr_datasets
            return get_hrrr_datasets(config, default_channel='vil')
        if name == 'HRRR_W20':
            from .hrrr import get_hrrr_datasets
            return get_hrrr_datasets(config, default_channel='w20')
        raise ValueError(f"Unknown tensor dataset: {config.data.dataset}")

    assert (config.data.dataset.upper() in [x.upper() for x in image_datasets]) \
        or (config.data.dataset.upper() in [x.upper() for x in Generator.available_distributions]), \
        "Dataset not available: {}.\nCan choose from:\n(Image)\t{}\n(2d data)\t{}".format(config.data.dataset,
                                                                                          image_datasets,
                                                                                          Generator.available_distributions)
    
    if config.data.dataset.lower() in [x.lower() for x in Generator.available_distributions]:
        # for the moment, only supports dim <= 2
        assert config.data.dim <= 2, 'Only supports at most N-D data with N <= 2 for the moment. Please change Data.generator and Data.Distributions objects.'

        data_gen = Generator(config.data.dataset.lower(),
                                    n = int(np.sqrt(config.data.n_mixture)) if config.data.dataset.split('_')[-1] == 'grid' else config.data.n_mixture,
                                    std = config.data.std,
                                    normalize = config.data.normalized,
                                    weights = config.data.weights,
                                    theta = config.data.theta,
                                    alpha = config.data.data_alpha,
                                    isotropic = config.data.isotropic,
                                    between_minus_1_1 = config.data.between_minus_1_1,
                                    quantile_cutoff = config.data.quantile_cutoff,
                                    grid_spacing = getattr(config.data, 'grid_spacing', 1.0),
                                    )
        data_gen.generate(n_samples = config.data.nsamples)
        # possibly remove a dimension if nfeatures == 1
        if config.data.dim == 1:
            data_gen.samples = data_gen.samples[:, 0].unsqueeze(1)
        # add channel
        data_gen.samples = data_gen.samples.unsqueeze(1)
        import copy
        # train_data
        dataset = TensorDataset(copy.deepcopy(data_gen.samples), torch.tensor([0.]).repeat(data_gen.samples.shape[0]))
        #dataset = TensorDataset(data_gen.samples)
        data_gen.generate(n_samples = config.data.nsamples)
        # possibly remove a dimension if nfeatures == 1
        if config.data.dim == 1:
            data_gen.samples = data_gen.samples[:, 0].unsqueeze(1)
        # add channel
        data_gen.samples = data_gen.samples.unsqueeze(1)
        #test_data
        test_dataset = TensorDataset(copy.deepcopy(data_gen.samples), torch.tensor([0.]).repeat(data_gen.samples.shape[0]))
        return dataset, test_dataset


    config.data.dataset = config.data.dataset.upper()
    print('loading dataset {}'.format(config.data.dataset))
    if config.data.dataset == "CIFAR10":
        dataset, test_dataset = _build_cifar10_datasets(config)
    elif config.data.dataset == "CIFAR10_LT":
        dataset, test_dataset = _build_cifar10_datasets(config)
        print('keeping CIFAR-10 test split unchanged')
        long_tail_config = _get_long_tail_config(config)
        long_tail_config['enabled'] = True
        targets = np.array(dataset.targets)
        initial_counts = np.bincount(targets, minlength=int(long_tail_config['num_classes'])).tolist()
        print(f'Initial CIFAR-10 class count: {initial_counts}')
        selected_indices = _build_or_load_long_tail_indices(targets, config.data.dataset, long_tail_config)
        dataset = Subset(dataset, selected_indices)
    elif config.data.dataset == "MINI_CIFAR10":
        dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "mini_cifar10"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )
        test_dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "mini_cifar10"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )
    
    elif config.data.dataset == "MNIST":
        mnist_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.CenterCrop(config.data.image_size),
                transforms.ToTensor(),
                transforms.Lambda(affine_transform),
            ]
        )
        dataset = MNIST(
            os.path.join(DATA_PATH, "mnist"),
            train=True,
            download=True,
            transform=mnist_transform,
        )
        test_dataset = MNIST(
            os.path.join(DATA_PATH, "mnist_test"),
            train=False,
            download=True,
            transform=mnist_transform, #test_transform,
        )
        
    elif config.data.dataset == "CELEBA":
        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64
        if config.data.random_flip:
            dataset = CelebA(
                root=os.path.join(DATA_PATH),#, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Lambda(affine_transform)
                    ]
                ),
                download=True,
            )
        else:
            dataset = CelebA(
                root=os.path.join(DATA_PATH),#, "celeba"),
                split="train",
                transform=transforms.Compose(
                    [
                        Crop(x1, x2, y1, y2),
                        transforms.Resize(config.data.image_size),
                        transforms.ToTensor(),
                        transforms.Lambda(affine_transform)
                    ]
                ),
                download=True,
            )

        test_dataset = CelebA(
            root=os.path.join(DATA_PATH),#, "celeba"),
            split="test",
            transform=transforms.Compose(
                [
                    Crop(x1, x2, y1, y2),
                    transforms.Resize(config.data.image_size),
                    transforms.ToTensor(),
                    transforms.Lambda(affine_transform)
                ]
            ),
            download=True,
        )
    
    elif config.data.dataset == "CELEBA_HQ":
        if config.data.random_flip:
            dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "celebahq"),
                transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(p=0.5),
                     transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )
        else:
            dataset = ImageFolder(
                root=os.path.join(DATA_PATH, "celebahq"),
                transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(config.data.image_size),
                     transforms.Lambda(affine_transform)]
                ),
            )

        num_items = len(dataset)
        indices = list(range(num_items))
        random_state = np.random.get_state()
        np.random.seed(2019)
        np.random.shuffle(indices)
        np.random.set_state(random_state)
        train_indices, test_indices = (
            indices[: int(num_items * 0.9)],
            indices[int(num_items * 0.9):],
        )
        test_dataset = Subset(dataset, test_indices)
        dataset = Subset(dataset, train_indices)        

    else:
        dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = (2. * X - 1.0)*5.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 5.0) / 10.0

    return torch.clamp(X, 0.0, 1.0)
