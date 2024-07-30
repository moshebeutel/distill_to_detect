import csv
import logging
import random
import time
from pathlib import Path
import clip
import numpy as np
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import resnet18, resnet50
from torchvision.transforms import v2


def initialize_weights(module: nn.Module):
    for m in module.modules():

        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()



def set_seed(seed, cudnn_enabled=True):
    """for reproducibility

    :param seed:
    :return:
    """

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )


def get_device(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    logging.info(f'Use device: {device}')
    return device


def get_data(args) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    # train_set = CIFAR10(root, train=True, download=True)
    train_set = torchvision.datasets.ImageFolder(root=f'{args.data_path}/train')
    train_set, validation_set = train_val_split(train_set)
    # test_set = CIFAR10(root, train=False, download=True)
    test_set = torchvision.datasets.ImageFolder(root=f'{args.data_path}/test')
    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with open(f'{args.data_path}/names.csv', 'r') as f:
        reader = csv.reader(f)
        classes = [row[0] for row in reader]
    logging.info(f'Classes {classes}')
    return train_set, validation_set, test_set, classes


def train_val_split(train_set: Dataset, split_ratio: float = 0.8, shuffle: bool = True) -> tuple[Dataset, Dataset]:
    from torch.utils.data import Subset
    dataset_size = len(train_set)
    split_index = int(dataset_size * split_ratio)
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)
    # Create Subset objects for the training and validation sets
    train_subset = Subset(train_set, indices[:split_index])
    validation_subset = Subset(train_set, indices[split_index:])
    return train_subset, validation_subset


class ImageTitleDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, list_txt: list[str], preprocess, ood=False, severity=5):
        # Initialize image paths and corresponding texts
        self._dataset = dataset

        self.tokenized_title_dict = {c: clip.tokenize(f"a photo of a {c}") for c in list_txt}
        self.tokenized_title_list = [clip.tokenize(f"a photo of a {c}") for c in list_txt]
        # Load the model
        self._preprocess = preprocess
        self._ood = ood

        noise_scale = [0.04, 0.06, .08, .09, .10][severity - 1]
        self._transform_ood = v2.Compose([v2.ToImage(),
                                          v2.ToDtype(torch.float32, scale=True),
                                          v2.GaussianNoise(sigma=noise_scale, clip=True),
                                          v2.ToDtype(torch.uint8, scale=True),
                                          v2.ToPILImage()])

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx, ood=False):
        image, label = self._dataset[idx]
        image = self._transform_ood(image) if self._ood else image
        image = self._preprocess(image)
        title = self.tokenized_title_list[label]
        return image, label, title


def get_resnet(args):
    logging.info(f'Loading resnet model {args.resnet_ver}')
    renet_ctor = resnet18 if args.resnet_ver == "resnet18" else resnet50
    model = renet_ctor(weights='DEFAULT')
    return model


def save_model(args, model, suff):
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    file_path = Path(args.save_path) / f'resnet_distilled_from_clip_on_{args.data_name}_{time.asctime()}_{suff}.pt'
    torch.save(model.state_dict(), file_path.as_posix())
