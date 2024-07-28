import csv
import logging
import random
import clip
import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18, resnet50


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
    from torch.utils.data import DataLoader, Subset
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
    def __init__(self, dataset: Dataset, list_txt: list[str], preprocess, ood=False):
        # Initialize image paths and corresponding texts
        self._dataset = dataset

        self.tokenized_title_dict = {c: clip.tokenize(f"a photo of a {c}") for c in list_txt}
        self.tokenized_title_list = [clip.tokenize(f"a photo of a {c}") for c in list_txt]
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.ColorJitter(contrast=0.5, brightness=1.0),
                                              transforms.ToPILImage()])
        # Load the model
        self._preprocess = preprocess
        self._ood = ood

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx, ood=False):
        image, label = self._dataset[idx]
        image = self._transform(image) if self._ood else image
        image = self._preprocess(image)
        title = self.tokenized_title_list[label]
        return image, label, title


def get_resnet(args):
    logging.info(f'Loading resnet model {args.resnet_ver}')
    renet_ctor = resnet18 if args.resnet_ver == resnet18 else resnet50
    model = renet_ctor(weights='DEFAULT')
    return model
