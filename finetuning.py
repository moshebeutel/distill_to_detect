import argparse
import csv
import logging
import time
from functools import partial
from pathlib import Path
import random
import clip
import numpy as np
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import gc


def save_iteration_results(accuracy_list_on_ood_data: list[float], accuracy_list_on_standard_data: list[float]):
    """
    Save the iteration results of accuracy on OOD and standard data to files.

    Parameters:
    accuracy_list_on_ood_data (List[float]): List of accuracies on OOD data.
    accuracy_list_on_standard_data (List[float]): List of accuracies on standard data.

    Returns:
    None
    """
    with open('accs_on_aug.npy', 'wb') as f:
        np.save(f, np.array(accuracy_list_on_ood_data))

    with open('accs_on_original.npy', 'wb') as f:
        np.save(f, np.array(accuracy_list_on_standard_data))


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


def save_model(args, model, suff):
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    file_path = Path(args.save_path) / f'clip_finetuned_to_{args.data_name}_{time.asctime()}_{suff}.pt'
    torch.save(model.state_dict(), file_path.as_posix())


def get_model(args):
    logging.info(f'Loading model {args.model}')
    model, preprocess = clip.load(args.model, device=get_device(args), jit=False)
    num_of_params = sum([torch.numel(l) for l in model.parameters()])
    logging.info(f'Loaded model num_of_params {num_of_params}')
    return model, preprocess


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


def get_data(args) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    # train_set = CIFAR10(root, train=True, download=True)
    train_set = torchvision.datasets.ImageFolder(root=f'{args.data_path}/train')
    train_set, validation_set = train_val_split(train_set)
    # test_set = CIFAR10(root, train=False, download=True)
    test_set = torchvision.datasets.ImageFolder(root=f'{args.data_path}/test')
    # classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    with open('/home/user1/datasets/StanfordCars/names.csv', 'r') as f:
        reader = csv.reader(f)
        classes = [row[0] for row in reader]
    logging.info(f'Classes {classes}')
    return train_set, validation_set, test_set, classes


def get_device(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )
    logging.info(f'Use device: {device}')
    return device


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


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


def convert_models_to_mix(model):
    clip.model.convert_weights(model)


def freeze_embed(model):
    freeze_list: list[str] = ['positional_embedding', 'text_projection', 'logit_scale',
                              'visual.class_embedding',
                              'visual.positional_embedding', 'visual.proj', 'visual.conv1.weight',
                              'visual.ln_pre.weight', 'visual.ln_pre.bias']
    for n, p in model.named_parameters():
        p.requires_grad = n not in freeze_list


@torch.no_grad()
def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    total: int = 0
    correct: int = 0
    text_inputs = torch.cat(loader.dataset.tokenized_title_list).to(device)
    pbar = tqdm(loader, total=len(loader))
    for image, label, _ in pbar:
        total += len(label)
        image = image.to(device)
        label = label.to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity.max(1)

        correct += (indices == label).sum().item()

        image, label, image_features, text_features, similarity, values, indices = (
            image.cpu(), label.cpu(), image_features.cpu(), text_features.cpu(), similarity.cpu(), values.cpu(),
            indices.cpu())
        del image, label, image_features, text_features, similarity, values, indices

    accuracy: float = 100.0 * float(correct) / float(total)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return accuracy


def train_iteration(batch, device, loss_img, loss_txt, model, optimizer):
    freeze_embed(model)
    optimizer.zero_grad()
    images, _, texts = batch
    images = images.to(device)
    texts = texts.squeeze()
    texts = texts.to(device)
    # Forward pass
    logits_per_image, logits_per_text = model(images, texts)
    # Compute loss
    ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
    # Backward pass
    total_loss.backward()

    if device == "cpu":
        optimizer.step()
    else:
        convert_models_to_fp32(model)
        optimizer.step()
        convert_models_to_mix(model)
    return float(total_loss)


def train_epoch(device, loss_img, loss_txt, num_epochs, optimizer,
                train_loader, validation_loaders: dict[str, (DataLoader, list[float])], model, epoch):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    model.train()
    for batch in pbar:
        iteration_loss = train_iteration(batch, device, loss_img, loss_txt, model, optimizer)
        epoch_loss += iteration_loss
        model.eval()
        s = f"Epoch {epoch}/{num_epochs}"
        for loader_name in validation_loaders:
            validation_loader, acc_list = validation_loaders[loader_name]
            val_acc: float = evaluate(validation_loader, model, device)
            acc_list.append(val_acc)
            s += f" Validation Acc {loader_name}: {val_acc}"
        if 'original' in validation_loaders and 'OOD' in validation_loaders:
            save_iteration_results(accuracy_list_on_standard_data=validation_loaders['original'][1],
                                   accuracy_list_on_ood_data=validation_loaders['OOD'][1])
        pbar.set_description(s + f"Loss: {iteration_loss:.4f}")

    return epoch_loss


def finetune(args):
    device = get_device(args)
    model, preprocess = get_model(args)
    train_set, validation_set, test_set, classes = get_data(args)

    train_set_original = ImageTitleDatasetWrapper(train_set, classes, preprocess)
    train_set_ood = ImageTitleDatasetWrapper(train_set, classes, preprocess, ood=True)
    validation_set_original = ImageTitleDatasetWrapper(validation_set, classes, preprocess)
    validation_set_ood = ImageTitleDatasetWrapper(validation_set, classes, preprocess, ood=True)
    test_set = ImageTitleDatasetWrapper(test_set, classes, preprocess)
    train_loader = DataLoader(train_set_original, batch_size=64, shuffle=True)
    train_loader_ood = DataLoader(train_set_ood, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set_original, batch_size=512, shuffle=False)
    validation_loader_ood = DataLoader(validation_set_ood, batch_size=512, shuffle=False)

    # Evaluate model - Baseline Accuracy
    model.eval()
    val_acc_baseline = evaluate(validation_loader, model, device)
    logging.info(f'Baseline validation accuracy {val_acc_baseline}')
    val_acc_ood_baseline = evaluate(validation_loader_ood, model, device)
    logging.info(f'Baseline validation accuracy OOD {val_acc_ood_baseline}')

    accuracy_list_on_original, accuracy_list_on_ood = [val_acc_baseline], [val_acc_ood_baseline]

    # fine tuning on standard data
    for epoch in range(args.finetune_epochs):
        # Train a single epoch on standard data
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        train_epoch(device, loss_img, loss_txt, 1, optimizer,
                    train_loader, {}, model, 0)

        # Evaluate again after a single epoch on standard data
        val_acc_standard = evaluate(validation_loader, model, device)
        accuracy_list_on_original.append(val_acc_standard)
        logging.info(f'*** Validation after {epoch + 1} epochs of fine tune regular data ***')
        logging.info(f'             Validation accuracy {val_acc_standard}')
        val_acc_ood = evaluate(validation_loader_ood, model, device)
        accuracy_list_on_ood.append(val_acc_ood)
        logging.info(f'             Validation accuracy OOD {val_acc_ood}')

        save_model(args, model, f'val_acc_standard_{val_acc_standard:.3f}_val_acc_ood_{val_acc_ood:.3f}')

    if args.finetune_on_ood:
        # Fine tune on OOD data and evaluate after each batch
        num_epochs = args.finetune_ood_epochs
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        train_epoch_fn = partial(train_epoch, device, loss_img, loss_txt, num_epochs, optimizer
                                 , train_loader_ood,
                                 {'original': (validation_loader, accuracy_list_on_original),
                                  'OOD': (validation_loader_ood, accuracy_list_on_ood)})
        logging.info('Fine tuning on OOD data')
        for epoch in range(num_epochs):
            epoch_loss = train_epoch_fn(model, epoch)
            logging.info(f'Epoch {epoch} loss {epoch_loss}')

    logging.info('Finished')


if __name__ == '__main__':
    data_name = 'StanfordCars'
    model_name = 'ViT-B/32'

    parser = argparse.ArgumentParser(description="Distillation Learning")

    ##################################
    #       Network args        #
    ##################################

    parser.add_argument("--model", type=str, default=model_name)
    # parser.add_argument("--num-blocks", type=int, default=3)
    # parser.add_argument("--block-size", type=int, default=3)

    #############################
    #       Dataset Args        #
    #############################

    parser.add_argument(
        "--data-name", type=str, default=data_name,
        choices=['cifar10', 'StanfordCars'], help="dataset name"
    )
    parser.add_argument("--data-path", type=str, default=Path.home() / f'datasets/{data_name}',
                        help="path for dataset")

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--save-path", type=str, default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")

    parser.add_argument("--finetune-on-ood", type=bool, default=True, help="preform fine tune on ood data")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="num epochs of fine tune")
    parser.add_argument("--finetune-ood-epochs", type=int, default=3, help="num epochs of fine tune")

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

    args['exp-name'] = f'finetune_{args.model}_to_dataset_{args.data_name}_' \
                       f'optimizer_{args.optimizer}_batch_size_{args.batch_size}' \
                       f'_lr_{args.lr}_wd_{args.wd}'

    finetune(args)
