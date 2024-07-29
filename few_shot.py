import argparse
import gc
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm
from utils import set_logger, set_seed, get_data, get_device, get_resnet, save_model

IMAGE_SIZE = 224


def save_iteration_results(accuracy_list_on_ood_data: list[float], accuracy_list_on_standard_data: list[float]):
    """
    Save the iteration results of accuracy on OOD and standard data to files.

    Parameters:
    accuracy_list_on_ood_data (List[float]): List of accuracies on OOD data.
    accuracy_list_on_standard_data (List[float]): List of accuracies on standard data.

    Returns:
    None
    """

    with open('resnet50_accs_on_aug.npy', 'wb') as f:
        np.save(f, np.array(accuracy_list_on_ood_data))

    with open('resnet50_accs_on_original.npy', 'wb') as f:
        np.save(f, np.array(accuracy_list_on_standard_data))


@torch.no_grad()
def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    total: int = 0

    pbar = tqdm(loader, total=len(loader))
    running_correct, running_samples = 0., 0.
    all_targets = []
    all_preds = []

    for image, label in pbar:
        total += len(label)
        image, label = image.to(device), label.to(device)
        pred = model(image)

        predicted = torch.max(pred, dim=1)[1].cpu().numpy()

        running_correct += pred.argmax(1).eq(label).sum().item()
        running_samples += label.size(0)

        target = label.cpu().numpy().reshape(predicted.shape)

        all_targets += target.tolist()
        all_preds += predicted.tolist()
    # calculate confusion matrix
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    eval_accuracy = float((y_true == y_pred).sum().item()) / float(running_samples)
    return eval_accuracy * 100.0


def train_iteration(batch, device, model, optimizer):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    # Forward pass
    outputs = model(images)
    # Compute loss

    loss = criterion(outputs, labels)
    # Backward pass
    loss.backward()

    optimizer.step()

    return float(loss)


def train_epoch(device, num_epochs, optimizer,
                train_loader, validation_loaders: dict[str, (DataLoader, list[float])], model, epoch):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, total=len(train_loader))
    model.train()
    for batch in pbar:
        iteration_loss = train_iteration(batch, device, model, optimizer)
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


class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super(Backbone, self).__init__()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer

    def forward(self, x):
        return self.backbone(x)


class ClassificationHead(nn.Module):
    def __init__(self, num_features=2048, num_classes=196):
        super(ClassificationHead, self).__init__()
        self.conv = nn.Conv2d(num_features, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        cls_logits = self.cls_head(x)

        return cls_logits


class ClassificationModel(nn.Module):
    def __init__(self, backbone, classification_head, freeze_backbone=True, use_softmax=True):
        super(ClassificationModel, self).__init__()
        self.backbone = backbone
        self.classification_head = classification_head
        self._freeze_backbone = freeze_backbone
        self._use_softmax = use_softmax

    @property
    def freeze_backbone(self):
        return self._freeze_backbone

    @freeze_backbone.setter
    def freeze_backbone(self, value):
        self._freeze_backbone = value

    @property
    def classification_head(self):
        return self._classification_head

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        cls_logits = self.classification_head(features)

        out = F.softmax(cls_logits, dim=1) if self._use_softmax else cls_logits

        return out.reshape(x.shape[0], -1)


class ImageDatasetWrapper(Dataset):
    def __init__(self, dataset: Dataset, ood=False):
        # Initialize image paths and corresponding texts
        self._dataset = dataset

        self._transform = v2.Compose([
            v2.ToPILImage(),
            v2.Resize(IMAGE_SIZE),  # Resize the shorter side to 224 pixels
            v2.CenterCrop(IMAGE_SIZE),  # Crop the center to 224x224
            v2.ToTensor(),  # Convert the image to a PyTorch tensor
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize the image
                         std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self._ood_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.ColorJitter(contrast=0.5, brightness=1.0)])

        self._ood = ood

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx, ood=False):
        image, label = self._dataset[idx]

        if self._ood:
            image = self._ood_transform(image)

        assert self._transform is not None, f'transform not defined'
        image = self._transform(image)

        return image, label


def few_shot(args):
    device = get_device(args)
    train_set, validation_set, test_set, classes = get_data(args)

    num_classes = len(classes)

    distilled_resnet = get_resnet(args)
    benchmark_resnet = get_resnet(args)

    distilled_resnet.fc = nn.Linear(distilled_resnet.fc.in_features, args.distilled_out_dim)
    benchmark_resnet.fc = nn.Linear(benchmark_resnet.fc.in_features, num_classes)

    distilled_resnet.load_state_dict(
        torch.load(f'{args.load_path}/resnet_distilled_from_clip_on_StanfordCars_best_distilled.pt'))
    benchmark_resnet.load_state_dict(
        torch.load(f'{args.load_path}/resnet_distilled_from_clip_on_StanfordCars_best_benchmark.pt'))

    distilled_backbone = Backbone(distilled_resnet)
    benchmark_backbone = Backbone(benchmark_resnet)

    distilled_classifier = ClassificationModel(backbone=distilled_backbone, classification_head=ClassificationHead())
    benchmark_classifier = ClassificationModel(backbone=benchmark_backbone, classification_head=ClassificationHead())

    distilled_classifier.to(device)
    benchmark_classifier.to(device)

    train_set_original = ImageDatasetWrapper(train_set, ood=False)
    train_loader = DataLoader(train_set_original, batch_size=64, shuffle=True)
    train_set_ood = ImageDatasetWrapper(train_set, ood=True)
    validation_set_original = ImageDatasetWrapper(validation_set)
    validation_set_ood = ImageDatasetWrapper(validation_set, ood=True)
    train_loader_ood = DataLoader(train_set_ood, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set_original, batch_size=512, shuffle=False)
    validation_loader_ood = DataLoader(validation_set_ood, batch_size=512, shuffle=False)

    # Evaluate model - Baseline Accuracy

    distilled_classifier.eval()
    val_acc_baseline = evaluate(validation_loader, distilled_classifier, device)
    logging.info(f'Baseline validation accuracy {val_acc_baseline}')
    val_acc_ood_baseline = evaluate(validation_loader_ood, distilled_classifier, device)
    logging.info(f'Baseline validation accuracy OOD {val_acc_ood_baseline}')
    accuracy_list_on_original, accuracy_list_on_ood = [val_acc_baseline], [val_acc_ood_baseline]

    benchmark_classifier.eval()
    val_acc_baseline = evaluate(validation_loader, benchmark_classifier, device)
    logging.info(f'Benchmark Baseline validation accuracy {val_acc_baseline}')
    val_acc_ood_baseline = evaluate(validation_loader_ood, benchmark_classifier, device)
    logging.info(f'Benchmark Baseline validation accuracy OOD {val_acc_ood_baseline}')

    benchmark_accuracy_list_on_original, benchmark_accuracy_list_on_ood = [val_acc_baseline], [val_acc_ood_baseline]

    criterion = nn.CrossEntropyLoss()
    distilled_optimizer = torch.optim.Adam(distilled_classifier.classification_head.parameters(), lr=args.lr)
    benchmark_optimizer = torch.optim.Adam(benchmark_classifier.classification_head.parameters(), lr=args.lr)
    num_epochs = args.num_epochs_train_head
    # train classification head on in distribution datasets
    for epoch in range(num_epochs):
        running_loss = 0.0
        benchmark_running_loss = 0.0
        logging.info(f'Epoch {epoch}/{num_epochs}')
        pbar = tqdm(train_loader)

        distilled_classifier.train()
        benchmark_classifier.train()

        for data in pbar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            batch_size = labels.shape[0]

            distilled_optimizer.zero_grad()
            benchmark_optimizer.zero_grad()

            distilled_outputs = distilled_classifier(images)
            benchmark_outputs = benchmark_classifier(images)

            distilled_loss = criterion(distilled_outputs, labels)
            benchmark_loss = criterion(benchmark_outputs, labels)

            distilled_loss.backward()
            benchmark_loss.backward()

            distilled_optimizer.step()
            benchmark_optimizer.step()

            running_loss += distilled_loss.item()
            benchmark_running_loss += benchmark_loss.item()

            pbar.set_postfix({"running_loss": running_loss / float(batch_size),
                              "benchmark_running_loss": benchmark_running_loss / float(batch_size)})

        if (epoch + 1) % args.eval_every == 0:
            # Evaluate again after a single epoch on standard data
            val_acc_standard = evaluate(validation_loader, distilled_classifier, device)
            accuracy_list_on_original.append(val_acc_standard)

            logging.info(f'*** Validation after {epoch + 1} epochs detector train ***')
            logging.info(f'             Distilled Validation accuracy {val_acc_standard}')

            benchmark_val_acc_standard = evaluate(validation_loader, benchmark_classifier, device)
            benchmark_accuracy_list_on_original.append(benchmark_val_acc_standard)

            logging.info(f'             Benchmark Validation accuracy {benchmark_val_acc_standard}')

            val_acc_ood = evaluate(validation_loader_ood, distilled_classifier, device)
            accuracy_list_on_ood.append(val_acc_ood)

            logging.info(f'             Distilled Validation accuracy OOD {val_acc_ood}')

            benchmark_val_acc_ood = evaluate(validation_loader_ood, benchmark_classifier, device)
            benchmark_accuracy_list_on_ood.append(benchmark_val_acc_ood)
            logging.info(f'             Benchmark Validation accuracy OOD {benchmark_val_acc_ood}')

            save_model(args, distilled_classifier,
                       f'distilled_classifier_val_acc_standard_{val_acc_standard:.3f}'
                       f'_val_acc_ood_{val_acc_ood:.3f}')
            save_model(args, benchmark_classifier,
                       f'benchmark_classifier_val_acc_standard_{benchmark_val_acc_standard:.3f}'
                       f'_val_acc_ood_{benchmark_val_acc_ood:.3f}')

    # few shot OOD
    logging.info('Fine tuning on OOD data')

    # Let backbone weights learn too
    distilled_classifier.freeze_backbone = False
    benchmark_classifier.freeze_backbone = False

    distilled_optimizer = torch.optim.Adam(distilled_classifier.parameters(), lr=args.lr)
    benchmark_optimizer = torch.optim.Adam(benchmark_classifier.parameters(), lr=args.lr)

    num_epochs = args.num_epochs_fewshot
    for epoch in range(num_epochs):
        # Fine tune on OOD data and evaluate after each batch

        distilled_epoch_loss = train_epoch(device, num_epochs, distilled_optimizer
                                           , train_loader_ood,
                                           {'original': (validation_loader, accuracy_list_on_original),
                                            'OOD': (validation_loader_ood, accuracy_list_on_ood)},
                                           distilled_classifier, epoch)
        logging.info(f'Epoch {epoch} distilled_epoch_loss {distilled_epoch_loss}')
        benchmark_epoch_loss = train_epoch(device, num_epochs, benchmark_optimizer
                                           , train_loader_ood,
                                           {'original': (validation_loader,
                                                         benchmark_accuracy_list_on_original),
                                            'OOD': (validation_loader_ood, benchmark_accuracy_list_on_ood)},
                                           benchmark_classifier, epoch)
        logging.info(f'Epoch {epoch} benchmark_epoch_loss {benchmark_epoch_loss}')

    logging.info('Finished')


if __name__ == '__main__':
    data_name = 'StanfordCars'
    model_name = 'resnet50'

    parser = argparse.ArgumentParser(description="Few Shot Learning")

    ##################################
    #       Network args        #
    ##################################

    parser.add_argument("--model", type=str, default=model_name)

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
    parser.add_argument("--num-epochs-train-head", type=int, default=1)
    parser.add_argument("--num-epochs-fewshot", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")

    #############################
    #       General args        #
    #############################
    parser.add_argument("--resnet-ver", type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                        help="resnet18 or resnet50")
    parser.add_argument("--distilled-out-dim", type=int, default=512, help='Distilled model output dimension')

    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
    parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
    parser.add_argument("--save-path", type=str, default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--load-path",
                        type=str,
                        default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")
    parser.add_argument("--eval-every", type=int, default=1)

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

    few_shot(args)
