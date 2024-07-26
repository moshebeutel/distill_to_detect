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
from torchvision.models import resnet18
from tqdm import tqdm
import gc

def get_teacher(args):
    logging.info(f'Loading model {args.teacher}')
    model, preprocess = clip.load(args.teacher, device=get_device(args), jit=False)
    model.load_state_dict(torch.load('/home/user1/saved_models/clip/clip_finetuned_to_StanfordCars_Fri Jul 26 14:31:50 2024_val_acc_standard_71.700_val_acc_ood_66.544.pt'))
    num_of_params = sum([torch.numel(l) for l in model.parameters()])
    logging.info(f'Loaded model num_of_params {num_of_params}')
    return model, preprocess

def get_student(args):
    logging.info(f'Loading student model {args.student}')
    student_model = resnet18(pretrained=False)  # or resnet50
    num_of_params = sum([torch.numel(l) for l in student_model.parameters()])
    logging.info(f'Loaded student model num_of_params {num_of_params}')
    return student_model


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


@torch.no_grad()
def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    total: int = 0
    correct: int = 0
    criteria = torch.nn.CrossEntropyLoss()
    pbar = tqdm(loader, total=len(loader))
    running_loss, running_correct, running_samples = 0., 0., 0.
    all_targets = []
    all_preds = []

    for image, label, _ in pbar:
        total += len(label)
        image, label = image.to(device), label.to(device)
        pred = model(image)

        loss = criteria(pred, label)
        predicted = torch.max(pred, dim=1)[1].cpu().numpy()

        running_loss += (loss.item() / label.size(0))
        running_correct += pred.argmax(1).eq(label).sum().item()
        running_samples += label.size(0)

        target = label.cpu().numpy().reshape(predicted.shape)

        all_targets += target.tolist()
        all_preds += predicted.tolist()
    # calculate confusion matrix
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    running_loss /= running_samples
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    eval_accuracy = float((y_true == y_pred).sum().item()) / float(running_samples)
    return eval_accuracy * 100.0

def distill(args):
    device = get_device(args)
    teacher, preprocess = get_teacher(args)
    student = get_student(args)
    student.fc = nn.Linear(student.fc.in_features, teacher.visual.output_dim)
    student = student.to(device)
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

    accuracy_list_on_original, accuracy_list_on_ood = [], []

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        student.train()  # Set student model to training mode
        running_loss = 0.0

        for images, _ , _ in train_loader:
            images = images.to(device).float()
            optimizer.zero_grad()

            # Get teacher model (CLIP) output
            with torch.no_grad():
                teacher_outputs = teacher.encode_image(images)

            # Get student model output
            student_outputs = student(images)

            # Compute loss
            loss = criterion(student_outputs, teacher_outputs.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Evaluate again after a single epoch on standard data
            val_acc_standard = evaluate(validation_loader, student, device)
            accuracy_list_on_original.append(val_acc_standard)
            logging.info(f'*** Validation after {epoch + 1} epochs of fine tune regular data ***')
            logging.info(f'             Validation accuracy {val_acc_standard}')
            val_acc_ood = evaluate(validation_loader_ood, student, device)
            accuracy_list_on_ood.append(val_acc_ood)
            logging.info(f'             Validation accuracy OOD {val_acc_ood}')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training completed.")



def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Distillation Learning")
    data_name = 'StanfordCars'
    model_name = 'ViT-B/32'
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
    #       Network args        #
    ##################################

    parser.add_argument("--teacher", type=str, default=model_name)
    parser.add_argument("--student", type=str, default='resnet18')
    # parser.add_argument("--num-blocks", type=int, default=3)
    # parser.add_argument("--block-size", type=int, default=3)

    ##################################
    #       Optimization args        #
    ##################################
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")

    args = parser.parse_args()

    set_logger()
    # set_seed(args.seed)

    exp_name = f'Distillation_{args.teacher}_to_{args.student}_' \
               f'optimizer_{args.optimizer}_batch_size_{args.batch_size}' \
               f'_lr_{args.lr}_wd_{args.wd}'

    distill(args)
