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
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import gc

def get_teacher(args):
    logging.info(f'Loading model {args.teacher}')
    model, preprocess = clip.load(args.teacher, device=get_device(args), jit=False)
    model.load_state_dict(torch.load(f'{args.load_path}/clip_finetuned_to_StanfordCars_best.pt'))
    num_of_params = sum([torch.numel(l) for l in model.parameters()])
    logging.info(f'Loaded model num_of_params {num_of_params}')
    return model, preprocess

def get_resnet(args):
    logging.info(f'Loading benchmark model {args.resnet_ver}')
    renet_ctor = resnet18 if args.resnet_ver == resnet18 else resnet50
    model = renet_ctor(weights='DEFAULT')
    return model

def get_student(args):
    logging.info(f'Loading student model {args.student}')
    renet_ctor = resnet18 if args.student == resnet18 else resnet50
    student_model = renet_ctor(weights=None)  # or resnet50
    # student_model.load_state_dict(torch.load(f'{args.load_path}/resnet_distilled_from_clip_on_StanfordCars_best.pt'))
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

    with open(f'{args.data_path}/names.csv', 'r') as f:
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

def save_model(args, model, suff):
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    file_path = Path(args.save_path) / f'resnet_distilled_from_clip_on_{args.data_name}_{time.asctime()}_{suff}.pt'
    torch.save(model.state_dict(), file_path.as_posix())

def distill(args):
    device = get_device(args)
    teacher, preprocess = get_teacher(args)
    student = get_student(args)
    student.fc = nn.Linear(student.fc.in_features, teacher.visual.output_dim)
    student = student.to(device)
    student.load_state_dict(torch.load(f'{args.load_path}/resnet_distilled_from_clip_on_StanfordCars_best.pt'))

    benchmark = get_resnet(args)
    
    train_set, validation_set, test_set, classes = get_data(args)
    benchmark.fc = nn.Linear(benchmark.fc.in_features, len(classes))
    benchmark = benchmark.to(device)

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
    benchmark_accuracy_list_on_original, benchmark_accuracy_list_on_ood = [], []

    # criterion = nn.CosineEmbeddingLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    benchmark_optimizer = torch.optim.Adam(benchmark.parameters(), lr=0.001)
    benchmark_criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    T=2.0
    soft_target_loss_weight = 0.1
    ce_loss_weight = 1.0 - soft_target_loss_weight
    teacher.eval()

    for epoch in range(num_epochs):
        student.train()  # Set student model to training mode
        benchmark.train()
        running_loss = 0.0
        logging.info(f'Epoch {epoch}/{num_epochs}')
        pbar = tqdm(train_loader)
        total = 0
        for images, labels , _ in pbar:
            images = images.to(device).float()
            labels = labels.to(device)
            batch_size = labels.shape[0]
            total += batch_size
            optimizer.zero_grad()
            benchmark_optimizer.zero_grad()
            # Get teacher model (CLIP) output
            with torch.no_grad():
                teacher_logits = teacher.encode_image(images)


            
            # Forward pass with the student model
            student_logits = student(images)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = criterion(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            benchmark_outputs = benchmark(images)
            benchmark_loss = benchmark_criterion(benchmark_outputs, labels)

            # backprop
            loss.backward()
            optimizer.step()

            
            benchmark_loss.backward()
            benchmark_optimizer.step()

            running_loss += loss.item() 

            pbar.set_postfix({"running_loss": running_loss / float(batch_size)})
        if (epoch + 1) % args.eval_every == 0: 
          # Evaluate again after a single epoch on standard data
          val_acc_standard = evaluate(validation_loader, student, device)
          accuracy_list_on_original.append(val_acc_standard)
          logging.info(f'*** Validation after {epoch + 1} epochs of fine tune regular data ***')
          logging.info(f'             Student Validation accuracy {val_acc_standard}')
          benchmark_val_acc_standard = evaluate(validation_loader, benchmark, device)
          benchmark_accuracy_list_on_original.append(benchmark_val_acc_standard)
          logging.info(f'             Benchmark Validation accuracy {benchmark_val_acc_standard}')
          val_acc_ood = evaluate(validation_loader_ood, student, device)
          accuracy_list_on_ood.append(val_acc_ood)
          logging.info(f'             Student Validation accuracy OOD {val_acc_ood}')
          benchmark_val_acc_ood = evaluate(validation_loader_ood, benchmark, device)
          benchmark_accuracy_list_on_ood.append(benchmark_val_acc_ood)
          logging.info(f'             Benchmark Validation accuracy OOD {benchmark_val_acc_ood}')
          
          save_model(args, student, f'val_acc_standard_{val_acc_standard:.3f}_val_acc_ood_{val_acc_ood:.3f}')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training completed.")

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
    parser.add_argument("--student", type=str, default='resnet50')
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
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument("--save-path", type=str, default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--load-path", type=str, default='',
                        help="dir path for checkpoints")
    parser.add_argument("--resnet-ver", type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                        help="resnet18 or resnet50")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")
    parser.add_argument("--eval-every", type=int, default=10)

    

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

   

    distill(args)
