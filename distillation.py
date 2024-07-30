import argparse
import csv
import gc
import logging
from pathlib import Path
import clip
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
from utils import set_seed, set_logger, get_device, get_data, ImageTitleDatasetWrapper, get_resnet, train_val_split, \
    save_model


def get_teacher(args):
    logging.info(f'Loading model {args.teacher}')
    model, preprocess = clip.load(args.teacher, device=get_device(args), jit=False)
    model.load_state_dict(torch.load(f'{args.load_path}/best_model.pt'))
    num_of_params = sum([torch.numel(l) for l in model.parameters()])
    logging.info(f'Loaded model num_of_params {num_of_params}')
    return model, preprocess


def get_student(args):
    logging.info(f'Loading student model {args.student}')
    renet_ctor = resnet18 if args.student == resnet18 else resnet50
    student_model = renet_ctor(weights='DEFAULT')  # or resnet50

    # student_model.load_state_dict(torch.load(f'{args.load_path}/resnet_distilled_from_clip_on_StanfordCars_best.pt'))
    num_of_params = sum([torch.numel(l) for l in student_model.parameters()])
    logging.info(f'Loaded student model num_of_params {num_of_params}')
    return student_model


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


def get_vision_language_alignment_loss(args, teacher_vision_output, teacher_language_output, student_output):
    # Calculate outputs norm
    student_outputs_norm = nn.functional.normalize(student_output, dim=-1)
    teacher_vision_outputs_norm = nn.functional.normalize(teacher_vision_output, dim=-1)
    teacher_language_outputs_norm = nn.functional.normalize(teacher_language_output, dim=-1)

    # Calculate open-set classification output logits - text feature logits has 2 dims
    student_outputs_for_align = torch.einsum('ni,ci->nc', student_outputs_norm, teacher_language_outputs_norm)
    teacher_outputs_for_align = torch.einsum('ni,ci->nc', teacher_vision_outputs_norm, teacher_language_outputs_norm)

    # Divide by distillation temperature
    student_outputs_for_align = student_outputs_for_align / args.temperature
    teacher_outputs_for_align = teacher_outputs_for_align / args.temperature

    # Take top k teacher outputs and corresponding student outputs for alignment
    teacher_for_align_topk_values, teacher_for_align_topk_ids = teacher_outputs_for_align.topk(
        k=min(args.clip_align_proximal_text_num, teacher_outputs_for_align.shape[-1]), dim=-1)  # [N, K]
    student_for_align_topk_values = student_outputs_for_align.gather(-1, teacher_for_align_topk_ids)  # [N, K]

    soft_teacher_values = F.softmax(teacher_for_align_topk_values, dim=-1)
    log_soft_teacher_values = F.log_softmax(teacher_for_align_topk_values, dim=-1)
    log_soft_student_values = F.log_softmax(student_for_align_topk_values, dim=-1)

    alignment_loss = 1.0 * (soft_teacher_values * log_soft_teacher_values - log_soft_student_values).sum(dim=-1).mean()

    return alignment_loss

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


def loss_fn_kd(outputs, teacher_outputs, T):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1)) * (T * T)

    return KD_loss


def distill(args):
    device = get_device(args)
    teacher, preprocess = get_teacher(args)
    student = get_student(args)
    student.fc = nn.Linear(student.fc.in_features, teacher.visual.output_dim)
    student = student.to(device)
    # student.load_state_dict(torch.load(f'{args.load_path}/resnet_distilled_from_clip_on_StanfordCars_best.pt'))

    benchmark = get_resnet(args)

    train_set, validation_set, test_set, classes = get_data(args)
    benchmark.fc = nn.Linear(benchmark.fc.in_features, len(classes))
    benchmark = benchmark.to(device)

    train_set_original = ImageTitleDatasetWrapper(train_set, classes, preprocess)
    validation_set_original = ImageTitleDatasetWrapper(validation_set, classes, preprocess)
    validation_set_ood = ImageTitleDatasetWrapper(validation_set, classes, preprocess, ood=True)
    train_loader = DataLoader(train_set_original, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set_original, batch_size=512, shuffle=False)
    validation_loader_ood = DataLoader(validation_set_ood, batch_size=512, shuffle=False)

    accuracy_list_on_original, accuracy_list_on_ood = [], []
    benchmark_accuracy_list_on_original, benchmark_accuracy_list_on_ood = [], []

    # criterion = nn.CosineEmbeddingLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    benchmark_optimizer = torch.optim.Adam(benchmark.parameters(), lr=args.lr)
    benchmark_criterion = nn.CrossEntropyLoss()

    num_epochs = args.num_epochs
    T = args.temperature
    soft_target_loss_weight = args.distillation_loss_weight
    alignment_loss_weight = args.alignment_loss_weight
    ce_loss_weight = 1.0 - soft_target_loss_weight - alignment_loss_weight
    teacher.eval()

    for epoch in range(num_epochs):
        student.train()  # Set student model to training mode
        benchmark.train()
        running_loss = 0.0
        logging.info(f'Epoch {epoch}/{num_epochs}')
        pbar = tqdm(train_loader)
        for images, labels, titles in pbar:
            images = images.to(device).float()
            labels = labels.to(device)
            # titles = titles.to(device)
            batch_size = labels.shape[0]
            optimizer.zero_grad()
            benchmark_optimizer.zero_grad()
            # Get teacher model (CLIP) output
            with torch.no_grad():
                teacher_logits = teacher.encode_image(images).float()
                # teacher_text_logits = teacher.encode_text(titles.squeeze()).float()

            # Forward pass with the student model
            student_logits = student(images)

            # alignment_loss = get_vision_language_alignment_loss(args, teacher_logits, teacher_text_logits,
            #                                                     student_logits)
            #
            # # Soften the student logits by applying softmax first and log() second
            # soft_targets = F.softmax(teacher_logits / T, dim=-1)
            # soft_prob = F.log_softmax(student_logits / T, dim=-1)
            #
            # # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling
            # # the knowledge in a neural network"
            # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / (
            #             soft_prob.size()[0] * (T ** 2))

            soft_targets_loss = loss_fn_kd(student_logits, teacher_logits, T)

            # Calculate the true label loss
            label_loss = criterion(student_logits, labels)

            # Weighted sum of the two losses
            loss = (
                    soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss
                # + alignment_loss_weight * alignment_loss
            )

            benchmark_outputs = benchmark(images)
            benchmark_loss = benchmark_criterion(benchmark_outputs, labels)

            # backprop
            loss.backward()
            optimizer.step()

            benchmark_loss.backward()
            benchmark_optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix({"running_loss": running_loss / float(batch_size),
                              "soft_targets_loss": soft_targets_loss.item() / float(batch_size),
                              'label_loss': label_loss.item() / float(batch_size)
                              #, 'alignment_loss': alignment_loss.item() / float(batch_size)}
                              })
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

            save_model(args, student,
                       f'aligned_distilled_val_acc_standard_{val_acc_standard:.3f}_val_acc_ood_{val_acc_ood:.3f}')
            save_model(args, benchmark,
                       f'benchmark_val_acc_standard_{benchmark_val_acc_standard:.3f}_val_acc_ood_{benchmark_val_acc_ood:.3f}')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training completed.")


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
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default='sgd',
                        choices=['adam', 'sgd'], help="optimizer type")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--seed", type=int, default=42, help="seed value")
    parser.add_argument("--save-path", type=str, default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--load-path", type=str, default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--resnet-ver", type=str, default='resnet18', choices=['resnet18', 'resnet50'],
                        help="resnet18 or resnet50")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")
    parser.add_argument("--eval-every", type=int, default=10)

    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation loss temperature")
    parser.add_argument("--distillation-loss-weight", type=float, default=0.5, help="Distillation loss weight")
    parser.add_argument("--alignment-loss-weight", type=float, default=0.0, help="alignment loss weight")

    parser.add_argument('--clip-align-proximal-text-num', type=int, default=32,
                        help="If >0, specifies the k in L-vlalign")

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

    distill(args)
