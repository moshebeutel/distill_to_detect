import argparse
import logging
import time
from functools import partial
from pathlib import Path
import clip
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

from utils import set_logger, set_seed, get_device, get_data, ImageTitleDatasetWrapper


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


def save_model(args, model, suff):
    save_path = Path(args.save_path)
    if not save_path.exists():
        save_path.mkdir()
    file_path = Path(args.save_path) / f'clip_finetuned_to_{args.data_name}_{time.asctime()}_{suff}.pt'
    torch.save(model.state_dict(), file_path.as_posix())


def get_model(args):
    logging.info(f'Loading model {args.model}')
    model, preprocess = clip.load(args.model, device=get_device(args), jit=False)
    if args.load_path:
        model.load_state_dict(torch.load(f'{args.load_path}/best_model.pt'))

    num_of_params = sum([torch.numel(l) for l in model.parameters()])
    logging.info(f'Loaded model num_of_params {num_of_params}')
    return model, preprocess


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
    parser.add_argument("--load-path", type=str, default=(Path.home() / f'saved_models' / 'clip').as_posix(),
                        help="dir path for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="seed value")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")

    parser.add_argument("--finetune-on-ood", type=bool, default=False, help="preform fine tune on ood data")
    parser.add_argument("--finetune-epochs", type=int, default=10, help="num epochs of fine tune")
    parser.add_argument("--finetune-ood-epochs", type=int, default=3, help="num epochs of fine tune")

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

    finetune(args)
