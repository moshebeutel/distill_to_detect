import argparse
import csv
import gc
import logging
from pathlib import Path


import torch
import torchvision.ops
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from utils import set_logger, set_seed, get_device, get_resnet, get_data, save_model

IMAGE_SIZE = 224

class Backbone(nn.Module):
    def __init__(self, resnet: nn.Module):
        super(Backbone, self).__init__()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last fully connected layer

    def forward(self, x):
        return self.backbone(x)


class DetectionHead(nn.Module):
    def __init__(self, num_features=2048, num_classes=196):
        super(DetectionHead, self).__init__()
        self.conv = nn.Conv2d(num_features, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cls_head = nn.Conv2d(512, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(512, 4, kernel_size=1)  # 4 coordinates for bounding box

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        cls_logits = self.cls_head(x)
        bbox_preds = F.relu(self.reg_head(x))
        return cls_logits, bbox_preds


class DetectionModel(nn.Module):
    def __init__(self, backbone, detection_head, freeze_backbone=True):
        super(DetectionModel, self).__init__()
        self.backbone = backbone
        self.detection_head = detection_head
        self._freeze_backbone = freeze_backbone

    def forward(self, x):
        if self._freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        cls_logits, bbox_preds = self.detection_head(features)
        return cls_logits, bbox_preds


class ImageTitleDetectionDatasetWrapper(Dataset):
    def __init__(self, annotations_file, img_dir, ood=False):
        self._img_dir = img_dir

        self._transform = v2.Compose([
            v2.Resize(IMAGE_SIZE),  # Resize the shorter side to 224 pixels
            v2.CenterCrop(IMAGE_SIZE),  # Crop the center to 224x224
            v2.ToTensor(),  # Convert the image to a PyTorch tensor
            v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize the image
                         std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self._ood_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.ColorJitter(contrast=0.5, brightness=1.0),
                                                  transforms.ToPILImage()])

        with open(annotations_file, 'r') as f:
            reader = csv.reader(f)
            self._annotations = [row for row in reader]

        with open(f'{args.data_path}/names.csv', 'r') as f:
            reader = csv.reader(f)
            self._classes = [row[0] for row in reader]

        self._ood = ood

    def __len__(self):
        return len(self._annotations)

    def __getitem__(self, idx):
        annotation = self._annotations[idx]

        bbox = annotation[1:5]  # x1, y1, x2, y2
        bbox = [float(b) for b in bbox]

        label = int(annotation[5]) - 1  # Convert 1-based to 0-based index
        img_name = annotation[0]
        img_cls = self._classes[label]
        img_path = Path(self._img_dir) / img_cls / img_name

        image = Image.open(img_path.as_posix()).convert('RGB')

        if self._ood:
            image = self._ood_transform(image)

        if self._transform:
            # Detection (re-using imports and transforms from above)
            from torchvision import tv_tensors

            boxes = tv_tensors.BoundingBoxes([bbox], format="XYXY", canvas_size=(IMAGE_SIZE, IMAGE_SIZE))
            image, bbox = self._transform(image, boxes)
            bbox = bbox / float(IMAGE_SIZE)


        return image, torch.tensor(label), bbox

@torch.no_grad()
def evaluate(loader: DataLoader, model: torch.nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_correct = 0
    total_images = 0
    sum_iou = 0.0

    with torch.no_grad():
        for batch in loader:
            image, label, bbox = batch
            images = image.to(device)
            true_bboxes = bbox.to(device)
            true_labels = label.to(device)

            cls_logits, bbox_preds = model(images)
            _, predicted_labels = torch.max(cls_logits, 1)

            total_correct += (predicted_labels.squeeze() == true_labels).sum().item()
            total_images += true_labels.shape[0]
            true_bboxes = true_bboxes.reshape(true_bboxes.shape[0], 4, -1)
            bbox_preds = bbox_preds.reshape(true_bboxes.shape)
            sum_iou += float(torchvision.ops.box_iou(bbox_preds, true_bboxes).squeeze().trace())

    accuracy = 100.0 * total_correct / float(total_images)
    avg_iou: float = 100.0 * sum_iou / float(total_images)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Evaluation - Accuracy: {accuracy:.4f}, Avg IoU: {avg_iou:.4f} total images: {total_images}")
    return accuracy, avg_iou


def detect(args):
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

    distilled_detector = DetectionModel(backbone=distilled_backbone, detection_head=DetectionHead())
    benchmark_detector = DetectionModel(backbone=benchmark_backbone, detection_head=DetectionHead())

    distilled_detector.to(device)
    benchmark_detector.to(device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()
    distilled_optimizer = torch.optim.Adam(distilled_detector.parameters(), lr=args.lr)
    benchmark_optimizer = torch.optim.Adam(benchmark_detector.parameters(), lr=args.lr)

    dataset_path = Path(args.data_path)
    assert dataset_path.exists(), f'{dataset_path} does not exist'

    train_set_path = dataset_path / 'train'
    assert train_set_path.exists(), f'{train_set_path} does not exist'

    test_set_path = dataset_path / 'test'
    assert test_set_path.exists(), f'{test_set_path} does not exist'

    annotations_train_file_path = dataset_path / 'anno_train.csv'
    assert annotations_train_file_path.exists(), f'{annotations_train_file_path} does not exist'

    annotations_test_file_path = dataset_path / 'anno_test.csv'
    assert annotations_test_file_path.exists(), f'{annotations_test_file_path} does not exist'

    train_set_original = ImageTitleDetectionDatasetWrapper(img_dir=train_set_path.as_posix(),
                                                           annotations_file=annotations_train_file_path.as_posix(),
                                                           ood=False)
    # validation_set_original = ImageTitleDetectionDatasetWrapper(validation_set, classes, preprocess)
    # validation_set_ood = ImageTitleDetectionDatasetWrapper(validation_set, classes, preprocess, ood=True)
    validation_set_original = ImageTitleDetectionDatasetWrapper(img_dir=test_set_path.as_posix(),
                                                                annotations_file=annotations_test_file_path.as_posix(),
                                                                ood=False)
    validation_set_ood = ImageTitleDetectionDatasetWrapper(img_dir=test_set_path.as_posix(),
                                                           annotations_file=annotations_test_file_path.as_posix(),
                                                           ood=True)
    train_loader = DataLoader(train_set_original, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_set_original, batch_size=512, shuffle=False)
    validation_loader_ood = DataLoader(validation_set_ood, batch_size=512, shuffle=False)

    accuracy_list_on_original, accuracy_list_on_ood = [], []
    iou_list_on_original, iou_list_on_ood = [], []
    benchmark_accuracy_list_on_original, benchmark_accuracy_list_on_ood = [], []
    benchmark_iou_list_on_original, benchmark_iou_list_on_ood = [], []

    # val_acc_standard, val_iou_standard = evaluate(validation_loader, distilled_detector, device)

    for epoch in range(args.num_epochs):
        running_loss = 0.0
        benchmark_running_loss = 0.0
        logging.info(f'Epoch {epoch}/{args.num_epochs}')
        pbar = tqdm(train_loader)

        distilled_detector.train()
        benchmark_detector.train()

        for data in pbar:
            images, labels, bbox = data
            images, labels, bbox = images.to(device), labels.to(device), bbox.to(device)
            batch_size = labels.shape[0]

            distilled_optimizer.zero_grad()
            benchmark_optimizer.zero_grad()

            distilled_cls_logits, distilled_bbox_preds = distilled_detector(images)
            benchmark_cls_logits, benchmark_bbox_preds = benchmark_detector(images)

            distilled_cls_logits = distilled_cls_logits.reshape(batch_size, num_classes)
            benchmark_cls_logits = benchmark_cls_logits.reshape(batch_size, num_classes)
            labels = labels.reshape(batch_size)

            distilled_cls_loss = criterion_cls(distilled_cls_logits, labels)
            benchmark_cls_loss = criterion_cls(benchmark_cls_logits, labels)

            bbox = bbox.reshape(batch_size, 4, -1)

            distilled_bbox_pred_loss = criterion_bbox(distilled_bbox_preds.reshape(bbox.shape), bbox)
            benchmark_bbox_pred_loss = criterion_bbox(benchmark_bbox_preds.reshape(bbox.shape), bbox)

            distilled_loss = distilled_cls_loss + distilled_bbox_pred_loss
            benchmark_loss = benchmark_cls_loss + benchmark_bbox_pred_loss

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
            val_acc_standard, val_iou_standard = evaluate(validation_loader, distilled_detector, device)
            accuracy_list_on_original.append(val_acc_standard)
            iou_list_on_original.append(val_iou_standard)
            logging.info(f'*** Validation after {epoch + 1} epochs detector train ***')
            logging.info(f'             Distilled Validation accuracy {val_acc_standard}')
            logging.info(f'             Distilled Validation iou {val_iou_standard}')
            benchmark_val_acc_standard, benchmark_val_iou_standard = evaluate(validation_loader, benchmark_detector,
                                                                              device)
            benchmark_accuracy_list_on_original.append(benchmark_val_acc_standard)
            benchmark_iou_list_on_original.append(benchmark_val_iou_standard)
            logging.info(f'             Benchmark Validation accuracy {benchmark_val_acc_standard}')
            logging.info(f'             Benchmark Validation iou {benchmark_val_iou_standard}')
            val_acc_ood, val_iou_ood = evaluate(validation_loader_ood, distilled_detector, device)
            accuracy_list_on_ood.append(val_acc_ood)
            iou_list_on_ood.append(val_iou_ood)
            logging.info(f'             Distilled Validation accuracy OOD {val_acc_ood}')
            logging.info(f'             Distilled Validation iou OOD {val_iou_ood}')
            benchmark_val_acc_ood, benchmark_val_iou_ood = evaluate(validation_loader_ood, benchmark_detector, device)
            benchmark_accuracy_list_on_ood.append(benchmark_val_acc_ood)
            benchmark_iou_list_on_ood.append(benchmark_val_iou_ood)
            logging.info(f'             Benchmark Validation accuracy OOD {benchmark_val_acc_ood}')
            logging.info(f'             Benchmark Validation iou OOD {benchmark_val_iou_ood}')

            save_model(args, distilled_detector,
                       f'distilled_classifier_val_acc_standard_{val_acc_standard:.3f}_val_acc_ood_{val_acc_ood:.3f}')
            save_model(args, benchmark_detector,
                       f'benchmark_classifier_val_acc_standard_{benchmark_val_acc_standard:.3f}_val_acc_ood_{benchmark_val_acc_ood:.3f}')

        logging.info(f"Epoch [{epoch + 1}/{args.num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    logging.info("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detection using distilled backbone")
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
    parser.add_argument("--distilled-out-dim", type=int, default=512, help='Distilled model output dimension')

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
    parser.add_argument("--resnet-ver", type=str, default='resnet50', choices=['resnet18', 'resnet50'],
                        help="resnet18 or resnet50")

    parser.add_argument("--use-cuda", type=bool, default=True, help="use cuda or cpu")
    parser.add_argument("--eval-every", type=int, default=1)

    args = parser.parse_args()

    set_logger()
    set_seed(args.seed)

    detect(args)
