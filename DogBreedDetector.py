import random
from typing import Tuple
import os
import xml.etree.ElementTree as ET

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_detection_model(num_classes: int) -> torchvision.models.detection:
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transforms() -> torchvision.transforms.Compose:
    return torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def parse_annotations_file(file_path: str) -> Tuple[str, str, Tuple[int, int, int, int]]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    folder = root.find('folder').text
    filename = root.find('filename').text

    if '%s' in folder or '%s' in filename:
        breed_folder, file_number = os.path.splitext(os.path.basename(file_path))[0].split('_')
        folder = folder.replace('%s', breed_folder)
        filename = filename.replace('%s', breed_folder + '_' + file_number)


    breed = root.find('object').find('name').text
    bndbox = root.find('object').find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    bbox = (xmin, ymin, xmax, ymax)

    return filename, breed, bbox


def load_annotations(root_folder: str, annotations_folder: str) -> Tuple[list, list]:
    annotations = []

    breed_folders = sorted(os.listdir(annotations_folder))
    breeds = []

    for breed_folder in breed_folders:
        breed_path = os.path.join(annotations_folder, breed_folder)
        for annotation_file in os.listdir(breed_path):
            file_path = os.path.join(breed_path, annotation_file)
            filename, breed, bbox = parse_annotations_file(file_path)
            img_path = os.path.join(breed_folder, filename + '.jpg')
            img_path = img_path.replace('\\', os.path.sep)
            annotations.append((img_path, breed, bbox))

            if breed not in breeds:
                breeds.append(breed)

    breeds = sorted(breeds)
    print('Breeds found:', breeds)

    return annotations, breeds


def split_data(annotations, test_size=0.2, random_state=42) -> Tuple[list, list]:
    train_annotations, test_annotations = train_test_split(annotations, test_size=test_size, random_state=random_state,
                                                           stratify=[a[1] for a in annotations])
    return train_annotations, test_annotations


def collate_fn(batch: list) -> Tuple[list, list]:
    images, targets = zip(*batch)
    images = list(images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    return images, targets


class StanfordDogsDataset(Dataset):
    def __init__(self, root, annotations, dog_breeds, transforms=None):
        self.root = root
        self.annotations = annotations
        self.transforms = transforms
        self.classes = dog_breeds
        print(self.classes)
        print(len(self.classes))

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict]:
        img_path, breed, bbox = self.annotations[idx]
        img = cv2.imread(os.path.join(self.root, img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = {
            'boxes': torch.as_tensor([bbox], dtype=torch.float32),
            'labels': torch.as_tensor([self.classes.index(breed)], dtype=torch.int64),
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target


def train_one_epoch(model, data_loader, optimizer, device, gradient_accumulation_steps=4) -> float:
    model.train()
    total_loss = 0
    num_batches = 0
    scaler = GradScaler()
    optimizer.zero_grad()

    for step, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        try:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            num_batches += 1

            scaled_losses = scaler.scale(losses)
            scaled_losses /= gradient_accumulation_steps
            scaled_losses.backward()
        except Exception as e:
            print(f"Error in step {step}: {e}")
            continue

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

    return total_loss / num_batches


def iou(box1, box2) -> float:
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xi_min = max(xmin1, xmin2)
    yi_min = max(ymin1, ymin2)
    xi_max = min(xmax1, xmax2)
    yi_max = min(ymax1, ymax2)

    inter_area = max(xi_max - xi_min, 0) * max(yi_max - yi_min, 0)

    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def evaluate(model, data_loader, device, iou_threshold=0.5) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            for idx, prediction in enumerate(predictions):
                gt_boxes = targets[idx]['boxes'].tolist()
                gt_labels = targets[idx]['labels'].tolist()
                pred_boxes = prediction['boxes'].tolist()
                pred_labels = prediction['labels'].tolist()

                for gt_idx, gt_box in enumerate(gt_boxes):
                    gt_label = gt_labels[gt_idx]
                    total += 1

                    for pred_idx, pred_box in enumerate(pred_boxes):
                        pred_label = pred_labels[pred_idx]

                        if gt_label == pred_label:
                            if iou(gt_box, pred_box) >= iou_threshold:
                                correct += 1
                                break

    accuracy = correct / total
    return accuracy


def plot_metric(metric_values, title, xlabel, ylabel, save_name) -> None:
    plt.figure()
    sns.lineplot(x=np.arange(1, len(metric_values) + 1), y=metric_values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(metric_values) + 1))

    plt.savefig(save_name)

    plt.show()


def visualize_predictions(model, dataset, device, num_images=5, save_dir='predictions'):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_images)

    for i, idx in enumerate(indices):
        img, target = dataset[idx]
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(img_tensor)[0]

            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = (img_np * 255).astype(np.uint8)

            # predicted boxes
            for box, label in zip(prediction['boxes'], prediction['labels']):
                box = box.to(torch.int64).tolist()
                label = dog_breeds[label.item() - 1]
                cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(img_np, f'Pred: {label}', (box[0], box[1] - 10), cv2.FRONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # ground truth boxes
            for box, label in zip(target['boxes'], target['labels']):
                box = box.to(torch.int64).tolist()
                label = dog_breeds[label.item() - 1]
                cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_np, f'GT: {label}', (box[0], box[1] - 10), cv2.FRONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(save_dir, f'prediction_{i}.jpg'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print(f'Saved prediction_{i}.jpg')


def run(train=True, num_epochs=10, model_save_path='./dog_breed_detection_model.pth'):
    annotations_folder = './Annotations'
    image_folder = './Images'
    annotations, dog_breeds = load_annotations(image_folder, annotations_folder)
    train_annotations, test_annotations = split_data(annotations)

    num_classes = len(dog_breeds) + 1
    model = get_detection_model(num_classes)

    train_root = image_folder
    val_root = image_folder

    train_dataset = StanfordDogsDataset(train_root, train_annotations, dog_breeds, get_transforms())
    val_dataset = StanfordDogsDataset(val_root, test_annotations, dog_breeds, get_transforms())

    # batch_size=8 too big
    train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    if train:
        best_accuracy = 0.0
        train_losses = []
        learning_rates = []

        for epoch in range(num_epochs):
            epoch_loss = train_one_epoch(model, train_data_loader, optimizer, device)
            train_losses.append(epoch_loss)
            learning_rates.append(optimizer.param_groups[0]['lr'])

            accuracy = evaluate(model, val_data_loader, device)
            print(f'Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), model_save_path)
                print(f'Model saved at epoch {epoch + 1} with accuracy {best_accuracy:.4f}')

        plot_metric(train_losses, 'Training Loss per Epoch', 'Epoch', 'Loss', 'dog_breed_detection_train_loss.png')
        plot_metric(learning_rates, 'Learning Rate per Epoch', 'Epoch', 'Learning Rate',
                    'dog_breed_detection_learning_rate.png')
    else:
        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
        visualize_predictions(model, val_dataset, device)


if __name__ == '__main__':
    # run(train=True, num_epochs=10)

    run(train=False)