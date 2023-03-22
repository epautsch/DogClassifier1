import os, re
import torch
from typing import Tuple

from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments, pipeline, \
    AutoModelForImageClassification, AutoFeatureExtractor, DefaultFlowCallback
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def find_latest_checkpoint(output_dir: str) -> str:
    checkpoint_files = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    checkpoint_numbers = [int(re.findall(r'\d+', chkpt)[0]) for chkpt in checkpoint_files if re.findall(r'\d+', chkpt)]
    if checkpoint_numbers:
        latest_checkpoint = max(checkpoint_numbers)
        return os.path.join(output_dir, f'checkpoint-{latest_checkpoint}')
    else:
        return None


def get_transform(model_name: str) -> transforms.Compose:
    if model_name == 'microsoft/resnet-50' or model_name == 'microsoft/resnet-152' or model_name == 'google/vit-base-patch16-224':
        size = (224, 224)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform


def compute_metrics(eval_preds: Tuple[torch.Tensor, torch.Tensor]) -> dict:
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {'accuracy': accuracy}


def fine_tune_model(pretrained_model_name: str,
                    train_dataloader: DataLoader,
                    num_epochs: int,
                    learning_rate: float,
                    weight_decay: float,
                    callbacks=None) -> Tuple[torch.nn.Module, dict]:

    model = ViTForImageClassification.from_pretrained(pretrained_model_name,
                                                      num_labels=len(full_dataset.classes),
                                                      ignore_mismatched_sizes=True).to(device)
    model.classifier = torch.nn.Linear(model.config.hidden_size, len(full_dataset.classes)).to(device)

    training_args = TrainingArguments(
        output_dir='./' + pretrained_model_name.replace('/', '_') + '_output',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir='./logs',
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=training_args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=training_args.num_train_epochs * len(train_dataloader))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {'pixel_values': torch.stack([x[0] for x in data]),
                                    'labels': torch.tensor([x[1] for x in data])},
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    trainer.save_model('./' + pretrained_model_name.replace('/', '_') + '_output')

    eval_results = trainer.evaluate(eval_dataset=test_dataset)

    return model, eval_results


def evaluate_saved_model(model_path: str,
                         feature_extractor_name: str,
                         test_dataloader: DataLoader) -> Tuple[float, pipeline]:

    model = AutoModelForImageClassification.from_pretrained(model_path).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)

    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in test_dataloader:
            pixel_values = batch[0].to(device)
            labels = batch[1].to(device)

            outputs = model(pixel_values)
            _, predicted_labels = torch.max(outputs.logits, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted_labels == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")

    classify = pipeline('image-classification', model=model, feature_extractor=feature_extractor,
                        device=0 if torch.cuda.is_available() else -1)

    return accuracy, classify


class LoggingCallback(DefaultFlowCallback):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.learning_rate = []

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None:
            self.losses.append(logs.get('loss'))
            self.learning_rate.append(logs.get('learning_rate'))


def plot_metrics(losses: list[float], learning_rate: list[float], pretrained_model_name: str) -> None:
    sns.set(style='whitegrid')

    plt.figure()
    sns.lineplot(x=range(len(losses)), y=losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss for ' + pretrained_model_name)
    plt.savefig(pretrained_model_name.replace('/', '_') + '_loss.png')
    plt.show()
    plt.close()

    plt.figure()
    sns.lineplot(x=range(len(learning_rate)), y=learning_rate)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate for ' + pretrained_model_name)
    plt.savefig(pretrained_model_name.replace('/', '_') + '_learningRate.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    pretrained_model_name = 'google/vit-base-patch16-224'

    logging_callback = LoggingCallback()

    transform = get_transform(pretrained_model_name)

    train_dir = "./images"
    full_dataset = ImageFolder(train_dir, transform=transform)

    train_ratio = 0.8
    train_size = int(len(full_dataset) * train_ratio)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # model, eval_results = fine_tune_model(pretrained_model_name,
    #                                       train_dataloader,
    #                                       num_epochs=3,
    #                                       learning_rate=1e-4,
    #                                       weight_decay=1e-2,
    #                                       callbacks=[logging_callback])
    #
    # print("Evaluation Results:", eval_results)
    #
    # plot_metrics(logging_callback.losses, logging_callback.learning_rate, pretrained_model_name)
    #
    # print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    #
    model_path = './' + pretrained_model_name.replace('/', '_') + '_output'
    feature_extractor_name = pretrained_model_name
    accuracy, classify = evaluate_saved_model(model_path, feature_extractor_name, test_dataloader)
