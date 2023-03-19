import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = "./images"
full_dataset = ImageFolder(train_dir, transform=transform)

train_ratio = 0.8
train_size = int(len(full_dataset) * train_ratio)
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224',
                                                  num_labels=len(full_dataset.classes),
                                                  ignore_mismatched_sizes=True).to(device)
model.classifier = torch.nn.Linear(model.config.hidden_size, len(full_dataset.classes)).to(device)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

training_args = TrainingArguments(
    output_dir='./output',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=lambda data: {'pixel_values': torch.stack([x[0] for x in data]),
                                'labels': torch.tensor([x[1] for x in data])},
)

trainer.train()

trainer.save_model('./output/fine_tuned_dogs')

trainer.evaluate(eval_dataset=test_dataset)

# model = ViTForImageClassification.from_pretrained('./output/fine_tuned_dogs').to(device)
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
#
# image_path = 'path/to/image'
# image = Image.open(image_path)
#
# inputs = feature_extractor(images=image, return_tensors='pt')
# inputs = {k: v.to(device) for k, v in inputs.items()}
#
# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits
#     predicted_class_index = torch.argmax(logits, dim=1).item()
#     predicted_class_name = train_dataset.dataset.classes[predicted_class_index]
#
# print('Predicted class index:', predicted_class_index)
# print('Predicted class name:', predicted_class_name)
