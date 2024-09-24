import torch
import torch.nn as nn
from torchvision import transforms
from model import get_pretrained_model
from utils import train_model, evaluate_model, load_data


# Data transformations
# Training set resized and normalized according to ResNet50 requirements
# Described in "Deep Residual Learning for Image Recognition" https://arxiv.org/abs/1512.03385 (since it was trained on ImageNet)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load dataset
data_dir = './data/raw-img/'
dataloaders, dataset_sizes, class_names = load_data(data_dir, data_transforms, batch_size=32)


# Load model, optimizer, and loss function

model = get_pretrained_model(num_classes=len(class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.01)


# Train the model
trained_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10)

# Save the best model
torch.save(trained_model.state_dict(), 'best_model.pth')

# Evaluate the model on the validation set
evaluate_model(trained_model, dataloaders['val'], device)