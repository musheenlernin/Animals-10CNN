import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_data(data_dir: str, data_transforms: dict, batch_size: int):
    """ 
    Create PyTorch DataLoaders for training. validation, and test datasets.
    
    Args:
        data_dir (str): Path to the dataset.
        data_transforms (dict): Transformations for train and validation datasets.
        batch_size (int): Number of samples per batch.
    
    Returns:
        dataloaders (dict): DataLoaders for training, validation, and test datasets.
        dataset_sizes (dict): Sizes of the training, validation, and test datasets.
        class_names (list): List of class names.
    """
    # Load dataset
    datasets = ImageFolder(root=data_dir) # TODO Labels are in Italian and should be translated
    # Class names
    class_names = datasets.classes

    # Define sizes for validation and test sets
    train_size = int(0.7 * len(datasets))  # 70% for training
    val_size = int(0.15 * len(datasets))   # 15% for validation
    test_size = len(datasets) - train_size - val_size  # Remaining 15% for testing


    # Split the val_test dataset into validation and test sets
    train_set, val_set, test_set = random_split(datasets, [train_size, val_size, test_size])


    # Apply custom transforms

    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            image, label = self.subset[index]
            image = self.transform(image)
            return image, label

        def __len__(self):
            return len(self.subset)

    datasets = {
        'train' : TransformedSubset(train_set, data_transforms['train']),
        'val' : TransformedSubset(val_set, data_transforms['val_test']),
        'test' : TransformedSubset(test_set, data_transforms['val_test'])
    }


    # Create DataLoaders
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=batch_size, shuffle=(x == 'train')) for x in ['train', 'val', 'test']
                } # NOTE: Consider setting num_workers (maybe write a script that sets num_workers device independently?)


    # dataset sizes
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

    return dataloaders, dataset_sizes, class_names


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
    """
    Train the model and return the model with the best validation accuracy.
    
    Args:
        model: The model to train.
        dataloaders: DataLoader for train and validation datasets.
        dataset_sizes: Sizes of the train and validation datasets.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Training device ('cuda' or 'cpu').
        num_epochs (int): Number of training epochs.
    
    Returns:
        model: The trained model with the best validation accuracy.
    """
    # Weights giving highest val_acc
    best_model_wts = copy.deepcopy(model.state_dict())
    # Best val_acc
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{num_epochs}')

        # Start timing the epoch
        start_time = time.time()

        # Train and evaluate
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                print('.', end="") # For timing

                # Zero the parameter gradients
                if phase == 'train': optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            print()


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if the accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # End timing the epoch
        epoch_time = time.time() - start_time
        print(f'Epoch completed in {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, dataloader, device, class_names):
    """
    Evaluate the trained model on the validation set.
    
    Args:
        model: The trained model.
        dataloader: DataLoader for the validation dataset.
        device: Evaluation device ('cuda' or 'cpu').
    
    Returns:
        None
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())


    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    plot_confusion_matrix(all_preds, all_labels, class_names)

    accuracy = 100 * correct / total
    print(f'Accuracy on validation set: {accuracy:.2f}%')



def plot_confusion_matrix(labels, preds, class_names):
    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()