from torch import nn
from torchvision import models

def get_model(num_classes: int):
    """
    Load pretrained ResNet50 model with final FC layer modified for Animals-10 dataset
    
    Args:
        num_classes (int): Number of output classes.
    
    Returns:
        model (torch.nn.Module): Modified ResNet50 model.
    """
    # Load ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer to match the number of classes in Animals-10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

