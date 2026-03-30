import torch
import torchvision.transforms as transforms
from PIL import Image


import torchvision.models as models

def get_image_transforms():
    """Returns the standard ImageNet preprocessing transforms."""
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_efficientnet_model(device):
    """Loads a pretrained EfficientNetV2-S model for feature extraction."""
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier = torch.nn.Identity() # Remove the final 1000-class linear layer
    
    for param in model.parameters():
        param.requires_grad = False
        
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image_path):
    """Loads and preprocesses a single image for VGG19."""
    img = Image.open(image_path).convert("RGB")
    transform = get_image_transforms()
    return transform(img).unsqueeze(0)
