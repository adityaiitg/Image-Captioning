import torch
import torchvision.transforms as transforms
from PIL import Image

def get_image_transforms():
    """Returns the standard ImageNet preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image_path):
    """Loads and preprocesses an image for VGG19"""
    img = Image.open(image_path).convert('RGB')
    transform = get_image_transforms()
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img
