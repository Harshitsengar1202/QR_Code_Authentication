######### LOAD DATA #########
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define transformations with data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Random rotation & shift
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Vary brightness/contrast
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to (-1,1)
])

# Load dataset
data_dir = "G:\\vikas laptop backup\\lucky\\Alemeno Assignment\\"  # Change this to your actual dataset path
train_data = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Class names
class_names = train_data.classes
print("Classes:", class_names)
