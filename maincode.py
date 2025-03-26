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
data_dir = "Dataset"  # Change this to your actual dataset path
train_data = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)

# Class names
class_names = train_data.classes
print("Classes:", class_names)


###### LOAD PRE-TRAINED MODEL AND PERFROM BINARY CLASSIFICATION #######################
import torch.nn as nn
import torchvision.models as models

# Load ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the last layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

############## Loss, Optimizer & Learning Rate Scheduler ################
import torch.optim as optim

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with weight decay (prevents overfitting)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler (reduces LR if validation loss plateaus)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

###################### Train the Model with Early Stopping ##################
import numpy as np

num_epochs = 10
best_val_loss = np.inf  # Track best validation loss for early stopping

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    lr_scheduler.step(avg_loss)  # Adjust learning rate if needed
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Early Stopping Condition
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pth")  # Save best model

print("Training complete. Best model saved.")

################### Evaluate the Model ################################
from sklearn.metrics import accuracy_score, classification_report

# Load best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print accuracy and classification report
print("Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=class_names))

