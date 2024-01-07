# train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

# Define your neural network architecture
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to calculate accuracy
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(preds)

def main():
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('data_directory', help="This is the dir of the training images e.g. if a sample file is in /flowers/train/daisy/001.png then supply /flowers. Expect 2 folders within, 'train' & 'valid'")
    # optional arguments
    parser.add_argument('--save_directory', dest='save_directory', help="This is the dir where the model will be saved after training.", default='checkpoint.pth')
    parser.add_argument('--learning_rate', dest='learning_rate', help="This is the learning rate when training the model. Default is 0.003. Expect float type", default=0.003, type=float)
    parser.add_argument('--epochs', dest='epochs', help="This is the number of epochs when training the model. Default is 5. Expect int type", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to train the model on the GPU via CUDA", action='store_true')
    parser.add_argument('--model_arch', dest='model_arch', help="This is type of pre-trained model that will be used", default="resnet18", type=str, choices=['resnet18', 'resnet34', 'resnet50'])

    args = parser.parse_args()

    # Set device (CPU/GPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Define transformations and dataset
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_directory, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(args.data_directory, 'valid'), transform=val_test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Initialize the model, loss function, and optimizer
    model = FlowerClassifier(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_corrects = 0
        total_val_samples = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()

                total_val_corrects += accuracy(outputs, labels) * len(labels)
                total_val_samples += len(labels)

        average_val_loss = total_val_loss / len(val_loader)
        val_accuracy = total_val_corrects / total_val_samples

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {average_train_loss:.4f}, '
              f'Validation Loss: {average_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    # Save the trained model as a checkpoint
    checkpoint_path = os.path.join(args.save_directory)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': average_train_loss,
        'val_loss': average_val_loss,
        'val_accuracy': val_accuracy
    }, checkpoint_path)

    print(f'Model saved as {checkpoint_path}')

if __name__ == "__main__":
    main()
    