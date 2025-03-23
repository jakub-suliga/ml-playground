# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import BirdDroneCNN
from dataset import create_dataset
import kagglehub


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            predicted = (outputs >= 0.5).long()
            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    path = kagglehub.dataset_download("muhammadsaoodsarwar/drone-vs-bird") + "/dataset"
    create_dataset(path + "/bird", "bird", train_size=0.7, val_size=0.15)
    create_dataset(path + "/drone", "drone", train_size=0.7, val_size=0.15)
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root="data/train", transform=train_transforms
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root="data/val", transform=val_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BirdDroneCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    best_val_acc = 0.0
    save_path = "cnn_bird_drone.pth"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predicted = (outputs >= 0.5).long()
            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
