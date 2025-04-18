# evaluate.py
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dein Modell
from model import BirdDroneCNN


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

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

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    return avg_loss, acc, all_labels, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        root="data/test", transform=test_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BirdDroneCNN()
    model_path = "my_cnn_bird_drone.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    criterion = nn.BCELoss()
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    cm = confusion_matrix(y_true, y_pred)
    labels = test_dataset.classes
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Confusion Matrix (Test Data)")
    plt.show()


if __name__ == "__main__":
    main()
