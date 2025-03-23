# inference.py
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

from model import MyBirdDroneCNN


def predict_image(model, img_path, device):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        prob = output.item()
        label = 1 if prob >= 0.5 else 0
    return label, prob


def main():
    if len(sys.argv) < 2:
        print("Bitte Pfad zum Bild angeben. Beispiel: python inference.py my_image.jpg")
        return
    img_path = sys.argv[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyBirdDroneCNN()
    model.load_state_dict(torch.load("my_cnn_bird_drone.pth", map_location=device))
    model.to(device)

    label, prob = predict_image(model, img_path, device)
    class_names = ["bird", "drone"]
    pred_class = class_names[label]
    print(f"Bild: {img_path}")
    print(f"Vorhergesagte Klasse: {pred_class} (Wahrscheinlichkeit = {prob:.3f})")


if __name__ == "__main__":
    main()
