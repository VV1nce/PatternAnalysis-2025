import torch
from dataset import test_loader
from modules import convnext_small


DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


CHECKPOINT_PATH = 'checkpoint_best.pth'

def test_model():

    num_classes = len(test_loader.dataset.classes)


    model = convnext_small(num_classes=num_classes, in_chans=1).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = outputs.max(1)

            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_model()