import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

from dataset import test_loader, get_data_info   # Use existing test_loader
from modules import convnext_small  # Keep the same model as training


CKPT_PATH = 'checkpoint_best.pth'     # Path to saved weights, if EMA used, trying 'checkpoint_best_ema.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM = 3  # Number of random samples to predict and visualize


@torch.no_grad()
def predict(model, loader, device):
    """Predict on the entire test set."""
    model.eval()
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc='Predicting')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot and save confusion matrix using matplotlib only."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.matshow(cm, cmap=plt.cm.Blues, fignum=1)
    plt.colorbar()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(cm[i, j]),
                     ha='center', va='center', color='red')
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def predict_and_visualize_samples(model, dataset, num_samples, device, class_names):
    """
    Randomly select num_samples images, predict, and visualize in a grid if num_samples > 1.
    """
    model.eval()
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))

    print(f"\nRandomly selected {len(indices)} samples for prediction:")

    # Prepare grid visualization
    cols = min(num_samples, 5)
    rows = (len(indices) + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 4 * rows))

    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img_input = img.unsqueeze(0).to(device)
        output = model(img_input)
        _, pred = torch.max(output, 1)
        pred = pred.item()

        # Print true and predicted labels
        print(f"Index {idx}: True = {class_names[label]}, Pred = {class_names[pred]}")

        # Add subplot for the image
        plt.subplot(rows, cols, i + 1)
        if img.shape[0] == 1:  # Grayscale image
            plt.imshow(img.squeeze(0), cmap='gray')
        else:  # RGB image
            plt.imshow(img.permute(1, 2, 0))
        plt.title(f"True: {class_names[label]}\nPred: {class_names[pred]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('random_predictions.png')
    plt.show()
    print("Random sample predictions saved as random_predictions.png")


def main():
    # Get class information
    data_info = get_data_info()
    num_classes = data_info['num_classes']
    class_names = ["AD", "NC"] if num_classes == 2 else [f"Class {i}" for i in range(num_classes)]
    print(f"Number of classes: {num_classes} | Class names: {class_names}")

    # Initialize model (must match training configuration)
    model = convnext_small(num_classes=num_classes, in_chans=1).to(DEVICE)

    # Load checkpoint
    print(f"Loading checkpoint: {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(" Weights loaded successfully")

    # Predict entire test set
    y_true, y_pred = predict(model, test_loader, DEVICE)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    print("Confusion matrix saved as confusion_matrix.png")

    # Calculate overall accuracy
    acc = (y_true == y_pred).mean() * 100
    print(f"\nTest set accuracy: {acc:.2f}%")

    # Randomly sample and visualize
    test_dataset = test_loader.dataset
    predict_and_visualize_samples(model, test_dataset, NUM, DEVICE, class_names)


if __name__ == "__main__":
    main()