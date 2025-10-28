import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import test_loader, get_data_info  # Import test dataset loader and data info function
from modules import convnext_small, convnext_tiny, convnext_base  # Import model definitions
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==================== Configuration ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')  # Select device (GPU, MPS, or CPU)
CHECKPOINT_PATH = 'checkpoint_best.pth'  # Path to the model checkpoint
NUM_EXAMPLES = 5  # Number of random examples to visualize

# ==================== Validation Function ====================
@torch.no_grad()
def validate_and_visualize(model, loader, device, num_examples=5):
    """
    Validate the test dataset, randomly select a few images as examples, and generate a confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []
    example_images = []
    example_preds = []
    example_trues = []
    example_confs = []

    criterion = nn.CrossEntropyLoss()  # Loss function
    pbar = tqdm(loader, desc='Testing')  # Progress bar for testing
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)  # Compute confidence scores
        confs, preds = torch.max(probs, dim=1)

        # Save all predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Randomly select examples
        if len(example_images) < num_examples:
            for i in range(images.size(0)):
                if len(example_images) < num_examples:
                    example_images.append(images[i].cpu())
                    example_preds.append(preds[i].item())
                    example_trues.append(labels[i].item())
                    example_confs.append(confs[i].item())

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return cm, example_images, example_preds, example_trues, example_confs

# ==================== Visualization Functions ====================
def plot_examples(images, preds, trues, confs, class_names):
    """
    Visualize example images, showing predictions, ground truths, and confidence scores.
    """
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        img = img.permute(1, 2, 0).numpy()  # Convert to HWC format
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f"Pred: {class_names[preds[i]]}\nTrue: {class_names[trues[i]]}\nConf: {confs[i]:.2f}")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# ==================== Main Process ====================
def main():
    # Load dataset information
    data_info = get_data_info()
    num_classes = data_info['num_classes']
    class_names = ['AD', 'NC']  # Manually define class names
    print(f'Number of classes in the test set: {num_classes}')

    # Load the model
    model = convnext_small(num_classes=num_classes, in_chans=1).to(DEVICE)

    # Load model weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])  # Load the 'model' key from the checkpoint
    print(f'Successfully loaded weights: {CHECKPOINT_PATH}')

    # Validate the test set and get examples and confusion matrix
    cm, example_images, example_preds, example_trues, example_confs = validate_and_visualize(
        model, test_loader, DEVICE, num_examples=NUM_EXAMPLES
    )

    # Visualize example images
    plot_examples(example_images, example_preds, example_trues, example_confs, class_names)

    # Plot the confusion matrix
    plot_confusion_matrix(cm, class_names)

if __name__ == '__main__':
    main()