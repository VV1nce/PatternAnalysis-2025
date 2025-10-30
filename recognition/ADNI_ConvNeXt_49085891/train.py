import os
import math
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report

from dataset import train_loader, val_loader, get_data_info, test_loader  
from modules import convnext_tiny, convnext_small, convnext_base 

# Training hyperparameters
EPOCHS = 150 
BASE_LR = 2e-4 
MIN_LR = 5e-5 
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 0.15
LABEL_SMOOTHING = 0.1 
DROP_PATH_RATE = 0.5 
GRAD_CLIP_NORM = 1.0  
USE_EMA = True
EVAL_WITH_EMA = True
EARLYSTOP_PATIENCE = 5

class EarlyStopping:
    """
    Early stopping to stop training when the validation loss doesn't improve after a certain patience.
    """
    def __init__(self, patience=EARLYSTOP_PATIENCE, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model, ckpt_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_path):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), ckpt_path)
        self.val_loss_min = val_loss

def overall_accuracy(output, target):
    """
    Calculates the accuracy for each class and returns the overall accuracy
    """
    with torch.no_grad():
        _, pred = output.max(1)  # Get the predicted class
        correct = pred.eq(target).sum().item()  # Calculate the number of correct predictions
        total = target.size(0)  # Total number of samples
        return (correct / total) * 100  # Convert to percentage

def build_warmup_cosine_scheduler(optimizer, max_epochs, warmup_epochs, base_lr, min_lr):
    """
    Builds a learning rate scheduler with linear warmup and cosine annealing.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # cos part returns [min_lr/base_lr, 1]
        progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.backup = {}

    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Apply EMA weights"""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
         
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_amp, scaler, ema=None):
    """
    Train the model for one epoch.
    :param model: The model to train.
    :param loader: DataLoader for the training dataset.
    :param criterion: Loss function.
    :param optimizer: Optimizer for updating model parameters.
    :param device: Device to use for training (e.g., 'cuda' or 'cpu').
    :param epoch: Current epoch number.
    :param use_amp: Whether to use Automatic Mixed Precision (AMP).
    :param scaler: Gradient scaler for AMP.
    :param ema: Exponential Moving Average object (optional).
    :return: Tuple containing the epoch loss and accuracy.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Accumulate the total loss for the epoch
    running_acc = 0.0  # Accumulate the total accuracy for the epoch
    total = 0  # Total number of samples processed

    # Progress bar for tracking training progress
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
    for images, labels in pbar:
        # Move images and labels to the specified device
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)  # Reset gradients

        if use_amp:
            # Use Automatic Mixed Precision (AMP) for faster training
            with autocast(device_type='cuda'):
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
            scaler.scale(loss).backward()  # Backward pass with scaled gradients
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # Clip gradients
            scaler.step(optimizer)  # Update model parameters
            scaler.update()  # Update the scaler for AMP
        else:
            # Standard training without AMP
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            if GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # Clip gradients
            optimizer.step()  # Update model parameters

        if ema is not None:
            ema.update()  # Update EMA weights if enabled

        batch_size = labels.size(0)  # Get the batch size
        acc = overall_accuracy(outputs, labels)  # Compute accuracy for the batch

        # Accumulate loss and accuracy for the epoch
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total += batch_size

        # Update progress bar with current metrics
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    # Compute average loss and accuracy for the epoch
    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, ema=None):
    """Validation set evaluation)"""
    if ema is not None:
        ema.apply_shadow()
    try:
        model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc='Validating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            acc = overall_accuracy(outputs, labels)

            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            total += batch_size

            # Save predictions and true labels
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["AD", "NC"]))

        epoch_loss = running_loss / total
        epoch_acc = running_acc / total
        return epoch_loss, epoch_acc
    finally:
        if ema is not None:
            ema.restore()

@torch.no_grad()
def test(model, loader, device, num_classes, ema=None):
    """
    Test set evaluation (without EMA weights). Test set is only evaluated once after training.
    """
    if ema is not None:
        ema.apply_shadow()
    try:
        model.eval()
        running_acc = 0.0
        total = 0

        all_preds = []
        all_labels = []

        pbar = tqdm(loader, desc='Testing')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            batch_size = labels.size(0)
            acc = overall_accuracy(outputs, labels)

            running_acc += acc * batch_size
            total += batch_size

            # Save predictions and true labels
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'acc': f'{acc:.2f}%'
            })

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["AD", "NC"]))

        epoch_acc = running_acc / total
        return epoch_acc
    finally:
        if ema is not None:
            ema.restore()


# ==================== 主流程 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Data information
    data_info = get_data_info()
    num_classes = data_info['num_classes']
    print(f'Using device: {device} | Number of classes: {num_classes}')

    # Model (Standard ConvNeXt setup: AdamW + DropPath + LabelSmoothing + CosineLR)
    model = convnext_small(num_classes=num_classes, in_chans=1, drop_path_rate=DROP_PATH_RATE).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    scheduler = build_warmup_cosine_scheduler(
        optimizer, max_epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR, min_lr=MIN_LR
    )

    # Initialize EMA
    ema = EMA(model, decay=0.999) if USE_EMA else None

    start_epoch = 0
    ckpt_path = 'checkpoint_best.pth'
    ckpt_path_ema = 'checkpoint_best_ema.pth'

    # Early Stopping
    early_stopping = EarlyStopping(patience=EARLYSTOP_PATIENCE, verbose=True)
    # If EMA is enabled, save EMA weights to the specified path
    if ema is not None:
        ema.apply_shadow()                     # Replace model weights with EMA weights
        torch.save(model.state_dict(), ckpt_path_ema)
        ema.restore()                          # Restore original training weights
    # For saving loss and acc during training
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 60)

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp, scaler, ema=ema
        )

 
        # Validation (can switch to using EMA weights)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, num_classes,
            ema=ema if (EVAL_WITH_EMA and ema is not None) else None
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\nTrain    - loss: {train_loss:.4f}, acc: {train_top1:.2f}%')
        print(f'Validate - loss: {val_loss:.4f}, acc: {val_acc:.2f}%')
        print(f'LR: {current_lr:.6f}')

        # Early Stopping check
        early_stopping(val_loss, model, ckpt_path)
        if ema is not None:
            # If the validation set has improved, resave the EMA weights
            if early_stopping.counter == 0:  # Indicates val_loss has improved
                ema.apply_shadow()
                torch.save(model.state_dict(), ckpt_path_ema)
                ema.restore()

        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        # Save current epoch's loss and acc
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_top1)
        val_accs.append(val_acc)
 
    # Load weights and test
    print("\nReloading best model weights for final test evaluation...")
    model.load_state_dict(torch.load(ckpt_path))
    final_test_acc = test(
        model, test_loader, device, num_classes,
        ema=ema if (EVAL_WITH_EMA and ema is not None) else None
    )

    dt = (time.time() - t0) / 60.0
    print('\n' + '=' * 60)
    print(f'Total time taken: {dt:.2f} mins')
    print(f'Testset accuracy: {final_test_acc:.2f}%')
    print('=' * 60)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plot loss and acc curves during training and save as two images
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_loss.png')  # Save Loss curve
    plt.close()

    # Plot Train and Validation Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_accuracy.png')  # Save Train and Validation Accuracy curve
    plt.close()


if __name__ == '__main__':
    main()