import os
import math
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from dataset import train_loader, val_loader, get_data_info, test_loader  # Import data loaders from dataset
from modules import convnext_small, convnext_tiny, convnext_micro, convnext_base  # Model definitions

# ==================== Configuration ====================
EPOCHS = 150  # Number of epochs
BASE_LR = 2e-4  # Base learning rate
MIN_LR = 5e-5  # Minimum learning rate
WARMUP_EPOCHS = 5  # Number of warmup epochs
WEIGHT_DECAY = 0.15  # Weight decay
LABEL_SMOOTHING = 0.1  # Label smoothing
DROP_PATH_RATE = 0.5  # Drop path rate
GRAD_CLIP_NORM = 1.0  # Gradient clipping norm
USE_EMA = True  # Use Exponential Moving Average
SEED = 42  # Random seed
EVAL_WITH_EMA = True  # Evaluate with EMA weights
TEST_TTA = False  # Use Test-Time Augmentation

USE_MIXUP = False  # Use Mixup augmentation
MIXUP_ALPHA = 0.4  # Mixup alpha value
EARLYSTOP_PATIENCE = 5  # Early stopping patience

# ==================== Utility Functions ====================
def grid_search_lr_weight_decay(model, train_loader, val_loader, device, num_classes):
    """
    Perform grid search for learning rate and weight decay, recording the average validation loss.
    """
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
    weight_decays = [0.1, 0.2, 0.3, 0.4]
    results = []

    for lr in learning_rates:
        for wd in weight_decays:
            print(f"Training with LR={lr}, Weight Decay={wd}")
            
            # Initialize model, optimizer, and scheduler
            model = convnext_tiny(num_classes=num_classes, in_chans=1).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            scheduler = build_warmup_cosine_scheduler(
                optimizer, max_epochs=5, warmup_epochs=1, base_lr=lr, min_lr=lr * 0.1
            )
            criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

            # Train for 5 epochs
            train_losses, val_losses = [], []
            for epoch in range(20):
                train_loss, _ = train_one_epoch(
                    model, train_loader, criterion, optimizer, device, epoch, use_amp=False, scaler=None
                )
                val_loss, _ = validate(model, val_loader, criterion, device, num_classes)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                scheduler.step()

            # Record results
            avg_val_loss = sum(val_losses) / len(val_losses)
            results.append((lr, wd, avg_val_loss))
            print(f"LR={lr}, WD={wd}, Avg Val Loss={avg_val_loss:.4f}")

    # Find the best combination
    best_lr, best_wd, best_loss = min(results, key=lambda x: x[2])
    print(f"Best LR={best_lr}, Best WD={best_wd}, Avg Val Loss={best_loss:.4f}")
    return best_lr, best_wd

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

def set_seed(seed=SEED):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def average_accuracy(output, target, num_classes):
    """Calculate the accuracy for each class and return the average accuracy."""
    with torch.no_grad():
        _, pred = output.max(1)  # Get predicted class
        correct = pred.eq(target).sum().item()  # Count correctly predicted samples
        total = target.size(0)  # Total number of samples
        return (correct / total) * 100  # Convert to percentage

def build_warmup_cosine_scheduler(optimizer, max_epochs, warmup_epochs, base_lr, min_lr):
    """Build a learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        # Cosine decay returns values in [min_lr/base_lr, 1]
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
        """Update EMA weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Apply EMA weights."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

def mixup_data(x, y, alpha: float):
    """Return mixed images, two sets of targets, and the mixup coefficient lam."""
    if alpha is None or alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    """Compute the mixup loss."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Implementation of Focal Loss.
        :param alpha: Class weight, default is 1.
        :param gamma: Focusing parameter for hard-to-classify samples, default is 2.
        :param reduction: Aggregation method for the loss, 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass to compute Focal Loss.
        :param inputs: Model predictions (logits).
        :param targets: Ground truth labels.
        :return: Focal Loss value.
        """
        # Convert logits to probabilities
        probs = torch.softmax(inputs, dim=1)
        # Get probabilities of the correct class
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1)
        p_t = (probs * targets_one_hot).sum(dim=1)

        # Compute Focal Loss
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t + 1e-8)

        # Aggregate the loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_amp, scaler, ema=None):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

   
        if USE_MIXUP:
            images_mix, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        else:
            images_mix, y_a, y_b, lam = images, labels, labels, 1.0

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(images_mix)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam) if USE_MIXUP else criterion(outputs, labels)
            scaler.scale(loss).backward()
            if GRAD_CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images_mix)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam) if USE_MIXUP else criterion(outputs, labels)
            loss.backward()
            if GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

        if ema is not None:
            ema.update()

        batch_size = labels.size(0)
  
        acc_a = average_accuracy(outputs, y_a, num_classes=len(loader.dataset.dataset.classes))
        if USE_MIXUP:
            acc_b = average_accuracy(outputs, y_b, num_classes=len(loader.dataset.dataset.classes))
            acc = lam * acc_a + (1.0 - lam) * acc_b
        else:
            acc = acc_a

        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
        total += batch_size

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

    epoch_loss = running_loss / total
    epoch_acc = running_acc / total
    return epoch_loss, epoch_acc

from sklearn.metrics import classification_report

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, ema=None):

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
            acc = average_accuracy(outputs, labels, num_classes)

            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            total += batch_size

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.2f}%'
            })

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["AD", "NC"]))

        epoch_loss = running_loss / total
        epoch_acc = running_acc / total
        return epoch_loss, epoch_acc
    finally:
        if ema is not None:
            ema.restore()

@torch.no_grad()
def test(model, loader, device, num_classes, ema=None, use_tta=False):
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

            if use_tta:
                logits = 0
                for x in (images, torch.flip(images, dims=[-1]), torch.flip(images, dims=[-2])):
                    logits = logits + model(x)
                outputs = logits / 3.0
            else:
                outputs = model(images)

            batch_size = labels.size(0)
            acc = average_accuracy(outputs, labels, num_classes)

            running_acc += acc * batch_size
            total += batch_size


            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'acc': f'{acc:.2f}%'
            })

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=["AD", "NC"]))

        epoch_acc = running_acc / total
        return epoch_acc
    finally:
        if ema is not None:
            ema.restore()

import matplotlib.pyplot as plt 


def main():
    # set_seed(SEED)  

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)


    data_info = get_data_info()
    num_classes = data_info['num_classes']
    print(f'device: {device} | num_classes: {num_classes}')


    model = convnext_small(num_classes=num_classes, in_chans=1, drop_path_rate=DROP_PATH_RATE).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

    scheduler = build_warmup_cosine_scheduler(
        optimizer, max_epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
        base_lr=BASE_LR, min_lr=MIN_LR
    )


    ema = EMA(model, decay=0.999) if USE_EMA else None


    start_epoch = 0
    ckpt_path = 'checkpoint_best.pth'
    ckpt_path_ema = 'checkpoint_best_ema.pth'

    # Early Stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    if ema is not None:
        ema.apply_shadow()                     
        torch.save(model.state_dict(), ckpt_path_ema)
        ema.restore()                        

    train_losses, val_losses = [], []
    train_accs, val_accs, test_accs = [], [], []

    t0 = time.time()
    for epoch in range(start_epoch, EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 60)

        train_loss, train_top1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp, scaler, ema=ema
        )


        val_loss, val_acc = validate(
            model, val_loader, criterion, device, num_classes,
            ema=ema if (EVAL_WITH_EMA and ema is not None) else None
        )


        test_acc = test(
            model, test_loader, device, num_classes,
            ema=ema if (EVAL_WITH_EMA and ema is not None) else None,
            use_tta=TEST_TTA
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\nTrain    - loss: {train_loss:.4f}, acc: {train_top1:.2f}%')
        print(f'Validate - loss: {val_loss:.4f}, acc: {val_acc:.2f}%')
        print(f'Test     - acc: {test_acc:.2f}%')
        print(f'LR: {current_lr:.6f}')

        # Early Stopping 
        early_stopping(val_loss, model, ckpt_path)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        # 保存当前 epoch 的 loss 和 acc
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_top1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)


    print("\nReloading best model weights for final test evaluation...")
    model.load_state_dict(torch.load(ckpt_path))
    final_test_acc = test(
        model, test_loader, device, num_classes,
        ema=ema if (EVAL_WITH_EMA and ema is not None) else None,
        use_tta=TEST_TTA
    )

    dt = (time.time() - t0) / 60.0
    print('\n' + '=' * 60)
    print(f'Training complete: {dt:.2f} 分钟')
    print(f'Final Val Accuracy: {val_accs[-1]:.2f}%')
    print(f'Final Test Accuracy: {final_test_acc:.2f}%')
    print('=' * 60)

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, test_accs)


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, test_accs):
    """
    绘制训练过程中的 loss 和 acc 曲线，并保存为三张图片
    """
    epochs = range(1, len(train_losses) + 1)

    # 绘制 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_loss.png')  # 保存 Loss 曲线
    plt.close()

    # 绘制 Train 和 Validation Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accs, label='Train Accuracy', color='blue')
    plt.plot(epochs, val_accs, label='Val Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('train_val_accuracy.png')  # 保存 Train 和 Validation Accuracy 曲线
    plt.close()


if __name__ == '__main__':
    main()