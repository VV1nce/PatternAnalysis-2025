import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import train_loader, test_loader, get_data_info
from modules import convnext_small

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
CHECKPOINT_PATH = 'checkpoint_best.pth'

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(correct / total) * 100:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(f"train Loss: {epoch_loss:.4f} | acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Validating')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)


        running_loss += loss.item() * labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{(correct / total) * 100:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100
    print(f"test Loss: {epoch_loss:.4f} | acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def main():
    data_info = get_data_info()
    num_classes = data_info['num_classes']
    print(f"num_classes: {num_classes}")


    model = convnext_small(num_classes=num_classes, in_chans=1).to(DEVICE)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)
            print(f"best model saved with acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()