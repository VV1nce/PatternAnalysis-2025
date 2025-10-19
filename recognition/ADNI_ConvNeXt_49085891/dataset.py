import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
from PIL import Image
import random

# File paths
train_data_path = 'ADNI/AD_NC/train'  
test_data_path = 'ADNI/AD_NC/test'      

# Data augmentation
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.116], std=[0.225]),
])

# Data preprocessing for validation and testing
test_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(256),
])

val_test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.117], std=[0.226])
])

def calculate_mean_std(loader):
    """
    Calculate the mean and standard deviation of a dataset.
    """
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        # Get the number of samples in the current batch
        batch_samples = images.size(0)
        total_samples += batch_samples

        # Calculate the mean and standard deviation for the current batch
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples

    # Calculate the global mean and standard deviation
    mean /= total_samples
    std /= total_samples

    return mean, std

class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        # Store the root directory and the transform
        self.root = root
        self.transform = transform

        # Discover class names from the subdirectories
        self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        # Create a mapping from class name to integer index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Create a list of all samples (image paths and their labels)
        self.samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(root, class_name)
            for file_name in sorted(os.listdir(class_dir)):
                # Ensure we are only picking up image files (simple check)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    path = os.path.join(class_dir, file_name)
                    item = (path, class_idx)
                    self.samples.append(item)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        """
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  
        if self.transform:
            image = self.transform(image)
        return image, label

# ==================== DataLoader Configuration ====================
# Configuration parameters
BATCH_SIZE = 16
NUM_WORKERS = 4
VAL_SPLIT = 0.2  # Validation set proportion of training set

# Create the full training dataset
full_train_dataset = CustomImageDataset(root=train_data_path, transform=train_transform)

# Split the training set into training and validation
train_size = int((1 - VAL_SPLIT) * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_indices, val_indices = torch.utils.data.random_split(
    range(len(full_train_dataset)), [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

# Create subsets
train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

# Apply different transform for validation set (no data augmentation)
val_dataset.dataset.transform = val_test_transform

# Test dataset
test_dataset = CustomImageDataset(root=test_data_path, transform=val_test_transform)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=True  
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ==================== Dataset test Functions ====================

def get_data_info():
    """Return dataset information"""
    info = {
        'num_classes': len(full_train_dataset.classes),
        'classes': full_train_dataset.classes,
        'class_to_idx': full_train_dataset.class_to_idx,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'batch_size': BATCH_SIZE
    }
    return info

def print_data_info():
    """Print dataset information"""
    info = get_data_info()
    print("=" * 50)
    print("Dataset Information:")
    print(f"Number of Classes: {info['num_classes']}")
    print(f"Class Names: {info['classes']}")
    print(f"Training Samples: {info['train_samples']}")
    print(f"Validation Samples: {info['val_samples']}")
    print(f"Test Samples: {info['test_samples']}")
    print(f"Batch Size: {info['batch_size']}")
    print("=" * 50)

def get_sample_batch():
    """Get a sample batch for testing"""
    for images, labels in train_loader:
        return images, labels
    

def compute_mean_std(root):
    """
    Compute the mean and standard deviation of the dataset
    root: Root directory of the dataset, containing class subfolders
    """
    tf = transforms.ToTensor()  # Convert image to tensor
    s1, s2, n = 0.0, 0.0, 0
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_tensor = tf(img)  # Convert to tensor
                s1 += img_tensor.mean().item()  # Accumulate mean
                s2 += img_tensor.pow(2).mean().item()  # Accumulate squared mean
                n += 1
    mean = s1 / n
    std = (s2 / n - mean**2)**0.5
    return mean, std

# ==================== Dataset Test Initialization ====================
if __name__ == "__main__":
    print_data_info()

    # Test data loading
    images, labels = get_sample_batch()
    mean, std = compute_mean_std(train_data_path)
    print(f"Training Set Mean: {mean:.3f} Training Set Std: {std:.3f}")
    mean, std = compute_mean_std(test_data_path)
    print(f"Test Set Mean: {mean:.3f} Test Set Std: {std:.3f}")
    print(f"Sample Image Shape: {images.shape}")
    print(f"Sample Label Shape: {labels.shape}")
    print(f"Image Data Range: {images.min():.3f} to {images.max():.3f}")