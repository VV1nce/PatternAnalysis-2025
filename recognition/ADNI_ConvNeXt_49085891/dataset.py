import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


# Data paths (replace with your actual paths)
train_data_path = 'ADNI/AD_NC/train'  
test_data_path = 'ADNI/AD_NC/test'      


# Data augmentation and preprocessing for training
train_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.116], std=[0.225]),
])

# Data preprocessing for validation and testing
test_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(256),
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
                # Ensure we are only picking up image files 
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
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# ==================== Preconfigured Data Loaders ====================

# Create datasets
train_dataset = CustomImageDataset(root=train_data_path, transform=train_transform)
test_dataset = CustomImageDataset(root=test_data_path, transform=test_transform)

# Configuration parameters
BATCH_SIZE = 64
NUM_WORKERS = 0

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=False  # Accelerates GPU transfer
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=False
)

# ==================== Utility Functions ====================

def get_data_info():
    """Returns dataset information."""
    info = {
        'num_classes': len(train_dataset.classes),
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx,
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'batch_size': BATCH_SIZE
    }
    return info

def print_data_info():
    """Prints dataset information."""
    info = get_data_info()
    print("=" * 50)
    print("Dataset Information:")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class names: {info['classes']}")
    print(f"Number of training samples: {info['train_samples']}")
    print(f"Number of testing samples: {info['test_samples']}")
    print(f"Batch size: {info['batch_size']}")
    print("=" * 50)

def get_sample_batch():
    """Fetches a sample batch for testing."""
    for images, labels in train_loader:
        return images, labels
    

def compute_mean_std(root):
    """
    Compute the mean and standard deviation of a dataset.
    Args:
        root: Root directory of the dataset, containing class subdirectories.
    """
    tf = transforms.ToTensor()  # Convert images to tensors
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
    
# ==================== Main Execution ====================
if __name__ == "__main__":

    print_data_info()
    
    # Load test data
    images, labels = get_sample_batch()
    mean, std = compute_mean_std(train_data_path)
    print(f"Training set mean: {mean:.3f}, Training set std: {std:.3f}")
    mean, std = compute_mean_std(test_data_path)
    print(f"Testing set mean: {mean:.3f}, Testing set std: {std:.3f}")
    print(f"Sample image shape: {images.shape}")
    print(f"Sample label shape: {labels.shape}")
    print(f"Image data range: {images.min():.3f} to {images.max():.3f}")