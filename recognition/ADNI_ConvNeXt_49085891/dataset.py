import os
import json
import random
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

BATCH_SIZE = 32
NUM_WORKERS = 16
VAL_SPLIT = 0.1
MEAN = 0.1177
STD = 0.2220
TRAIN_DATA_PATH = 'ADNI/AD_NC/train'
TEST_DATA_PATH = 'ADNI/AD_NC/test'
META_PATH = 'ADNI/meta_data_with_label.json'  
IMG_EXTS = ('.jpg', '.jpeg')

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=(-10,10), translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
    transforms.RandomErasing(p=0.25, scale=(0.1, 0.2), ratio=(0.3, 3.3), value='random'),
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
])


def _load_meta_index(meta_json: str) -> Dict[str, str]:
    """
    Load meta_data_with_label.json and return {basename: patient_id} index.
    Supports both list and dictionary structures, and tries to extract patient_id/path from common keys.
    """
    if not os.path.isfile(meta_json):
        print(f"[WARN] Metadata not found: {meta_json}. Defaulting to grouping by filename, which may cause patient leakage.")
        return {}

    with open(meta_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    index = {}
    def pick(d: dict, keys: List[str], default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, dict):
                pid = pick(v, ['patient_id', 'subject_id', 'pid', 'RID', 'sid'])
                path = pick(v, ['path', 'filepath', 'image', 'img', 'filename'], k)
            else:
                pid = v
                path = k
            basename = os.path.basename(str(path))
            index[basename] = str(pid) if pid is not None else None
    elif isinstance(meta, list):
        for item in meta:
            if not isinstance(item, dict):
                continue
            pid = pick(item, ['patient_id', 'subject_id', 'pid', 'RID', 'sid'])
            path = pick(item, ['path', 'filepath', 'image', 'img', 'filename'])
            if path is None:
                continue
            basename = os.path.basename(str(path))
            index[basename] = str(pid) if pid is not None else None
    else:
        print(f"[WARN] Unrecognized metadata structure: {type(meta)}. Defaulting to grouping by filename.")
    return index

class CustomImageDataset(Dataset):
    def __init__(self, root: str, transform=None, meta_index: Dict[str, str] = None):
        self.root = root
        self.transform = transform
        self.meta_index = meta_index or {}
        self.classes = sorted([d.name for d in os.scandir(root) if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        self.patient_ids: List[str] = []

        for cls in self.classes:
            cls_idx = self.class_to_idx[cls]
            cls_dir = os.path.join(root, cls)
            for fn in sorted(os.listdir(cls_dir)):
                if not fn.lower().endswith(IMG_EXTS):
                    continue
                path = os.path.join(cls_dir, fn)
                self.samples.append((path, cls_idx))
                pid = self.meta_index.get(os.path.basename(fn))
                if pid is None:
                    stem = os.path.splitext(fn)[0]
                    pid = stem.split('_')[0]
                self.patient_ids.append(str(pid))

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def center_brain(img: Image.Image, target_size: Tuple[int, int] = (240, 256), padding_factor: float = 1.5) -> Image.Image:
        '''
        Args:
            img: PIL Image object (L mode).
            target_size: The final output image size, e.g., (224, 224).
            padding_factor: A factor to add extra space around the brain's bounding box.
                            For example, 1.5 means adding 50% more space in each direction.
        Returns:
            Processed PIL Image object.
        '''
        img_arr = np.array(img)
        mask = img_arr > np.percentile(img_arr, 1) # Use a low percentile to determine the brain region, avoiding misidentifying lesions as background
        
        ys, xs = np.nonzero(mask)
        y_min_brain, y_max_brain = ys.min(), ys.max()
        x_min_brain, x_max_brain = xs.min(), xs.max()

        center_y = (y_min_brain + y_max_brain) / 2 # Calculate the center and size of the brain region
        center_x = (x_min_brain + x_max_brain) / 2
        height_brain = y_max_brain - y_min_brain
        width_brain = x_max_brain - x_min_brain
        
        max_brain_dim = max(height_brain, width_brain)  # Determine a new, looser crop box (centered on the brain and enlarged)
        
        crop_dim = int(max_brain_dim * padding_factor) # The side length of the new crop box, ensuring the entire brain region is included with extra space
        
        x_min_crop = int(max(0, center_x - crop_dim / 2)) # Ensure the crop box does not exceed the original image boundaries
        y_min_crop = int(max(0, center_y - crop_dim / 2))
        
        x_max_crop = int(min(img_arr.shape[1], center_x + crop_dim / 2))
        y_max_crop = int(min(img_arr.shape[0], center_y + crop_dim / 2))
        
        cropped_arr = img_arr[y_min_crop:y_max_crop, x_min_crop:x_max_crop] # Ensure the crop region is square (by further adjusting boundaries or padding)
        cropped_img = Image.fromarray(cropped_arr)

        max_side = max(cropped_img.size) # Place the cropped image at the center of the target size without changing the aspect ratio
        padded_square = ImageOps.pad(cropped_img, (max_side, max_side), color=0, centering=(0.5, 0.5))
        resized_img = padded_square.resize(target_size, Image.BILINEAR)
  
        return resized_img

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')

        # Center the brain region
        img = self.center_brain(img)

        # Apply transformations
        if self.transform:
            img = self.transform(img)
        return img, label


def split_by_patient(dataset: CustomImageDataset, val_ratio=0.2, seed=42) -> Tuple[List[int], List[int]]:
    """
    Split dataset by patient_id to ensure no patient appears in both train and validation sets.
    :param dataset: The dataset to split, which includes patient IDs for each sample.
    :param val_ratio: The ratio of the dataset to allocate to the validation set.
    :param seed: Random seed for reproducibility.
    :return: Two lists of indices, one for the training set and one for the validation set.
    """
    pid_to_indices: Dict[str, List[int]] = {} # Group indices by patient ID
    for i, pid in enumerate(dataset.patient_ids):
        pid_to_indices.setdefault(pid, []).append(i)

    pids = list(pid_to_indices.keys())  # Shuffle the patient IDs to ensure randomness
    rnd = random.Random(seed)
    rnd.shuffle(pids)

    total = len(dataset)
    target_val = int(round(total * val_ratio))
    val_indices: List[int] = []
    train_indices: List[int] = []

    # Allocate indices to validation or training sets based on the target validation size
    acc = 0
    for pid in pids:
        inds = pid_to_indices[pid]
        if acc < target_val:
            val_indices.extend(inds)
            acc += len(inds)
        else:
            train_indices.extend(inds)

    # Shuffle the final indices to ensure randomness within each set
    rnd.shuffle(train_indices)
    rnd.shuffle(val_indices)
    return train_indices, val_indices

_meta_index = _load_meta_index(META_PATH)

full_train_dataset = CustomImageDataset(TRAIN_DATA_PATH, transform=train_transform, meta_index=_meta_index)
train_idx, val_idx = split_by_patient(full_train_dataset, val_ratio=VAL_SPLIT, seed=42)

train_dataset = Subset(full_train_dataset, train_idx)
val_dataset = Subset(
    CustomImageDataset(TRAIN_DATA_PATH, transform=val_test_transform, meta_index=_meta_index),
    val_idx
)

test_dataset = CustomImageDataset(TEST_DATA_PATH, transform=val_test_transform, meta_index=_meta_index)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

def get_data_info():
    """
    Print detailed information about the dataset.
    """
    return {
        'num_classes': len(full_train_dataset.classes),
        'classes': full_train_dataset.classes,
        'class_to_idx': full_train_dataset.class_to_idx,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset),
        'batch_size': BATCH_SIZE
    }

def print_data_info():
    info = get_data_info()
    print("=" * 50)
    print("Dataset Information:")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class names: {info['classes']}")
    print(f"Number of training samples: {info['train_samples']}")
    print(f"Number of validation samples: {info['val_samples']}")
    print(f"Number of testing samples: {info['test_samples']}")
    print(f"Batch size: {info['batch_size']}")
    print("=" * 50)

# calculate_mean_std and visualize_augmented_images functions are only used for mean and std calculation 
def calculate_mean_std(loader):
    """
    Calculate the mean and standard deviation of a dataset.
    :param loader: DataLoader for the dataset
    :return: Tuple (mean, std)
    """
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        # Flatten the images to calculate mean and std across all pixels
        batch_size = images.size(0)
        images = images.view(batch_size, -1)  # Flatten to (batch_size, num_pixels)
        mean += images.mean(dim=1).sum().item()
        std += images.std(dim=1).sum().item()
        total_images += batch_size

    mean /= total_images
    std /= total_images

    return mean, std


def calculate_dataset_mean_std(dataset_path, meta_index, batch_size=32, num_workers=4):
    """
    Calculate the mean and standard deviation of a dataset.
    :param dataset_path: Path to the dataset
    :param meta_index: Metadata index for the dataset
    :param batch_size: Batch size for the DataLoader
    :param num_workers: Number of workers for the DataLoader
    :return: Tuple (mean, std)
    """
    # Create a temporary DataLoader for the dataset without normalization
    temp_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    temp_dataset = CustomImageDataset(dataset_path, transform=temp_transform, meta_index=meta_index)
    temp_loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Calculate mean and std
    mean, std = calculate_mean_std(temp_loader)
    print(f"Calculated Mean: {mean:.4f}, Std: {std:.4f}")
    return mean, std


def visualize_augmented_images(loader, num_images=5, class_names=None):
    """
    Visualize augmented images from the data loader.
    :param loader: Data loader
    :param num_images: Number of images to display
    :param class_names: List of class names (e.g., ['AD', 'NC'])
    """
    # Get a batch of images and labels
    images, labels = next(iter(loader))

    # Denormalize the images (restore them to a visualizable range)
    mean = torch.tensor([MEAN])
    std = torch.tensor([STD])
    images = images * std[None, :, None, None] + mean[None, :, None, None]

    # Display the images
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        img = images[i].squeeze(0).numpy()  # Convert to numpy format
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        label = class_names[labels[i].item()] if class_names else labels[i].item()
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("augmented_images.png")


if __name__ == "__main__":
    # Uncomment to recalculate mean and std
    #mean, std = calculate_dataset_mean_std(TRAIN_DATA_PATH, _meta_index, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print_data_info()
    visualize_augmented_images(val_loader, num_images=5, class_names=full_train_dataset.classes)
