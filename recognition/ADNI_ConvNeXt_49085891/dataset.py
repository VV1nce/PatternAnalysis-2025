import os
import json
import random
from typing import List, Tuple, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

BATCH_SIZE = 32
NUM_WORKERS = 64
VAL_SPLIT = 0.2
TRAIN_DATA_PATH = 'ADNI/AD_NC/train'
TEST_DATA_PATH = 'ADNI/AD_NC/test'
META_PATH = 'ADNI/meta_data_with_label.json'  

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.116], std=[0.225]),
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.117], std=[0.226]),
])

IMG_EXTS = ('.jpg', '.jpeg')

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

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, label

def split_by_patient(dataset: CustomImageDataset, val_ratio=0.2, seed=42) -> Tuple[List[int], List[int]]:
    """
    Split dataset by patient_id to ensure no patient appears in both train and validation sets.
    """
    pid_to_indices: Dict[str, List[int]] = {}
    for i, pid in enumerate(dataset.patient_ids):
        pid_to_indices.setdefault(pid, []).append(i)

    pids = list(pid_to_indices.keys())
    rnd = random.Random(seed)
    rnd.shuffle(pids)

    total = len(dataset)
    target_val = int(round(total * val_ratio))
    val_indices: List[int] = []
    train_indices: List[int] = []

    acc = 0
    for pid in pids:
        inds = pid_to_indices[pid]
        if acc < target_val:
            val_indices.extend(inds)
            acc += len(inds)
        else:
            train_indices.extend(inds)

    rnd.shuffle(train_indices)
    rnd.shuffle(val_indices)
    return train_indices, val_indices

def check_patient_overlap(train_dataset, val_dataset):
    """
    Check if there are overlapping patients between the training and validation sets.
    """
    train_patient_ids = set(train_dataset.dataset.patient_ids[i] for i in train_dataset.indices)
    val_patient_ids = set(val_dataset.dataset.patient_ids[i] for i in val_dataset.indices)

    overlap = train_patient_ids.intersection(val_patient_ids)
    if overlap:
        print(f"Overlapping patients found: {len(overlap)} patients.")
        print(overlap)
    else:
        print("No overlapping patients between training and validation sets.")

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


if __name__ == "__main__":
    print_data_info()
    check_patient_overlap(train_dataset, val_dataset)