import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image


# 数据路径（请替换为你的实际路径）
train_data_path = 'ADNI/AD_NC/train'  
test_data_path = 'ADNI/AD_NC/test'      


# dataset.py
train_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.116], std=[0.225]),

])

# this for validation and test
test_transform = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.117], std=[0.226])
])

def calculate_mean_std(loader):
    """
    计算数据集的均值和标准差
    """
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        # get
        batch_samples = images.size(0)  # 当前批次的样本数
        total_samples += batch_samples

        # 计算当前批次的均值和标准差
        mean += images.mean([0, 2, 3]) * batch_samples
        std += images.std([0, 2, 3]) * batch_samples

    # 计算全局均值和标准差
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

# ==================== 预配置的数据加载器 ====================

# Create dataset
train_dataset = CustomImageDataset(root=train_data_path, transform=train_transform)
test_dataset = CustomImageDataset(root=test_data_path, transform=test_transform)

# 配置参数
BATCH_SIZE = 64
NUM_WORKERS = 0

# 创建数据加载器
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS,
    pin_memory=False  # 加速GPU传输
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS,
    pin_memory=False
)

# ==================== 便捷函数 ====================

def get_data_info():
    """返回数据集信息"""
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
    """打印数据集信息"""
    info = get_data_info()
    print("=" * 50)
    print("数据集信息:")
    print(f"类别数: {info['num_classes']}")
    print(f"类别名称: {info['classes']}")
    print(f"训练集样本数: {info['train_samples']}")
    print(f"测试集样本数: {info['test_samples']}")
    print(f"批次大小: {info['batch_size']}")
    print("=" * 50)

def get_sample_batch():
    """获取一个样本批次用于测试"""
    for images, labels in train_loader:
        return images, labels
    


def compute_mean_std(root):
    """
    计算数据集的均值和标准差
    root: 数据集根目录，包含类别子文件夹
    """
    tf = transforms.ToTensor()  # 将图像转换为张量
    s1, s2, n = 0.0, 0.0, 0
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img = Image.open(img_path).convert('L')  # 转为灰度图像
                img_tensor = tf(img)  # 转为张量
                s1 += img_tensor.mean().item()  # 累加均值
                s2 += img_tensor.pow(2).mean().item()  # 累加平方均值
                n += 1
    mean = s1 / n
    std = (s2 / n - mean**2)**0.5
    return mean, std
    
    # ==================== initialise ====================
if __name__ == "__main__":

    print_data_info()
    
    # load test data
    images, labels = get_sample_batch()
    mean, std = compute_mean_std(train_data_path)
    print(f"训练集均值: {mean:.3f} 训练集标准差: {std:.3f}")
    mean, std = compute_mean_std(test_data_path)
    print(f"测试集均值: {mean:.3f} 测试集 标准差: {std:.3f}")
    print(f"样本图像形状: {images.shape}")
    print(f"样本标签形状: {labels.shape}")
    print(f"图像数据范围: {images.min():.3f} 到 {images.max():.3f}")
