"""
Dataset Preparation and Image Augmentation
For training disease classification model on 1000+ medical images
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json


class MedicalImageAugmentation:
    """Advanced augmentation for medical images"""
    
    @staticmethod
    def get_train_transforms(image_size=300):
        """Augmented transforms for training data"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.2),
            A.Rotate(limit=30, p=0.7),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.GaussBlur(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_val_transforms(image_size=300):
        """Transforms for validation/test data"""
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    
    @staticmethod
    def augment_image_pil(image_path, num_augmentations=3):
        """Generate augmented versions of a single image using PIL"""
        image = Image.open(image_path).convert('RGB')
        augmented_images = [image]
        
        augmentations = [
            lambda img: ImageOps.mirror(img),
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            lambda img: img.rotate(15),
            lambda img: img.rotate(-15),
            lambda img: ImageEnhance.Brightness(img).enhance(1.2),
            lambda img: ImageEnhance.Brightness(img).enhance(0.8),
            lambda img: ImageEnhance.Contrast(img).enhance(1.3),
            lambda img: ImageEnhance.Contrast(img).enhance(0.7),
            lambda img: ImageEnhance.Sharpness(img).enhance(1.5),
        ]
        
        np.random.shuffle(augmentations)
        for aug in augmentations[:min(num_augmentations, len(augmentations))]:
            try:
                augmented_images.append(aug(image.copy()))
            except:
                pass
        
        return augmented_images


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for medical images
    Supports classification with multiple disease categories
    """
    
    CLASSES = {
        'normal': 0,
        'ovarian_cyst': 1,
        'pcos': 2,
        'breast_cancer': 3
    }
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: list of image file paths
            labels: list of labels (class indices)
            transform: augmentation transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label


class DatasetBuilder:
    """Build and manage training datasets from medical images"""
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.image_paths = []
        self.labels = []
        self.class_distribution = {}
    
    def load_dataset(self, class_folders=None):
        """
        Load images from folder structure
        Expects: data_directory/class_name/*.jpg
        """
        if class_folders is None:
            class_folders = MedicalImageDataset.CLASSES.keys()
        
        for class_name in class_folders:
            class_folder = os.path.join(self.data_directory, class_name)
            if not os.path.exists(class_folder):
                continue
            
            class_label = MedicalImageDataset.CLASSES.get(class_name, -1)
            if class_label == -1:
                continue
            
            image_count = 0
            for filename in os.listdir(class_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_path = os.path.join(class_folder, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(class_label)
                    image_count += 1
            
            self.class_distribution[class_name] = image_count
            print(f"Loaded {image_count} images from {class_name}")
        
        print(f"\nTotal images loaded: {len(self.image_paths)}")
        return len(self.image_paths)
    
    def get_class_distribution(self):
        """Get distribution of images across classes"""
        return self.class_distribution
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        """
        # First split: train + val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            self.image_paths, self.labels, test_size=test_size, random_state=random_state,
            stratify=self.labels
        )
        
        # Second split: train vs val
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=val_size/(1-test_size),
            random_state=random_state, stratify=train_val_labels
        )
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }
    
    def create_dataloaders(self, batch_size=32, num_workers=0, test_size=0.2, val_size=0.1):
        """Create PyTorch dataloaders for train, val, and test"""
        # Get augmentation transforms
        train_transform = MedicalImageAugmentation.get_train_transforms()
        val_transform = MedicalImageAugmentation.get_val_transforms()
        
        # Split dataset
        split_data = self.split_dataset(test_size=test_size, val_size=val_size)
        
        # Create datasets
        train_dataset = MedicalImageDataset(
            split_data['train'][0], split_data['train'][1], transform=train_transform
        )
        val_dataset = MedicalImageDataset(
            split_data['val'][0], split_data['val'][1], transform=val_transform
        )
        test_dataset = MedicalImageDataset(
            split_data['test'][0], split_data['test'][1], transform=val_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }
    
    def save_dataset_info(self, output_path='dataset_info.json'):
        """Save dataset information and metadata"""
        info = {
            'total_images': len(self.image_paths),
            'class_distribution': self.class_distribution,
            'image_paths': self.image_paths,
            'labels': self.labels
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        return output_path


def validate_dataset(data_directory):
    """Validate dataset structure and count images"""
    builder = DatasetBuilder(data_directory)
    total = builder.load_dataset()
    distribution = builder.get_class_distribution()
    
    print("\n" + "="*50)
    print("DATASET VALIDATION REPORT")
    print("="*50)
    print(f"Total Images: {total}")
    print("\nClass Distribution:")
    for class_name, count in distribution.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    if total >= 1000:
        print(f"\n✓ Dataset is sufficient for training ({total} images)")
    else:
        print(f"\n⚠ Dataset may need augmentation or more images ({total} current)")
    
    return builder


if __name__ == "__main__":
    # Example usage
    print("Medical Image Dataset Preparation Module")
