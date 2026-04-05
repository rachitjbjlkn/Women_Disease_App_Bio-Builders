"""
GCN Segmentation Model Training Script
Trains on synthetic ultrasound data to detect hypoechoic cysts.
Usage: python src/train_segmentation.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.gcn_segmentation import UltraSoundGCN
from src.generate_training_data import generate_sample


class UltrasoundDataset(Dataset):
    """Dataset loader for generated synthetic ultrasound images."""

    def __init__(self, data_dir=None, n_synthetic=None, size=512):
        self.size = size
        self.samples = []

        if data_dir and Path(data_dir).exists():
            img_dir = Path(data_dir) / 'images'
            mask_dir = Path(data_dir) / 'masks'
            img_files = sorted(img_dir.glob('*.png'))
            for img_f in img_files:
                mask_f = mask_dir / img_f.name
                if mask_f.exists():
                    self.samples.append(('file', str(img_f), str(mask_f)))
            print(f"Loaded {len(self.samples)} samples from {data_dir}")
        elif n_synthetic:
            self.samples = [('synthetic', None, None)] * n_synthetic
            print(f"Will generate {n_synthetic} synthetic samples on-the-fly")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kind = self.samples[idx][0]

        if kind == 'file':
            img_path, mask_path = self.samples[idx][1], self.samples[idx][2]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.size, self.size))
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            img, mask, _ = generate_sample(self.size, self.size)
            mask = mask.astype(np.float32)

        img_tensor = self.img_transform(img)
        mask_tensor = torch.from_numpy(mask).long()
        return img_tensor, mask_tensor


class DiceLoss(nn.Module):
    """Dice loss for segmentation — better than cross-entropy for imbalanced masks."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]  # infected class probability
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets_f.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy for robust training."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]))  # upweight infected class

    def forward(self, logits, targets):
        return self.dice(logits, targets) + 0.5 * self.ce(logits, targets)


def compute_iou(pred_mask, true_mask):
    """Compute IoU (Intersection over Union) score."""
    pred = (pred_mask > 0.5).bool()
    true = (true_mask > 0).bool()
    intersection = (pred & true).sum().float()
    union = (pred | true).sum().float()
    return (intersection / (union + 1e-6)).item()


def train(
    epochs=50,
    batch_size=8,
    lr=1e-3,
    n_synthetic=2000,
    data_dir=None,
    save_path='models/gcn_weights.pth',
    device=None
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")

    # Dataset
    if data_dir and Path(data_dir).exists():
        train_ds = UltrasoundDataset(data_dir=str(Path(data_dir) / 'train'))
        val_ds   = UltrasoundDataset(data_dir=str(Path(data_dir) / 'val'))
    else:
        print("No data_dir found — using on-the-fly synthetic generation")
        train_ds = UltrasoundDataset(n_synthetic=n_synthetic)
        val_ds   = UltrasoundDataset(n_synthetic=max(200, n_synthetic // 10))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Model
    model = UltraSoundGCN(input_channels=3, num_classes=2).to(device)

    # Load existing weights to fine-tune if available
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"Loaded existing weights from {save_path} for fine-tuning")
        except Exception as e:
            print(f"Could not load existing weights ({e}) — training from scratch")

    criterion = CombinedLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_iou = 0.0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        # ── TRAIN ────────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ── VALIDATE ─────────────────────────────────────────────────────────────
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                pred = torch.softmax(logits, dim=1)[:, 1]
                for p, m in zip(pred, masks):
                    val_iou += compute_iou(p.cpu(), m.cpu())
        val_iou /= max(1, len(val_ds))

        scheduler.step()
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | Val IoU: {val_iou:.4f}"
              + (" ← BEST" if val_iou > best_val_iou else ""))

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete. Best Val IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {save_path}")
    return best_val_iou


if __name__ == "__main__":
    train(
        epochs=60,
        batch_size=8,
        lr=5e-4,
        n_synthetic=2000,
        data_dir="data/ultrasound_synthetic",
        save_path="models/gcn_weights.pth"
    )
