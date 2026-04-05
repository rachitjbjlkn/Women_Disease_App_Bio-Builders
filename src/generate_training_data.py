"""
Synthetic Ultrasound Training Data Generator
Generates realistic ultrasound-like images with labeled cyst masks.
"""

import numpy as np
import cv2
import os
from pathlib import Path


def generate_speckle_noise(shape, intensity=0.4):
    """Generate multiplicative speckle noise characteristic of ultrasound."""
    noise = np.random.rayleigh(scale=intensity, size=shape)
    return np.clip(noise, 0, 1)


def generate_ultrasound_background(h, w):
    """Generate realistic ultrasound tissue background with speckle (V16)."""
    # Base gradient (brighter in center, darker at edges - typical US)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    base = 0.40 + 0.30 * np.exp(-(xx**2 + yy**2) * 2.0)

    # Add multiscale speckle noise
    speckle1 = generate_speckle_noise((h, w), intensity=0.30)
    speckle2 = generate_speckle_noise((h, w), intensity=0.15)
    speckle2 = cv2.GaussianBlur(speckle2, (7, 7), 2)
    base = base * (speckle1 + speckle2)

    # Add some tissue-like texture via blurred random noise
    texture = np.random.rand(h, w) * 0.4
    texture = cv2.GaussianBlur(texture.astype(np.float32), (21, 21), 8)
    base = base + texture * 0.25

    return np.clip(base, 0, 1)


def draw_cyst(canvas, mask, cx, cy, rx, ry, angle=0, cyst_intensity=0.06):
    """Draw a single hypoechoic cyst with realistic fuzzy boundaries (V16)."""
    h, w = canvas.shape
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

    # Fuzzy boundary effect
    fuzzy_mask = cv2.GaussianBlur(ellipse_mask.astype(np.float32), (15, 15), 5) / 255.0
    
    # Interior: very dark with minor internal echoes
    interior_noise = np.random.rand(h, w) * 0.08
    target_val = cyst_intensity + interior_noise
    
    # Blending the cyst into the canvas
    canvas = canvas * (1.0 - fuzzy_mask) + target_val * fuzzy_mask

    # Subtle bright rim (posterior acoustic enhancement - often only at bottom)
    rim_mask = np.zeros((h, w), dtype=np.uint8)
    # Enhancement is usually deeper than the cyst
    cv2.ellipse(rim_mask, (cx, cy + ry//4), (rx + 6, ry + 2), angle, 0, 180, 255, 4)
    rim_fuzzy = cv2.GaussianBlur(rim_mask.astype(np.float32), (11, 11), 3) / 255.0
    canvas = np.clip(canvas + rim_fuzzy * 0.3, 0, 1)

    # Update ground truth mask
    mask[ellipse_mask > 127] = 1
    return canvas, mask


def generate_scan_cone(h, w):
    """Generate an ultrasound scan cone mask."""
    cone = np.zeros((h, w), dtype=np.uint8)
    apex_x, apex_y = w // 2, -h // 4
    left_angle = 200
    right_angle = 340
    radius = int(h * 1.5)
    cv2.ellipse(cone, (apex_x, apex_y), (radius, radius), 0, left_angle, right_angle, 255, -1)
    return cone


def generate_sample(h=512, w=512, mode='random'):
    """
    Generate one synthetic ultrasound image + ground truth mask.
    mode: 'normal', 'pcos', 'cyst', or 'random'
    """
    if mode == 'random':
        mode = np.random.choice(['normal', 'pcos', 'cyst'], p=[0.25, 0.35, 0.40])

    background = generate_ultrasound_background(h, w)
    mask = np.zeros((h, w), dtype=np.uint8)
    cone = generate_scan_cone(h, w)

    # Apply black outside cone
    background[cone == 0] = 0.0

    if mode == 'normal':
        # No cysts - normal ovary
        pass

    elif mode == 'pcos':
        # Many small peripheral follicles (PCOS pattern)
        n_follicles = np.random.randint(8, 18)
        margin = int(min(h, w) * 0.30)
        for _ in range(n_follicles):
            # Place follicles near edge of ovary (peripheral)
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0.25, 0.38) * min(h, w)
            cx = int(w // 2 + dist * np.cos(angle))
            cy = int(h // 2 + dist * np.sin(angle))
            cx = np.clip(cx, margin, w - margin)
            cy = np.clip(cy, margin, h - margin)
            if cone[cy, cx] > 0:
                r = np.random.randint(8, 20)
                background, mask = draw_cyst(background, mask, cx, cy, r, r,
                                             cyst_intensity=0.04)

    elif mode == 'cyst':
        # 1-2 large dominant cysts
        n_cysts = np.random.randint(1, 3)
        for _ in range(n_cysts):
            margin = int(min(h, w) * 0.20)
            cx = np.random.randint(margin, w - margin)
            cy = np.random.randint(margin, h - margin)
            if cone[cy, cx] > 0:
                rx = np.random.randint(60, int(min(h, w) * 0.35))
                ry = int(rx * np.random.uniform(0.7, 1.3))
                angle = np.random.randint(0, 180)
                cx = np.clip(cx, rx + 10, w - rx - 10)
                cy = np.clip(cy, ry + 10, h - ry - 10)
                background, mask = draw_cyst(background, mask, cx, cy, rx, ry,
                                             angle, cyst_intensity=0.03)

    # Add caliper/measurement lines (common artefact in real images)
    if np.random.random() > 0.6 and np.any(mask > 0):
        ys, xs = np.where(mask > 0)
        if len(xs) > 40:
            cx_m = int(np.mean(xs))
            cy_m = int(np.mean(ys))
            # Crosshairs
            cv2.line(background, (cx_m - 50, cy_m), (cx_m + 50, cy_m), 0.9, 1)
            cv2.line(background, (cx_m, cy_m - 50), (cx_m, cy_m + 50), 0.9, 1)
            # Text-like labels (Measurement numbers)
            cv2.putText(background, f"{np.random.uniform(2.0, 5.0):.1f}cm", 
                        (cx_m + 10, cy_m - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0.8, 1)

    # Add Vertical Shadowing (V16 Addition)
    if np.random.random() > 0.7:
        n_shadows = np.random.randint(1, 4)
        for _ in range(n_shadows):
            sx = np.random.randint(50, w-50)
            sw = np.random.randint(4, 15)
            background[:, sx:sx+sw] *= np.random.uniform(0.4, 0.7)

    # Convert to uint8
    img_uint8 = (np.clip(background, 0, 1) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    return img_rgb, mask, mode


def create_dataset(output_dir, n_samples=2000, split=(0.8, 0.1, 0.1)):
    """Generate full dataset with train/val/test splits."""
    output_dir = Path(output_dir)
    splits = ['train', 'val', 'test']
    counts = [int(n_samples * s) for s in split]
    counts[-1] = n_samples - sum(counts[:-1])

    for split_name, count in zip(splits, counts):
        img_dir = output_dir / split_name / 'images'
        mask_dir = output_dir / split_name / 'masks'
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating {count} {split_name} samples...")
        for i in range(count):
            img, mask, mode = generate_sample()
            cv2.imwrite(str(img_dir / f"{i:04d}_{mode}.png"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_dir / f"{i:04d}_{mode}.png"),
                        (mask * 255).astype(np.uint8))
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{count} done")

    print(f"Dataset created at: {output_dir}")
    print(f"  Train: {counts[0]}, Val: {counts[1]}, Test: {counts[2]}")


if __name__ == "__main__":
    create_dataset("data/ultrasound_synthetic", n_samples=2000)
