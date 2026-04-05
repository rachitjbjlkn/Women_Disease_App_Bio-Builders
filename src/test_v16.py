import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())
from src.gcn_segmentation import UltrasoundSegmentation
from src.generate_training_data import generate_sample

def verify_v16():
    print("Testing Clinical V16 Segmentation...")
    os.makedirs('tests/v16_results', exist_ok=True)
    
    seg = UltrasoundSegmentation()
    
    # Test cases: normal, pcos, cyst
    modes = ['normal', 'pcos', 'cyst']
    
    for mode in modes:
        print(f"\n--- Testing Mode: {mode.upper()} ---")
        img_rgb, mask, label = generate_sample(mode=mode)
        
        # Save original and ground truth
        img_path = f"tests/v16_results/{mode}_original.png"
        cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        # Run segmentation
        res = seg.segment_image(img_path)
        pct = res['predicted_infected_area_percentage']
        print(f"Detected Infected Area Pct: {pct:.2f}%")
        
        # Save result overlay
        annotated, _ = seg.mark_infected_area(img_path, f"tests/v16_results/{mode}_detected.png")
        
        # Simple heuristic check
        if mode == 'normal' and pct > 5.0:
            print("WARNING: High false positive in normal image.")
        elif mode == 'cyst' and pct < 2.0:
            print("WARNING: Failed to detect large cyst.")
        elif mode == 'pcos' and pct < 0.5:
             print("WARNING: Failed to detect PCOS follicles.")
        else:
            print("PASS: Result within expected clinical range.")

if __name__ == "__main__":
    verify_v16()
