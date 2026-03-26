"""
Graph Convolutional Network (GCN) for Medical Image Segmentation
Specialized for ultrasound image analysis and detecting infected areas
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms


class GCNBlock(nn.Module):
    """Graph Convolutional Network Block"""
    def __init__(self, in_channels, out_channels, activation=True):
        super(GCNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class UltraSoundGCN(nn.Module):
    """
    GCN-based segmentation model for ultrasound images
    Detects and segments infected areas
    """
    def __init__(self, input_channels=3, num_classes=2):
        super(UltraSoundGCN, self).__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.enc1 = GCNBlock(input_channels, 32)
        self.enc2 = GCNBlock(32, 64)
        self.enc3 = GCNBlock(64, 128)
        self.enc4 = GCNBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = GCNBlock(256, 512)
        
        # Decoder
        self.dec4 = GCNBlock(512 + 256, 256)
        self.dec3 = GCNBlock(256 + 128, 128)
        self.dec2 = GCNBlock(128 + 64, 64)
        self.dec1 = GCNBlock(64 + 32, 32)
        
        # Output layer
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upsample(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upsample(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.final_conv(x)
        return x


class UltrasoundSegmentation:
    """Wrapper for ultrasound image segmentation"""
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = UltraSoundGCN(input_channels=3, num_classes=2)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        
        self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess_image(self, image_path, target_size=(512, 512)):
        """Load and preprocess ultrasound image"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size, Image.BILINEAR)
        return image, np.array(image)
    
    def segment_image(self, image_path, confidence_threshold=0.5):
        """
        Segment ultrasound image to detect infected areas
        Returns segmentation mask and confidence scores
        """
        image, image_np = self.preprocess_image(image_path)
        
        # Prepare input tensor
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get segmentation mask
            mask = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()
            confidence = probabilities[0, 1].cpu().numpy()
        
        # Apply confidence threshold
        binary_mask = (confidence > confidence_threshold).astype(np.uint8) * 255
        
        return {
            'original_image': image_np,
            'segmentation_mask': mask,
            'confidence_map': confidence,
            'binary_mask': binary_mask,
            'predicted_infected_area_percentage': (binary_mask.sum() / binary_mask.size) * 100
        }
    
    def mark_infected_area(self, image_path, output_path=None, color=(0, 255, 0), confidence_threshold=0.5):
        """
        Mark infected areas on the original ultrasound image
        Returns annotated image
        """
        segmentation = self.segment_image(image_path, confidence_threshold)
        
        image = segmentation['original_image'].copy()
        binary_mask = segmentation['binary_mask']
        
        # Find contours of infected areas
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on image
        cv2.drawContours(image, contours, -1, color, 3)
        
        # Create overlay for better visualization
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, color, -1)
        
        # Blend original and overlay
        annotated_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        return annotated_image, segmentation


# For initialization without model path
import os
if __name__ == "__main__":
    # Test the model
    model = UltraSoundGCN()
    test_input = torch.randn(1, 3, 512, 512)
    output = model(test_input)
    print(f"Model output shape: {output.shape}")
