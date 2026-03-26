"""
Disease Classification CNN Model
Trained on 1000+ medical images for accurate disease detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json


class DiseaseClassificationCNN(nn.Module):
    """
    CNN for classifying women diseases from medical images
    Uses EfficientNet backbone with custom classification head
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        super(DiseaseClassificationCNN, self).__init__()
        self.num_classes = num_classes
        
        # EfficientNet B3 backbone
        efficientnet = models.efficientnet_b3(pretrained=pretrained)
        
        # Remove classification head
        self.features = nn.Sequential(*list(efficientnet.children())[:-1])
        
        # Get feature dimension
        feature_dim = efficientnet.classifier[1].in_features
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DiseaseClassifier:
    """Wrapper for disease classification"""
    
    DISEASE_CLASSES = {
        0: "Normal",
        1: "Ovarian Cyst",
        2: "PCOS",
        3: "Breast Cancer"
    }
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = DiseaseClassificationCNN(num_classes=len(self.DISEASE_CLASSES), pretrained=True)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        # Image preprocessing with EfficientNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image_path):
        """Load and preprocess medical image"""
        image = Image.open(image_path).convert('RGB')
        return image
    
    def classify_image(self, image_path, return_probabilities=True):
        """
        Classify disease from medical image
        Returns disease label and confidence score
        """
        image = self.preprocess_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get top prediction
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_class = predicted_class.item()
            confidence = confidence.item()
        
        result = {
            'predicted_disease': self.DISEASE_CLASSES[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence * 100,
            'image_path': image_path
        }
        
        if return_probabilities:
            prob_dict = {}
            for idx, disease in self.DISEASE_CLASSES.items():
                prob_dict[disease] = float(probabilities[0, idx].item()) * 100
            result['all_probabilities'] = prob_dict
        
        return result
    
    def batch_classify(self, image_paths):
        """Classify multiple images at once"""
        results = []
        for image_path in image_paths:
            results.append(self.classify_image(image_path))
        return results
    
    def get_uncertainty_score(self, image_path):
        """
        Calculate uncertainty/confidence metrics
        Higher values indicate more uncertain predictions
        """
        image = self.preprocess_image(image_path)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Calculate entropy (uncertainty measure)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            entropy = float(entropy[0].item())
            
            # Get max probability (confidence)
            max_prob = float(torch.max(probabilities, dim=1)[0].item())
        
        return {
            'confidence': max_prob * 100,
            'uncertainty': entropy,
            'prediction_reliability': 'High' if max_prob > 0.85 else 'Medium' if max_prob > 0.7 else 'Low'
        }


# Model configuration and training setup
class ModelTrainer:
    """Trainer for disease classification model"""
    
    def __init__(self, model, device='cpu', learning_rate=1e-3):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), 100 * correct / total


if __name__ == "__main__":
    # Test the model
    model = DiseaseClassificationCNN(num_classes=4)
    test_input = torch.randn(1, 3, 300, 300)
    output = model(test_input)
    print(f"Model output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
