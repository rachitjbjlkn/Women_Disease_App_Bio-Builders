"""
Complete training pipeline for disease classification model
Handles dataset loading, model training, and evaluation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from src.disease_classifier import DiseaseClassificationCNN, ModelTrainer
from src.dataset_builder import DatasetBuilder, MedicalImageAugmentation


class TrainingPipeline:
    """Complete training pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def prepare_data(self, data_directory, batch_size=32, num_workers=0):
        """Prepare dataset and dataloaders"""
        print(f"\n{'='*60}")
        print("DATASET PREPARATION")
        print(f"{'='*60}")
        
        # Load dataset
        builder = DatasetBuilder(data_directory)
        total = builder.load_dataset()
        
        if total < 100:
            print(f"⚠ Warning: Dataset has only {total} images. Minimum 1000 recommended.")
        
        # Print class distribution
        distribution = builder.get_class_distribution()
        print("\nClass Distribution:")
        for class_name, count in distribution.items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {class_name}: {count:4d} images ({percentage:5.1f}%)")
        
        # Create dataloaders
        dataloaders = builder.create_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            test_size=0.15,
            val_size=0.15
        )
        
        print(f"\nDataLoaders created successfully!")
        return dataloaders
    
    def setup_model(self):
        """Initialize model and trainer"""
        print(f"\n{'='*60}")
        print("MODEL SETUP")
        print(f"{'='*60}")
        
        self.model = DiseaseClassificationCNN(
            num_classes=len(self.config.get('num_classes', 4)),
            pretrained=True
        )
        
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: EfficientNet B3 with custom head")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        
        # Initialize trainer
        self.trainer = ModelTrainer(
            self.model,
            device=self.device,
            learning_rate=float(self.config.get('learning_rate', 1e-3))
        )
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
        
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.trainer.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.trainer.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.trainer.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': total_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader, epoch):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(val_loader, desc=f'Epoch {epoch} - Validation')
        
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.trainer.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': total_loss / (progress_bar.n + 1),
                    'acc': 100 * correct / total
                })
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def train(self, dataloaders, num_epochs, save_dir='models'):
        """Complete training loop"""
        print(f"\n{'='*60}")
        print("TRAINING")
        print(f"{'='*60}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        best_val_acc = 0
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 5)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch+1)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, epoch+1)
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }
                
                model_path = os.path.join(save_dir, 'best_model.pt')
                torch.save(checkpoint, model_path)
                print(f"  ✓ Best model saved: {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Learning rate scheduler
            self.trainer.scheduler.step(val_loss)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        print(f"\n{'='*60}")
        print("EVALUATION ON TEST SET")
        print(f"{'='*60}")
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def save_training_report(self, save_path='training_report.json'):
        """Save training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'history': self.history,
            'device': str(self.device)
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved: {save_path}")


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train disease classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'num_classes': 4,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'early_stopping_patience': 5
    }
    
    # Create pipeline
    pipeline = TrainingPipeline(config)
    
    # Prepare data
    dataloaders = pipeline.prepare_data(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Setup model
    pipeline.setup_model()
    
    # Train
    history = pipeline.train(dataloaders, args.epochs, args.save_dir)
    
    # Evaluate
    results = pipeline.evaluate(dataloaders['test'])
    
    # Save report
    pipeline.save_training_report(os.path.join(args.save_dir, 'training_report.json'))
    
    print("\nTraining pipeline completed successfully!")


if __name__ == '__main__':
    main()
