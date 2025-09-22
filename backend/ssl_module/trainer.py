import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import time
import datetime
import shutil
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import io
from PIL import Image

from .utils import SSLDataset, save_checkpoint, get_device
from .augmentation import get_weak_strong_augment, get_augmentation

class SSLTrainer:
    """Semi-Supervised Learning Trainer for the species classifier."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_classes: int = 2,
                 device: str = None,
                 output_dir: str = 'output',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 max_epochs: int = 50,
                 patience: int = 10,
                 pseudo_label_threshold: float = 0.9,
                 consistency_weight: float = 0.5,
                 use_ema: bool = True,
                 ema_decay: float = 0.999,
                 warmup_epochs: int = 5,
                 grad_accum_steps: int = 1,
                 max_grad_norm: float = 1.0,
                 use_amp: bool = True,
                 use_strong_aug: bool = True,
                 label_smoothing: float = 0.1):
        """
        Initialize the SSL trainer.
        
        Args:
            model: PyTorch model for classification
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_classes: Number of output classes
            device: Device to run training on ('cuda', 'mps', or 'cpu')
            output_dir: Directory to save checkpoints and logs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            max_epochs: Maximum number of training epochs
            patience: Number of epochs to wait before early stopping
            pseudo_label_threshold: Confidence threshold for pseudo-labels
            consistency_weight: Weight for consistency loss
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device or get_device()
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.pseudo_label_threshold = pseudo_label_threshold
        self.consistency_weight = consistency_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and ('cuda' in str(self.device))
        self.use_strong_aug = use_strong_aug
        self.label_smoothing = label_smoothing
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        self.supervised_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.consistency_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Mixed precision training
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=1e-8
        )
        
        # Learning rate warmup + cosine annealing
        num_warmup_steps = warmup_epochs * len(train_loader) // self.grad_accum_steps
        num_training_steps = max_epochs * len(train_loader) // self.grad_accum_steps
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1
        )
        
        # EMA model for better teacher predictions
        if self.use_ema:
            self.ema_model = self._get_ema_model()
        
        # TensorBoard logging
        log_dir = os.path.join(output_dir, 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Track best metrics
        self.best_metrics = {
            'val_loss': float('inf'),
            'val_accuracy': 0.0,
            'epoch': 0
        }
        
        # Training state
        self.best_accuracy = 0.0
        self.epochs_since_improvement = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def _get_ema_model(self):
        """Create an EMA copy of the model for better teacher predictions."""
        ema_model = type(self.model)(num_classes=self.num_classes).to(self.device)
        ema_model.load_state_dict(self.model.state_dict())
        for param in ema_model.parameters():
            param.detach_()
        return ema_model
    
    def _update_ema_model(self):
        """Update the EMA model using exponential moving average."""
        if not hasattr(self, 'ema_model'):
            return
            
        model_params = dict(self.model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())
        
        for name, param in model_params.items():
            ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def _log_confusion_matrix(self, targets, preds, class_names, epoch, tag):
        """Log confusion matrix to TensorBoard."""
        cm = confusion_matrix(targets, preds)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]}\n({cm_norm[i,j]:.2f})",
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to PyTorch tensor and log to TensorBoard
        image = Image.open(buf)
        image = torchvision.transforms.ToTensor()(image)
        self.writer.add_image(f'{tag}/confusion_matrix', image, epoch)
        
        # Log classification report
        report = classification_report(
            targets, preds, target_names=class_names, output_dict=True
        )
        
        # Log metrics for each class
        for class_name in class_names:
            self.writer.add_scalars(
                f'{tag}/class_metrics/{class_name}',
                {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                },
                epoch
            )
        
        # Log overall metrics
        self.writer.add_scalar(f'{tag}/accuracy', report['accuracy'], epoch)
        self.writer.add_scalar(
            f'{tag}/weighted_avg/precision', 
            report['weighted avg']['precision'], 
            epoch
        )
        self.writer.add_scalar(
            f'{tag}/weighted_avg/recall', 
            report['weighted avg']['recall'], 
            epoch
        )
        self.writer.add_scalar(
            f'{tag}/weighted_avg/f1-score', 
            report['weighted avg']['f1-score'], 
            epoch
        )
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_sup_loss = 0.0
        total_cons_loss = 0.0
        total_samples = 0
        correct = 0
        
        # Initialize gradient accumulation
        self.optimizer.zero_grad()
        
        # Get class names from dataset if available
        class_names = getattr(self.train_loader.dataset, 'classes', 
                            [str(i) for i in range(self.num_classes)])
        
        progress = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f'Epoch {epoch + 1}/{self.max_epochs}',
            leave=True
        )
        
        for batch_idx, (batch) in progress:
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            unlabeled_images = batch['unlabeled_image'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                # Forward pass for labeled data
                outputs = self.model(images)
                sup_loss = self.supervised_loss(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                total_samples += labels.size(0)
                correct += batch_correct
                
                # Forward pass for unlabeled data (consistency loss)
                cons_loss = torch.tensor(0.0, device=self.device)
                if unlabeled_images.size(0) > 0:
                    # Use EMA model for teacher predictions if available
                    teacher_model = self.ema_model if hasattr(self, 'ema_model') else self.model
                    teacher_model.eval()
                    
                    with torch.no_grad():
                        # Get teacher predictions
                        teacher_outputs = teacher_model(unlabeled_images)
                        teacher_probs = F.softmax(teacher_outputs / 0.5, dim=1)  # Temperature scaling
                        
                        # Apply confidence threshold for pseudo-labels
                        max_probs, pseudo_labels = torch.max(teacher_probs, dim=1)
                        mask = max_probs > self.pseudo_label_threshold
                        
                        if mask.any():
                            # Only use high-confidence pseudo-labels
                            pseudo_labels = pseudo_labels[mask]
                            unlabeled_images = unlabeled_images[mask]
                    
                    if mask.any():
                        # Student predictions for unlabeled data with strong augmentation
                        if self.use_strong_aug:
                            unlabeled_images = get_strong_augment()(unlabeled_images)
                        
                        student_outputs = self.model(unlabeled_images)
                        student_probs = F.log_softmax(student_outputs / 0.5, dim=1)  # Temperature scaling
                        
                        # Consistency loss (KL divergence between teacher and student)
                        cons_loss = self.consistency_loss(
                            student_probs,
                            teacher_probs[mask].detach()
                        )
                
                # Combined loss with consistency weighting
                current_cons_weight = self.consistency_weight * min(epoch / 10.0, 1.0)  # Ramp up
                loss = sup_loss + current_cons_weight * cons_loss
                
                # Scale loss for mixed precision training
                loss = loss / self.grad_accum_steps
            
            # Backward pass with gradient scaling for mixed precision
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update EMA model
                if self.use_ema:
                    self._update_ema_model()
            
            # Update metrics
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size * self.grad_accum_steps
            total_sup_loss += sup_loss.item() * batch_size
            total_cons_loss += cons_loss.item() * batch_size if unlabeled_images.size(0) > 0 else 0.0
            
            # Log batch metrics
            global_step = epoch * len(self.train_loader) + batch_idx
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item() * self.grad_accum_steps, global_step)
                self.writer.add_scalar('train/sup_loss', sup_loss.item(), global_step)
                self.writer.add_scalar('train/cons_loss', cons_loss.item(), global_step)
                self.writer.add_scalar('train/cons_weight', current_cons_weight, global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)
                self.writer.add_scalar('train/acc', 100. * batch_correct / labels.size(0), global_step)
                
                # Log gradient norms
                if (batch_idx + 1) % (self.grad_accum_steps * 10) == 0:
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    self.writer.add_scalar('train/grad_norm', total_norm, global_step)
            
            # Update progress bar
            progress.set_postfix({
                'loss': f"{loss.item() * self.grad_accum_steps:.4f}",
                'sup': f"{sup_loss.item():.4f}",
                'cons': f"{cons_loss.item():.4f}",
                'acc': f"{100. * correct / total_samples:.2f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader.dataset)
        epoch_sup_loss = total_sup_loss / len(self.train_loader.dataset)
        epoch_cons_loss = total_cons_loss / len(self.train_loader.dataset)
        
        return {
            'train_loss': epoch_loss,
            'sup_loss': epoch_sup_loss,
            'cons_loss': epoch_cons_loss
        }
    
    @torch.no_grad()
    def validate(self, epoch: int = 0) -> Dict[str, float]:
        """Validate the model on the validation set."""
        self.model.eval()
        if hasattr(self, 'ema_model'):
            self.ema_model.eval()
        
        metrics = {}
        
        # Get class names from dataset if available
        class_names = getattr(self.val_loader.dataset, 'classes', 
                            [str(i) for i in range(self.num_classes)])
        
        # Validate both regular and EMA models
        for model_name in (['model', 'ema_model'] if hasattr(self, 'ema_model') else ['model']):
            total_loss = 0.0
            correct = 0
            total = 0
            all_labels = []
            all_preds = []
            all_probs = []
            
            model = getattr(self, model_name)
            
            for batch in tqdm(self.val_loader, desc=f'Validating {model_name}'):
                images = batch['image'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp):
                    outputs = model(images)
                    loss = self.supervised_loss(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                probs = F.softmax(outputs, dim=1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item() * images.size(0)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
            # Calculate metrics
            val_loss = total_loss / len(self.val_loader.dataset)
            val_accuracy = 100.0 * correct / total
            
            # Store metrics
            prefix = 'ema/' if model_name == 'ema_model' else ''
            metrics.update({
                f'{prefix}val_loss': val_loss,
                f'{prefix}val_accuracy': val_accuracy,
            })
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(f'val/{prefix}loss', val_loss, epoch)
                self.writer.add_scalar(f'val/{prefix}accuracy', val_accuracy, epoch)
                
                # Log confusion matrix and classification report
                if (epoch + 1) % 5 == 0:  # Log every 5 epochs to reduce overhead
                    self._log_confusion_matrix(
                        all_labels, all_preds, class_names, epoch, f'val/{prefix}'
                    )
                
                # Log sample predictions with images
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    # Log a batch of validation images with predictions
                    self._log_validation_images(
                        images, labels, predicted, probs, class_names, epoch, prefix
                    )
                
                # Log ROC curve and PR curve
                if len(class_names) == 2:  # Binary classification
                    self._log_binary_metrics(
                        all_labels, all_probs, epoch, prefix
                    )
                
                # Log per-class metrics
                self._log_per_class_metrics(
                    all_labels, all_preds, class_names, epoch, prefix
                )
        
        return metrics
    
    def _log_validation_images(self, images, labels, preds, probs, class_names, epoch, prefix):
        """Log a grid of validation images with predictions."""
        # Only log a small subset of images
        num_images = min(8, images.size(0))
        if num_images == 0:
            return
            
        # Create a grid of images with predictions
        grid = torchvision.utils.make_grid(images[:num_images], nrow=4, normalize=True, scale_each=True)
        self.writer.add_image(f'{prefix}val_samples', grid, epoch)
        
        # Add text predictions
        img_labels = []
        for i in range(num_images):
            true_label = class_names[labels[i].item()]
            pred_label = class_names[preds[i].item()]
            prob = probs[i].max().item()
            img_labels.append(f"True: {true_label}\nPred: {pred_label} ({prob:.2f})")
        
        # Log the text labels separately
        self.writer.add_text(
            f'{prefix}val_predictions',
            '  |  '.join(img_labels),
            epoch
        )
    
    def _log_binary_metrics(self, labels, probs, epoch, prefix):
        """Log binary classification metrics (ROC, PR curves)."""
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probs])
        roc_auc = auc(fpr, tpr)
        
        # Create figure
        fig = plt.figure(figsize=(10, 4))
        
        # Plot ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(labels, [p[1] for p in probs])
        pr_auc = auc(recall, precision)
        
        plt.subplot(1, 2, 2)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall Curve (AP={pr_auc:.2f})')
        
        # Save to TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        image = Image.open(buf)
        image = torchvision.transforms.ToTensor()(image)
        self.writer.add_image(f'{prefix}val_curves', image, epoch)
    
    def _log_per_class_metrics(self, labels, preds, class_names, epoch, prefix):
        """Log per-class metrics."""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=range(len(class_names))
        )
        
        for i, class_name in enumerate(class_names):
            self.writer.add_scalar(f'{prefix}val_metrics/{class_name}/precision', precision[i], epoch)
            self.writer.add_scalar(f'{prefix}val_metrics/{class_name}/recall', recall[i], epoch)
            self.writer.add_scalar(f'{prefix}val_metrics/{class_name}/f1', f1[i], epoch)
    
    def train(self) -> Dict[str, List[float]]:
        """Run the training loop."""
        start_time = time.time()
        
        # Log model architecture
        if self.writer is not None:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            try:
                self.writer.add_graph(self.model, dummy_input)
            except Exception as e:
                print(f"Failed to log model graph: {e}")
        
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            
            # Update best metrics
            current_val_acc = val_metrics.get('ema/val_accuracy', val_metrics['val_accuracy'])
            if current_val_acc > self.best_metrics['val_accuracy']:
                self.best_metrics.update({
                    'val_accuracy': current_val_acc,
                    'val_loss': val_metrics.get('ema/val_loss', val_metrics['val_loss']),
                    'epoch': epoch
                })
                self.epochs_since_improvement = 0
            else:
                self.epochs_since_improvement += 1
            
            # Print epoch summary
            print(f'\nEpoch {epoch + 1}/{self.max_epochs} - {epoch_time:.1f}s')
            print('-' * 80)
            print(f'Train Loss: {train_metrics["train_loss"]:.4f} ' \
                  f'(Sup: {train_metrics["sup_loss"]:.4f}, ' \
                  f'Cons: {train_metrics["cons_loss"]:.4f})')
            
            for prefix in (['', 'ema/'] if 'ema/val_accuracy' in val_metrics else ['']):
                model_name = 'EMA Model' if prefix else 'Model'
                print(f'{model_name} Val Loss: {val_metrics[prefix+"val_loss"]:.4f}, ' \
                      f'Val Acc: {val_metrics[prefix+"val_accuracy"]:.2f}%')
            
            print(f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}')
            
            # Save checkpoint
            is_best = current_val_acc == self.best_metrics['val_accuracy']
            
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_metrics': self.best_metrics,
                'config': {
                    'num_classes': self.num_classes,
                    'backbone': getattr(self.model, 'backbone_name', 'resnet18'),
                }
            }
            
            if hasattr(self, 'ema_model'):
                checkpoint['ema_state_dict'] = self.ema_model.state_dict()
            
            # Save latest checkpoint
            checkpoint_path = self.output_dir / 'checkpoint_latest.pth'
            torch.save(checkpoint, checkpoint_path)
            
            # Save best checkpoint
            if is_best:
                best_path = self.output_dir / 'model_best.pth'
                shutil.copy2(checkpoint_path, best_path)
                print(f'New best model saved with accuracy: {current_val_acc:.2f}%')
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                epoch_path = self.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                shutil.copy2(checkpoint_path, epoch_path)
            
            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('epoch/train_loss', train_metrics['train_loss'], epoch)
                self.writer.add_scalar('epoch/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('epoch/time', epoch_time, epoch)
            
            # Early stopping
            if self.epochs_since_improvement >= self.patience:
                print(f'\nEarly stopping after {self.patience} epochs without improvement')
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s')
        print(f'Best validation accuracy: {self.best_metrics["val_accuracy"]:.2f}% ' \
              f'at epoch {self.best_metrics["epoch"] + 1}')
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return self.best_metrics
        
        return self.history
    
    def save_model(self, path: str) -> None:
        """Save the trained model with additional metadata."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model state with additional metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch if hasattr(self, 'epoch') else 0,
            'best_metrics': self.best_metrics,
            'config': {
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'use_ema': self.use_ema,
                'ema_decay': self.ema_decay,
                'pseudo_label_threshold': self.pseudo_label_threshold,
                'consistency_weight': self.consistency_weight,
                'use_amp': self.use_amp,
                'use_strong_aug': self.use_strong_aug,
                'label_smoothing': self.label_smoothing,
            }
        }, path)
        
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        # Move optimizer state to the correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        
        print(f"Loaded model from {path} (epoch {self.epoch}, best val_acc: {self.best_metrics.get('val_accuracy', 0):.2f}%)")
    
    def save_history(self, path: str) -> None:
        """Save training history to a JSON file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Add additional metadata to history
        history = {
            'config': {
                'num_classes': self.num_classes,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'use_ema': self.use_ema,
                'ema_decay': self.ema_decay,
                'pseudo_label_threshold': self.pseudo_label_threshold,
                'consistency_weight': self.consistency_weight,
                'use_amp': self.use_amp,
                'use_strong_aug': self.use_strong_aug,
                'label_smoothing': self.label_smoothing,
            },
            'metrics': self.history,
            'best_metrics': self.best_metrics,
            'timestamp': datetime.datetime.now().isoformat(),
            'training_time_seconds': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
