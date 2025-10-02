# gnn_trainer.py
"""
Training pipeline for GNN-based code detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from pathlib import Path
import logging
import json
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import wandb

logger = logging.getLogger(__name__)


class GNNTrainer:
    """
    Trainer class for GNN models
    """
    
    def __init__(
        self,
        model: nn.Module,
        data_module,
        config: Dict,
        use_wandb: bool = False
    ):
        """
        Initialize trainer
        
        Args:
            model: GNN model
            data_module: Data module with dataloaders
            config: Training configuration
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.model = model
        self.data_module = data_module
        self.config = config
        self.use_wandb = use_wandb
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")
        
        # Loss function with class weights for imbalanced data
        if hasattr(data_module, 'class_weights'):
            self.criterion = nn.CrossEntropyLoss(weight=data_module.class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="code-detection-gnn",
                config=config,
                name=f"{config['model_config']['gnn_type']}_{config['task']}"
            )
            wandb.watch(model)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam')
        lr = opt_config.get('lr', 0.001)
        weight_decay = opt_config.get('weight_decay', 1e-4)
        
        if opt_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif opt_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            return None
    
    def train_epoch(self, dataloader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        batch_pos_probs = []
        grad_norms = []

        with tqdm(dataloader, desc='Training') as pbar:
            for batch in pbar:
                batch = batch.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                out = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    batch.batch
                )

                # Prepare target and compute loss
                target = batch.y.view(-1).long()
                loss = self.criterion(out, target)

                # Backward pass
                loss.backward()

                # Compute gradient norm before clipping
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)

                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )

                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                probs = F.softmax(out, dim=1)
                # save mean positive class prob for binary problems
                try:
                    if probs.size(1) >= 2:
                        batch_pos_probs.append(probs[:, 1].mean().item())
                except Exception:
                    pass

                preds = out.argmax(dim=1).cpu().numpy()
                labels = batch.y.squeeze().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'grad_norm': total_norm})
        
        # Convert to numpy arrays for stable metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        # Debug: prediction distribution and mean positive probability (if binary)
        try:
            unique, counts = np.unique(all_preds, return_counts=True)
            pred_dist = dict(zip(unique.tolist(), counts.tolist()))
        except Exception:
            pred_dist = {}

        mean_pos_prob = None
        try:
            if len(batch_pos_probs) > 0:
                mean_pos_prob = float(np.mean(batch_pos_probs))
        except Exception:
            mean_pos_prob = None

        logger.debug(f"Train preds distribution: {pred_dist} | mean_pos_prob: {mean_pos_prob}")
        if self.use_wandb:
            logd = {'train/pred_dist': pred_dist, 'train/mean_pos_prob': mean_pos_prob}
            logd['train/mean_grad_norm'] = float(np.mean(grad_norms)) if len(grad_norms) > 0 else None
            # wandb.log(logd)

        # Warn if model collapsed to single-class predictions
        if len(pred_dist) == 1:
            logger.warning(f"Model predictions collapsed to a single class during training: {pred_dist}")
        
        return avg_loss, accuracy
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validate model
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                batch = batch.to(self.device)
                
                # Forward pass
                out = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    batch.batch
                )
                
                # Prepare target and compute loss
                target = batch.y.view(-1).long()
                loss = self.criterion(out, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1).cpu().numpy()
                labels = batch.y.squeeze().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.append(probs.cpu().numpy())
        
        # Convert to numpy arrays for stable metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # ROC-AUC for binary classification
        roc_auc = None
        if len(np.unique(all_labels)) == 2:
            try:
                all_probs = np.vstack(all_probs)
                roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
            except:
                pass

        # Debug: show prediction distribution and mean positive probability
        try:
            unique, counts = np.unique(all_preds, return_counts=True)
            pred_dist = dict(zip(unique.tolist(), counts.tolist()))
        except Exception:
            pred_dist = {}

        mean_pos_prob = None
        try:
            if len(np.vstack(all_probs).shape) == 2:
                mean_pos_prob = float(np.vstack(all_probs)[:, 1].mean())
        except Exception:
            mean_pos_prob = None

        logger.info(f"Val preds distribution: {pred_dist} | mean_pos_prob: {mean_pos_prob}")
        if self.use_wandb:
            log_debug = {'val/pred_dist': pred_dist, 'val/mean_pos_prob': mean_pos_prob}
            if roc_auc is not None:
                log_debug['val/roc_auc'] = roc_auc
            wandb.log(log_debug)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'roc_auc': roc_auc
        }

        return metrics
    
    def train(self, num_epochs: Optional[int] = None, save_dir: str = './checkpoints'):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        if num_epochs is None:
            num_epochs = self.config.get('epochs', 100)
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        train_loader = self.data_module.get_train_dataloader()
        val_loader = self.data_module.get_val_dataloader()
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = None
            if val_loader:
                val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['accuracy'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['lr'].append(current_lr)
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
                self.history['val_precision'].append(val_metrics['precision'])
                self.history['val_recall'].append(val_metrics['recall'])
            
            epoch_time = time.time() - start_time
            
            # Log metrics
            log_msg = (
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"LR: {current_lr:.6f}"
            )
            
            if val_metrics:
                log_msg += (
                    f" - Val Loss: {val_metrics['loss']:.4f} - "
                    f"Val Acc: {val_metrics['accuracy']:.4f} - "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
            
            logger.info(log_msg)
            
            # Wandb logging
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'train/lr': current_lr
                }
                if val_metrics:
                    log_dict.update({
                        'val/loss': val_metrics['loss'],
                        'val/accuracy': val_metrics['accuracy'],
                        'val/f1': val_metrics['f1'],
                        'val/precision': val_metrics['precision'],
                        'val/recall': val_metrics['recall']
                    })
                    if val_metrics['roc_auc']:
                        log_dict['val/roc_auc'] = val_metrics['roc_auc']
                
                wandb.log(log_dict)
            
            # Save best model
            if val_metrics:
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.best_val_acc = val_metrics['accuracy']
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    
                    # Save checkpoint
                    self.save_checkpoint(
                        save_dir / 'best_model.pt',
                        epoch + 1,
                        val_metrics
                    )
                    logger.info(f"New best model saved! F1: {self.best_val_f1:.4f}")
                else:
                    self.patience_counter += 1
                
                # Early stopping
                patience = self.config.get('patience', 20)
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(
                    save_dir / f'checkpoint_epoch_{epoch+1}.pt',
                    epoch + 1,
                    val_metrics
                )
        
        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pt', num_epochs, val_metrics)
        
        # Save training history
        self.save_history(save_dir / 'training_history.json')
        
        logger.info("Training complete!")
        logger.info(f"Best epoch: {self.best_epoch}")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics:
            checkpoint['metrics'] = {k: v for k, v in metrics.items() 
                                    if k != 'confusion_matrix'}
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.best_val_f1 = checkpoint.get('best_val_f1', 0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        
        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch']}")
    
    def save_history(self, path: Path):
        """Save training history to JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {path}")
    
    def evaluate(self, test_loader=None) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test dataloader (uses data_module's if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if test_loader is None:
            test_loader = self.data_module.get_test_dataloader()
        
        if test_loader is None:
            logger.warning("No test data available")
            return {}
        
        logger.info("Evaluating model on test set...")
        metrics = self.validate(test_loader)
        
        # Log results
        logger.info("Test Results:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        if self.use_wandb:
            wandb.finish()
        
        return metrics
    
    def predict(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a dataset
        
        Args:
            dataloader: DataLoader to predict on
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting'):
                batch = batch.to(self.device)
                
                out = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    batch.batch
                )
                
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_preds)
        probabilities = np.vstack(all_probs)
        
        return predictions, probabilities