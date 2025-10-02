# main.py
"""
Main module for AST-based code detection system
SemEval-2026 Task 13: Detecting Machine-Generated Code
"""

import argparse
import logging
from pathlib import Path
from data_loader import DataLoader
from ast_constructor import ASTConstructor
from graph_converter import GraphConverter
from feature_extractor import FeatureExtractor
import pickle
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import math
from gnn_model import create_model
from gnn_dataset import CodeASTDataModule, load_processed_data, CodeASTDataset
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from gnn_trainer import GNNTrainer
import warnings

# Silence deprecation warnings from torch_geometric about the old DataLoader
warnings.filterwarnings(
    "ignore",
    message=r".*'data.DataLoader' is deprecated.*",
    category=UserWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> dict:
    """Load training configuration"""
    default_config = {
        'task': 'subtask_a',
        'model_config': {
            'model_type': 'standard',  # or 'hierarchical'
            'gnn_type': 'gat',  # 'gcn', 'gat', 'custom'
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.3,
            'use_edge_features': True,
            'edge_feature_dim': 6,
            'pooling': 'mean'  # 'mean', 'max', 'add', 'attention', 'all'
        },
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'num_workers': 4,
            'augment_train': True,
            'patience': 20,
            'grad_clip': 1.0,
            'save_every': 10
        },
        'optimizer': {
            'type': 'adamw',  # 'adam', 'adamw', 'sgd'
            'lr': 0.001,
            'weight_decay': 1e-4,
            'betas': (0.9, 0.999)
        },
        'scheduler': {
            'type': 'cosine',  # 'cosine', 'reduce_on_plateau'
            'eta_min': 1e-6,
            'factor': 0.5,
            'patience': 5,
            'min_lr': 1e-6
        }
    }
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            user_config = json.load(f)
        # Merge configs
        default_config.update(user_config)
    
    return default_config


class ASTCodeDetector:
    """Main class for AST-based code detection pipeline"""
    
    def __init__(self, task='subtask_a', languages=None):
        self.task = task
        self.languages = [
        'python',
        'java',
        'cpp',  # Note: 'cpp' is the correct name in tree-sitter-languages
        'c',
        'javascript',
        'go',
        'php',
        'csharp'
    ]
        # self.data_loader = DataLoader(task=task)
        self.ast_constructor = ASTConstructor()
        self.graph_converter = GraphConverter()
        self.feature_extractor = FeatureExtractor()
        
    def process_dataset(self, data):
        """
        Process dataset and extract AST features
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            limit: Limit number of samples (for testing)
            save_path: Path to save processed data
        """
        # logger.info(f"Processing {split} dataset for {self.task}")
        
        # # Load data
        # data = self.data_loader.load_data(split=split, limit=limit)
        # logger.info(f"Loaded {len(data)} samples")
        
        processed_data = []
        failed_count = 0
        
        for idx, sample in enumerate(data):
            # if idx % 100 == 0:
            #     logger.info(f"Processing sample {idx}/{len(data)}")
            
            try:
                # Extract code and metadata
                code = sample['code']
                language = sample.get('language', 'python').lower()
                label = sample.get('label', -1)
                # print(sample)
                if language == 'c++':
                    language = 'cpp'
                if language == 'c#':
                    language = 'csharp'
                
                # Skip if language not supported
                if language not in self.languages:
                    print(f"Skipping unsupported language: {language}")
                    failed_count += 1
                    continue
                
                # Construct AST (now returns serializable dict)
                ast_dict = self.ast_constructor.parse_code(code, language)
                if ast_dict is None:
                    print("AST is None")
                    failed_count += 1
                    continue
                
                # Convert to graph representation
                graph_data = self.graph_converter.ast_to_graph(ast_dict, language)
                if graph_data is None:
                    print("Graph data is None")
                    failed_count += 1
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_features(
                    ast_dict, 
                    graph_data, 
                    code,
                    language
                )
                if features is None:
                    print("Features are None")
                    failed_count += 1
                    continue
                
                # Combine all data
                processed_sample = {
                    'id': idx,
                    'code': code,
                    'language': language,
                    'label': label,
                    'ast_tree': ast_dict,  # Now a serializable dict
                    'graph': graph_data,
                    'features': features,
                    'metadata': {
                        'generator': sample.get('generator', 'unknown'),
                        'domain': sample.get('domain', 'unknown')
                    }
                }
                # print(processed_sample)
                processed_data.append(processed_sample)
                
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {str(e)}")
                failed_count += 1
                continue
        
        # logger.info(f"Successfully processed {len(processed_data)} samples")
        # logger.info(f"Failed to process {failed_count} samples")
        
        return processed_data
    
    def save_processed_data(self, data, save_path):
        """Save processed data to disk"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for easy loading
        with open(save_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata as JSON for inspection
        metadata = {
            'total_samples': len(data),
            'languages': list(set(d['language'] for d in data)),
            'labels': list(set(d['label'] for d in data)),
            'task': self.task
        }
        
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved processed data to {save_path}")
    
    def load_processed_data(self, load_path):
        """Load previously processed data"""
        load_path = Path(load_path)
        
        with open(load_path.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded {len(data)} processed samples from {load_path}")
        return data

def main():
    parser = argparse.ArgumentParser(
        description='AST-based Code Detection for SemEval-2026 Task 13'
    )
    parser.add_argument(
        '--task', 
        type=str, 
        default='subtask_a',
        choices=['subtask_a', 'subtask_b', 'subtask_c'],
        help='Task to process'
    )
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=['python', 'java', 'cpp'],
        help='Programming languages to process'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of samples (for testing)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config.json',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./experiments',
        help='Output directory for checkpoints and logs'
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Use Weights & Biases for logging'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    parser.add_argument(
        '--eval_only',
        action='store_true',
        help='Only evaluate on test set (requires --resume)'
    )
    # parser.add_argument(
    #     '--save_path',
    #     type=str,
    #     default='./processed_data/ast_features',
    #     help='Path to save processed data'
    # )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ASTCodeDetector(
        task=args.task,
        languages=args.languages
    )
    data_loader = DataLoader(task=args.task)
    train_data = data_loader.load_data(split="train", limit=args.limit)
    # val_data = data_loader.load_data(split="validation", limit= args.limit*0.15 if args.limit else 10000 )
    train_data, val_data = train_test_split(
        train_data,
        test_size=0.01,
        random_state=42,
        stratify=[d['label'] for d in train_data]
    )   
    test_data = data_loader.load_data(split="test", limit=(args.limit*0.15) if args.limit else 10000)
    
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    config = load_config(args.config)
    config['task'] = args.task
    
    # Update config with command line args
    if args.use_wandb:
        config['use_wandb'] = True
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.task / f"gnn_{config['model_config']['gnn_type']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
        
        # --- Prepare validation and test sets (fully processed; these are small) ---
        logger.info("Processing validation and test splits (in-memory)")
        processed_val = detector.process_dataset(val_data)
        processed_test = detector.process_dataset(test_data)

        # Create PyG datasets and dataloaders for val/test
        val_loader = None
        test_loader = None
        if len(processed_val) > 0:
            val_dataset = CodeASTDataset(processed_val, augment=False, task=args.task)
            val_loader = PyGDataLoader(
                val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers'],
                pin_memory=True if torch.cuda.is_available() else False
            )

        if len(processed_test) > 0:
            test_dataset = CodeASTDataset(processed_test, augment=False, task=args.task)
            test_loader = PyGDataLoader(
                test_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training']['num_workers'],
                pin_memory=True if torch.cuda.is_available() else False
            )

        # Infer node feature dim from validation or a small processed train sample
        node_feature_dim = None
        if val_loader is not None and len(val_loader.dataset) > 0:
            node_feature_dim = val_loader.dataset[0].x.size(1)
        else:
            # Process a tiny train sample to infer dims
            sample_proc = detector.process_dataset(train_data[: min(10, len(train_data))])
            if len(sample_proc) > 0:
                sample_ds = CodeASTDataset(sample_proc, augment=False, task=args.task)
                node_feature_dim = sample_ds[0].x.size(1)

        if node_feature_dim is None:
            raise RuntimeError("Unable to infer node feature dimension from data")

        # Determine number of classes
        if args.task == 'subtask_a':
            num_classes = 2
        elif args.task == 'subtask_b':
            num_classes = 11
        elif args.task == 'subtask_c':
            num_classes = 4
        else:
            # fallback: infer from labels in val/test
            labels = [d['label'] for d in processed_val + processed_test]
            num_classes = len(set(labels)) if len(labels) > 0 else 2

        logger.info(f"Node feature dimension: {node_feature_dim}")
        logger.info(f"Number of classes: {num_classes}")

        # Update model config and create model
        config['model_config']['node_feature_dim'] = node_feature_dim
        config['model_config']['num_classes'] = num_classes
        model = create_model(config['model_config'])
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

        # Compute simple class weights from val+test (fallback) to handle imbalanced loss
        all_labels = [d['label'] for d in (processed_val + processed_test) if 'label' in d]
        if len(all_labels) == 0:
            # fallback uniform weights
            class_weights = torch.FloatTensor([1.0 for _ in range(num_classes)])
        else:
            import collections
            cnt = collections.Counter(all_labels)
            total = sum(cnt.values())
            # ensure ordering by label index
            weights = [total / (num_classes * cnt.get(i, 1)) for i in range(num_classes)]
            class_weights = torch.FloatTensor(weights)

        # Create a minimal data_module-like object that only supplies class_weights (used by GNNTrainer)
        class MinimalDataModule:
            def __init__(self, class_weights):
                self.class_weights = class_weights

        trainer_config = {
            **config['training'],
            'optimizer': config['optimizer'],
            'scheduler': config['scheduler'],
            'task': args.task,
            'model_config': config['model_config']
        }

        trainer = GNNTrainer(
            model=model,
            data_module=MinimalDataModule(class_weights),
            config=trainer_config,
            use_wandb=args.use_wandb
        )

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(Path(args.resume))

        # If eval only, just run evaluation on test set
        if args.eval_only:
            if not args.resume:
                raise ValueError("--eval_only requires --resume to specify a checkpoint")
            logger.info("Evaluation mode - running test evaluation only")
            test_metrics = trainer.evaluate(test_loader)

            # Save results
            with open(output_dir / 'test_results.json', 'w') as f:
                results = {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}
                results['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
                json.dump(results, f, indent=2)

            return

        # --- Streaming / chunked training over the raw train_data ---
        logger.info("Starting streaming chunked training...")
        num_epochs = config['training'].get('epochs', 100)
        batch_size = config['training'].get('batch_size', 32)
        num_workers = config['training'].get('num_workers', 4)
        augment_train = config['training'].get('augment_train', True)
        chunk_size = config['training'].get('chunk_size', batch_size * 16)
        save_dir = output_dir / 'checkpoints'
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            epoch_start = logger.handlers[0].formatter.converter if logger.handlers else None
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            # Shuffle raw train_data indices each epoch for robustness
            import random
            indices = list(range(len(train_data)))
            random.shuffle(indices)

            epoch_losses = []
            epoch_accs = []

            # Process train data in chunks (show chunk-level progress)
            num_chunks = math.ceil(len(indices) / chunk_size) if len(indices) > 0 else 0
            for start in tqdm(range(0, len(indices), chunk_size), desc='Chunks', total=num_chunks, unit='chunk', leave=True):
                idxs = indices[start:start + chunk_size]
                chunk_raw = [train_data[i] for i in idxs]
                # Convert chunk to processed form (AST -> graph -> features)
                chunk_processed = detector.process_dataset(chunk_raw)
                if len(chunk_processed) == 0:
                    continue

                # Create dataset and dataloader for this chunk
                chunk_dataset = CodeASTDataset(chunk_processed, augment=augment_train, task=args.task)
                chunk_loader = PyGDataLoader(
                    chunk_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True if torch.cuda.is_available() else False,
                    drop_last=True
                )

                # Train on this chunk (one pass)
                try:
                    loss, acc = trainer.train_epoch(chunk_loader)
                    epoch_losses.append((loss, len(chunk_dataset)))
                    epoch_accs.append((acc, len(chunk_dataset)))
                except Exception as e:
                    logger.error(f"Error training on chunk starting at {start}: {e}")
                    continue

            # Aggregate epoch metrics (weighted)
            if len(epoch_losses) > 0:
                total_samples = sum(n for _, n in epoch_losses)
                weighted_loss = sum(l * n for l, n in epoch_losses) / total_samples
                weighted_acc = sum(a * n for a, n in epoch_accs) / total_samples
            else:
                weighted_loss = None
                weighted_acc = None

            # Validation
            val_metrics = None
            if val_loader is not None:
                val_metrics = trainer.validate(val_loader)

            # Scheduler step
            if trainer.scheduler:
                if hasattr(trainer.scheduler, '__class__') and trainer.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    if val_metrics:
                        trainer.scheduler.step(val_metrics['accuracy'])
                else:
                    trainer.scheduler.step()

            current_lr = trainer.optimizer.param_groups[0]['lr']

            # Log epoch summary
            logger.info(
                f"Epoch {epoch+1} summary - Train Loss: {weighted_loss} - Train Acc: {weighted_acc} - Val Acc: {val_metrics['accuracy'] if val_metrics else 'N/A'} - LR: {current_lr}"
            )

            # Save best model based on validation F1 (if available)
            if val_metrics:
                if val_metrics['f1'] > trainer.best_val_f1:
                    trainer.best_val_f1 = val_metrics['f1']
                    trainer.best_val_acc = val_metrics['accuracy']
                    trainer.best_epoch = epoch + 1
                    trainer.patience_counter = 0
                    trainer.save_checkpoint(save_dir / 'best_model.pt', epoch + 1, val_metrics)
                    logger.info(f"New best model saved at epoch {epoch+1} (F1={trainer.best_val_f1:.4f})")
                else:
                    trainer.patience_counter += 1

                # Early stopping
                if trainer.patience_counter >= config.get('training', {}).get('patience', 20):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

            # Periodic checkpoint
            if (epoch + 1) % config.get('training', {}).get('save_every', 10) == 0:
                trainer.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch+1}.pt', epoch + 1, val_metrics)

        # Save final model and history
        trainer.save_checkpoint(save_dir / 'final_model.pt', num_epochs, val_metrics)
        trainer.save_history(save_dir / 'training_history.json')

        # Evaluate on test set if available
        if test_loader is not None:
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate(test_loader)
            with open(output_dir / 'test_results.json', 'w') as f:
                results = {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}
                results['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
                json.dump(results, f, indent=2)

        logger.info("Streaming training and evaluation complete")

if __name__ == "__main__":
    main()