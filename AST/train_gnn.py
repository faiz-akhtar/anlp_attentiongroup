# train_gnn.py
"""
Training script for GNN-based code detection
"""

import argparse
import logging
import json
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

from gnn_model import create_model
from gnn_dataset import CodeASTDataModule, load_processed_data
from gnn_trainer import GNNTrainer

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


def prepare_data(data_path: str, task: str, val_split: float = 0.15, test_split: float = 0.15):
    """
    Load and split data into train/val/test sets
    
    Args:
        data_path: Path to processed data
        task: Task type
        val_split: Validation split ratio
        test_split: Test split ratio
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    logger.info(f"Loading processed data from {data_path}")
    data = load_processed_data(data_path)
    
    # Split data
    train_val_data, test_data = train_test_split(
        data,
        test_size=test_split,
        random_state=42,
        stratify=[d['label'] for d in data]
    )
    
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_split / (1 - test_split),
        random_state=42,
        stratify=[d['label'] for d in train_val_data]
    )
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Train GNN for code detection')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./processed_data/ast_features_subtask_a_train.pkl',
        # required=True,
        help='Path to processed data (pickle file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config.json',
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='subtask_a',
        choices=['subtask_a', 'subtask_b', 'subtask_c'],
        help='Task type'
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
    
    args = parser.parse_args()
    
    # Load configuration
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
    
    # Prepare data
    train_data, val_data, test_data = prepare_data(args.data_path, args.task)
    
    # Create data module
    data_module = CodeASTDataModule(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        augment_train=config['training']['augment_train'],
        task=args.task
    )
    
    # Get feature dimensions
    node_feature_dim = data_module.get_feature_dim()
    num_classes = data_module.get_num_classes()
    
    logger.info(f"Node feature dimension: {node_feature_dim}")
    logger.info(f"Number of classes: {num_classes}")
    
    # Update model config
    config['model_config']['node_feature_dim'] = node_feature_dim
    config['model_config']['num_classes'] = num_classes
    
    # Create model
    model = create_model(config['model_config'])
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer_config = {
        **config['training'],
        'optimizer': config['optimizer'],
        'scheduler': config['scheduler'],
        'task': args.task,
        'model_config': config['model_config']
    }
    
    trainer = GNNTrainer(
        model=model,
        data_module=data_module,
        config=trainer_config,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Evaluation only mode
    if args.eval_only:
        if not args.resume:
            raise ValueError("--eval_only requires --resume to specify a checkpoint")
        logger.info("Evaluation mode - running test evaluation only")
        test_metrics = trainer.evaluate()
        
        # Save results
        with open(output_dir / 'test_results.json', 'w') as f:
            results = {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}
            results['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
            json.dump(results, f, indent=2)
        
        return
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        num_epochs=config['training']['epochs'],
        save_dir=output_dir / 'checkpoints'
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate()
    
    # Save final results
    with open(output_dir / 'test_results.json', 'w') as f:
        results = {k: v for k, v in test_metrics.items() if k != 'confusion_matrix'}
        results['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
        json.dump(results, f, indent=2)
    
    logger.info("Training and evaluation complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()