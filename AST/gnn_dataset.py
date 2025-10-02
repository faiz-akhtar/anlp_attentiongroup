# gnn_dataset.py
"""
PyTorch Geometric dataset wrapper for code AST graphs
"""

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CodeASTDataset(Dataset):
    """
    PyTorch Geometric dataset for code AST graphs
    """
    
    def __init__(
        self,
        processed_data: List[Dict],
        transform=None,
        pre_transform=None,
        augment: bool = False,
        task: str = 'subtask_a'
    ):
        """
        Initialize dataset
        
        Args:
            processed_data: List of processed samples from main.py
            transform: Optional transform to apply
            pre_transform: Optional pre-transform to apply
            augment: Whether to apply data augmentation
            task: Task type for label mapping
        """
        super().__init__(None, transform, pre_transform)
        self.processed_data = processed_data
        self.augment = augment
        self.task = task
        self.data_list = self._create_data_list()
        
    def _create_data_list(self) -> List[Data]:
        """
        Convert processed data to PyTorch Geometric Data objects
        """
        data_list = []
        
        for sample in self.processed_data:
            try:
                # Extract graph data
                graph = sample['graph']
                features = sample['features']
                label = sample['label']
                
                # Node features
                node_features = graph['node_features']
                if isinstance(node_features, np.ndarray):
                    x = torch.FloatTensor(node_features)
                else:
                    x = torch.FloatTensor(np.array(node_features))
                
                # Edge indices
                edge_index = graph['edge_index']
                if isinstance(edge_index, np.ndarray):
                    edge_index = torch.LongTensor(edge_index)
                else:
                    edge_index = torch.LongTensor(np.array(edge_index))
                
                # Edge features
                edge_attr = None
                if 'edge_features' in graph and graph['edge_features'] is not None:
                    edge_features = graph['edge_features']
                    if isinstance(edge_features, np.ndarray):
                        edge_attr = torch.FloatTensor(edge_features)
                    else:
                        edge_attr = torch.FloatTensor(np.array(edge_features))
                
                # Label
                y = torch.LongTensor([label])
                
                # Additional features (robust to dict or array formats)
                # robust extraction helper
                struct = features.get('structural') if isinstance(features, dict) else features['structural']
                lex = features.get('lexical') if isinstance(features, dict) else features['lexical']
                
                # structural: if dict use keys, else assume array and use indices [0,1,2]
                if isinstance(struct, dict):
                    num_nodes = float(struct.get('num_nodes', 0.0))
                    max_depth = float(struct.get('max_depth', 0.0))
                    avg_branch = float(struct.get('avg_branching_factor', 0.0))
                else:
                    struct_arr = np.asarray(struct)
                    num_nodes = float(struct_arr[0]) if struct_arr.size > 0 else 0.0
                    max_depth = float(struct_arr[1]) if struct_arr.size > 1 else 0.0
                    avg_branch = float(struct_arr[2]) if struct_arr.size > 2 else 0.0
                
                # lexical: if dict use keys, else assume array and use indices [3,1]
                if isinstance(lex, dict):
                    avg_line_length = float(lex.get('avg_line_length', 0.0))
                    num_comments = float(lex.get('num_comments', 0.0))
                else:
                    lex_arr = np.asarray(lex)
                    avg_line_length = float(lex_arr[3]) if lex_arr.size > 3 else 0.0
                    num_comments = float(lex_arr[1]) if lex_arr.size > 1 else 0.0
                
                additional_features = torch.FloatTensor([
                    num_nodes,
                    max_depth,
                    avg_branch,
                    avg_line_length,
                    num_comments
                ])
                
                # Create PyTorch Geometric Data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    num_nodes=graph['num_nodes'],
                    additional_features=additional_features,
                    language=sample['language'],
                    metadata=sample.get('metadata', {})
                )
                
                # Apply augmentation if enabled
                if self.augment:
                    data = self._augment_graph(data)
                
                data_list.append(data)
                
            except Exception as e:
                logger.warning(f"Error creating data object: {e}")
                continue
        
        return data_list
    
    def _augment_graph(self, data: Data) -> Data:
        """
        Apply data augmentation to graph
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Augmented data object
        """
        # Random node feature dropout
        if torch.rand(1) > 0.5:
            mask = torch.rand(data.x.size(0)) > 0.1
            data.x = data.x * mask.unsqueeze(1)
        
        # Random edge dropout
        if torch.rand(1) > 0.5 and data.edge_index.size(1) > 10:
            num_edges = data.edge_index.size(1)
            keep_edges = int(num_edges * 0.9)
            perm = torch.randperm(num_edges)[:keep_edges]
            data.edge_index = data.edge_index[:, perm]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[perm]
        
        return data
    
    def len(self) -> int:
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        return self.data_list[idx]


class CodeASTDataModule:
    """
    Data module for handling train/val/test splits and dataloaders
    """
    
    def __init__(
        self,
        train_data: List[Dict],
        val_data: Optional[List[Dict]] = None,
        test_data: Optional[List[Dict]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        augment_train: bool = True,
        task: str = 'subtask_a'
    ):
        """
        Initialize data module
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            batch_size: Batch size
            num_workers: Number of workers for data loading
            augment_train: Whether to augment training data
            task: Task type
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task = task
        
        # Create datasets
        self.train_dataset = CodeASTDataset(
            train_data,
            augment=augment_train,
            task=task
        )
        
        self.val_dataset = None
        if val_data:
            self.val_dataset = CodeASTDataset(
                val_data,
                augment=False,
                task=task
            )
        
        self.test_dataset = None
        if test_data:
            self.test_dataset = CodeASTDataset(
                test_data,
                augment=False,
                task=task
            )
        
        # Compute statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute dataset statistics for normalization"""
        all_features = []
        all_labels = []
        
        for data in self.train_dataset:
            all_features.append(data.x)
            all_labels.append(data.y)
        
        # Compute mean and std for normalization
        all_features = torch.cat(all_features, dim=0)
        self.feature_mean = all_features.mean(dim=0)
        self.feature_std = all_features.std(dim=0) + 1e-8
        
        # Label distribution
        all_labels = torch.cat(all_labels)
        unique_labels, counts = torch.unique(all_labels, return_counts=True)
        self.label_distribution = {
            int(label): int(count) 
            for label, count in zip(unique_labels, counts)
        }
        
        # Class weights for imbalanced data
        total = len(all_labels)
        self.class_weights = torch.FloatTensor([
            total / (len(unique_labels) * counts[i])
            for i in range(len(unique_labels))
        ])
        
        logger.info(f"Dataset statistics computed:")
        logger.info(f"  Total samples: {total}")
        logger.info(f"  Label distribution: {self.label_distribution}")
        logger.info(f"  Class weights: {self.class_weights}")
    
    def get_train_dataloader(self) -> DataLoader:
        """Get training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            drop_last=True
        )
    
    def get_val_dataloader(self) -> DataLoader:
        """Get validation dataloader"""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def get_test_dataloader(self) -> DataLoader:
        """Get test dataloader"""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    def get_feature_dim(self) -> int:
        """Get input feature dimension"""
        sample_data = self.train_dataset[0]
        return sample_data.x.size(1)
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        if self.task == 'subtask_a':
            return 2  # Human vs Machine
        elif self.task == 'subtask_b':
            return 11  # Human + 10 generators
        elif self.task == 'subtask_c':
            return 4  # Human, Machine, Hybrid, Adversarial
        else:
            return 2


def collate_fn(batch: List[Data]) -> Data:
    """
    Custom collate function for batching graphs
    
    Args:
        batch: List of Data objects
        
    Returns:
        Batched Data object
    """
    # Filter out None values
    batch = [data for data in batch if data is not None]
    
    if len(batch) == 0:
        return None
    
    # Use default PyG batching
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


def load_processed_data(path: str) -> List[Dict]:
    """
    Load processed data from disk
    
    Args:
        path: Path to processed data file
        
    Returns:
        List of processed samples
    """
    path = Path(path)
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {len(data)} samples from {path}")
    return data