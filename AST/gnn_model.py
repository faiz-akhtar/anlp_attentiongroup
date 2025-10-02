# gnn_model.py (Fixed version with missing import)
"""
Graph Neural Network model for code detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as pyg_nn
from torch_geometric.utils import add_self_loops, degree, softmax
import numpy as np
from typing import Dict, Tuple, Optional

class CodeGNN(nn.Module):
    """
    Graph Neural Network for code detection
    Supports multiple GNN architectures: GCN, GAT, and custom Message Passing
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        gnn_type: str = 'gat',
        use_edge_features: bool = True,
        edge_feature_dim: int = 6,
        pooling: str = 'mean'
    ):
        """
        Initialize GNN model
        
        Args:
            node_feature_dim: Dimension of node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
            gnn_type: Type of GNN ('gcn', 'gat', 'custom')
            use_edge_features: Whether to use edge features
            edge_feature_dim: Dimension of edge features
            pooling: Pooling method ('mean', 'max', 'add', 'attention')
        """
        super(CodeGNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.use_edge_features = use_edge_features
        self.pooling = pooling
        
        # Node embedding layer
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge embedding layer (if using edge features)
        if use_edge_features:
            self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim // 4)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        # keep track of each layer's input/output dimension to build correct skip projections
        layer_input_dims = []
        layer_output_dims = []
        
        for i in range(num_layers):
            # determine input dim for this layer
            in_dim = hidden_dim if i == 0 else layer_output_dims[i - 1]
            layer_input_dims.append(in_dim)

            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(in_dim, hidden_dim))
                out_dim = hidden_dim
            elif gnn_type == 'gat':
                # GAT with multi-head attention
                heads = 4 if i < num_layers - 1 else 1
                # when concat=True intermediate out dim = hidden_dim * heads
                concat = (i < num_layers - 1)
                self.gnn_layers.append(
                    GATConv(
                        in_dim,
                        hidden_dim,
                        heads=heads,
                        dropout=dropout,
                        concat=concat
                    )
                )
                out_dim = hidden_dim * heads if concat else hidden_dim
            elif gnn_type == 'custom':
                self.gnn_layers.append(CustomMessagePassing(in_dim, hidden_dim))
                out_dim = hidden_dim

            # Add normalization sized to this layer's output dim (LayerNorm is stable for small batches)
            self.batch_norms.append(nn.LayerNorm(out_dim))
            layer_output_dims.append(out_dim)
         
        # Attention pooling layer (if selected)
        if pooling == 'attention':
            self.attention_pooling = AttentionPooling(hidden_dim)
        
        # Final classifier
        classifier_input_dim = hidden_dim * (3 if pooling == 'all' else 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Additional components for advanced features
        self.skip_connections = nn.ModuleList()
        # build skip projections that map previous-layer output dim -> current-layer output dim
        # so projection sizes match actual h before the layer
        for i in range(1, num_layers):
            prev_dim = layer_output_dims[i - 1] if i - 1 >= 0 else hidden_dim
            cur_out_dim = layer_output_dims[i]
            self.skip_connections.append(nn.Linear(prev_dim, cur_out_dim))
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features (num_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_feature_dim)
            batch: Batch assignment vector (num_nodes,)
            
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Encode node features
        h = self.node_encoder(x)
        initial_h = h.clone()
        
        # Process through GNN layers (apply skip from previous h -> current out dim)
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            h_before = h  # actual tensor before this layer (may change dim across layers)

            # Apply GNN layer
            if self.gnn_type == 'gat':
                h_out = gnn_layer(h_before, edge_index)
            elif self.use_edge_features and edge_attr is not None and self.gnn_type == 'custom':
                edge_features = self.edge_encoder(edge_attr)
                h_out = gnn_layer(h_before, edge_index, edge_features)
            else:
                h_out = gnn_layer(h_before, edge_index)

            # If skip exists, project previous-h to current out dim and add
            if i > 0:
                skip = self.skip_connections[i - 1](h_before)
                h = h_out + skip
            else:
                h = h_out

            # Apply batch normalization
            h = batch_norm(h)

            # Apply activation and dropout (except last layer)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling == 'mean':
            graph_repr = global_mean_pool(h, batch)
        elif self.pooling == 'max':
            graph_repr = global_max_pool(h, batch)
        elif self.pooling == 'add':
            graph_repr = global_add_pool(h, batch)
        elif self.pooling == 'attention':
            graph_repr = self.attention_pooling(h, batch)
        elif self.pooling == 'all':
            # Concatenate multiple pooling methods
            mean_pool = global_mean_pool(h, batch)
            max_pool = global_max_pool(h, batch)
            add_pool = global_add_pool(h, batch)
            graph_repr = torch.cat([mean_pool, max_pool, add_pool], dim=1)
        else:
            graph_repr = global_mean_pool(h, batch)
        
        # Classification
        out = self.classifier(graph_repr)
        
        return out
    
    def get_node_embeddings(self, x, edge_index, edge_attr=None):
        """
        Get node embeddings after GNN layers
        
        Returns:
            Node embeddings (num_nodes, hidden_dim)
        """
        h = self.node_encoder(x)
        
        for i, (gnn_layer, batch_norm) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            if self.use_edge_features and edge_attr is not None and self.gnn_type == 'custom':
                edge_features = self.edge_encoder(edge_attr)
                h = gnn_layer(h, edge_index, edge_features)
            else:
                h = gnn_layer(h, edge_index)
            
            h = batch_norm(h)
            
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h


class CustomMessagePassing(MessagePassing):
    """
    Custom message passing layer for code AST graphs
    """
    
    def __init__(self, in_channels, out_channels):
        super(CustomMessagePassing, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(in_channels // 4, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform node features
        x_self = self.lin_self(x)
        x = self.lin(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Combine with self-features
        return out + x_self
    
    def message(self, x_j, edge_attr=None):
        # x_j: Features of source nodes
        if edge_attr is not None:
            edge_features = self.lin_edge(edge_attr)
            return x_j + edge_features
        return x_j
    
    def update(self, aggr_out):
        return F.relu(aggr_out)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for graph-level representation
    """
    
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, batch):
        # Compute attention scores
        scores = self.attention(x)
        
        # Apply softmax per graph using PyG's softmax
        scores = softmax(scores.squeeze(-1), batch, num_nodes=x.size(0))
        
        # Weighted sum
        weighted = x * scores.unsqueeze(-1)
        return global_add_pool(weighted, batch)


class HierarchicalGNN(nn.Module):
    """
    Hierarchical GNN that processes AST at multiple levels
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(HierarchicalGNN, self).__init__()
        
        # Local GNN for subtrees
        self.local_gnn = CodeGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim // 2,
            num_layers=2,
            num_classes=hidden_dim // 2,
            dropout=dropout,
            gnn_type='gat'
        )
        
        # Global GNN for entire AST
        self.global_gnn = CodeGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_classes=hidden_dim,
            dropout=dropout,
            gnn_type='gat'
        )
        
        # Final classifier combining local and global features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Process with local GNN
        local_features = self.local_gnn(x, edge_index, edge_attr, batch)
        
        # Process with global GNN
        global_features = self.global_gnn(x, edge_index, edge_attr, batch)
        
        # Combine features
        combined = torch.cat([local_features, global_features], dim=1)
        
        # Final classification
        return self.classifier(combined)


def create_model(config: Dict) -> nn.Module:
    """
    Factory function to create GNN model based on configuration
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        GNN model
    """
    model_type = config.get('model_type', 'standard')
    
    if model_type == 'hierarchical':
        return HierarchicalGNN(
            node_feature_dim=config['node_feature_dim'],
            hidden_dim=config.get('hidden_dim', 256),
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3)
        )
    else:
        return CodeGNN(
            node_feature_dim=config['node_feature_dim'],
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3),
            gnn_type=config.get('gnn_type', 'gat'),
            use_edge_features=config.get('use_edge_features', True),
            edge_feature_dim=config.get('edge_feature_dim', 6),
            pooling=config.get('pooling', 'mean')
        )