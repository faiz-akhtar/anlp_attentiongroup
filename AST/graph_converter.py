# graph_converter.py
"""
Graph converter module to convert AST to graph representation for GNN
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

class GraphConverter:
    """Converts AST to graph representation suitable for GNN"""
    
    def __init__(self, max_nodes=500, node_vocab_size=1000):
        """
        Initialize graph converter
        
        Args:
            max_nodes: Maximum number of nodes in graph
            node_vocab_size: Size of node type vocabulary
        """
        self.max_nodes = max_nodes
        self.node_vocab_size = node_vocab_size
        self.node_type_vocab = {}
        self.node_type_counter = 0
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.UNK_TOKEN = 1
        self.node_type_counter = 2
        
        # Edge types
        self.edge_types = {
            'child': 0,
            'next_sibling': 1,
            'parent': 2,
            'prev_sibling': 3,
            'data_flow': 4,
            'control_flow': 5
        }
    
    def ast_to_graph(self, ast_node, language: str) -> Dict[str, Any]:
        """
        Convert AST to graph representation
        
        Args:
            ast_node: AST root node (tree-sitter Node or dict)
            language: Programming language
            
        Returns:
            Dictionary with graph data
        """
        # Extract nodes and edges
        nodes, edges, node_types, positions = self._extract_graph_components(ast_node)
        
        # Limit number of nodes
        if len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
            # Filter edges to only include valid nodes
            edges = [(u, v, t) for u, v, t in edges if u < self.max_nodes and v < self.max_nodes]
        
        # Create adjacency information
        num_nodes = len(nodes)
        edge_index = self._create_edge_index(edges, num_nodes)
        
        # Create node features
        node_features = self._create_node_features(nodes, node_types, positions, language)
        
        # Create edge features
        edge_features = self._create_edge_features(edges)
        
        return {
            'num_nodes': num_nodes,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'node_types': node_types[:num_nodes],
            'positions': positions[:num_nodes],
            'adjacency_list': self._create_adjacency_list(edges, num_nodes)
        }
    
    def _extract_graph_components(self, ast_node) -> Tuple:
        """
        Extract nodes and edges from AST
        
        Args:
            ast_node: AST root node
            
        Returns:
            Tuple of (nodes, edges, node_types, positions)
        """
        nodes = []
        edges = []
        node_types = []
        positions = []
        node_id_map = {}
        
        def traverse(node, parent_id=None):
            # Assign node ID
            node_id = len(nodes)
            nodes.append(node)
            
            # Get node type
            if hasattr(node, 'type'):
                node_type = node.type
            elif isinstance(node, dict) and 'type' in node:
                node_type = node['type']
            else:
                node_type = str(type(node).__name__)
            
            # Map node type to ID
            if node_type not in self.node_type_vocab:
                if self.node_type_counter < self.node_vocab_size:
                    self.node_type_vocab[node_type] = self.node_type_counter
                    self.node_type_counter += 1
                else:
                    self.node_type_vocab[node_type] = self.UNK_TOKEN
            
            node_types.append(self.node_type_vocab.get(node_type, self.UNK_TOKEN))
            
            # Get position
            position = [0, 0, 0, 0]  # start_line, start_col, end_line, end_col
            if hasattr(node, 'start_point') and hasattr(node, 'end_point'):
                position = [
                    node.start_point[0], node.start_point[1],
                    node.end_point[0], node.end_point[1]
                ]
            elif isinstance(node, dict):
                if 'start_point' in node and 'end_point' in node:
                    position = [
                        node['start_point'][0], node['start_point'][1],
                        node['end_point'][0], node['end_point'][1]
                    ]
            positions.append(position)
            
            # Add edge from parent
            if parent_id is not None:
                edges.append((parent_id, node_id, self.edge_types['child']))
                edges.append((node_id, parent_id, self.edge_types['parent']))
            
            # Process children
            children = []
            if hasattr(node, 'children'):
                children = node.children
            elif isinstance(node, dict) and 'children' in node:
                children = node['children']
            
            child_ids = []
            for child in children:
                child_id = traverse(child, node_id)
                child_ids.append(child_id)
            
            # Add sibling edges
            for i in range(len(child_ids) - 1):
                edges.append((child_ids[i], child_ids[i+1], self.edge_types['next_sibling']))
                edges.append((child_ids[i+1], child_ids[i], self.edge_types['prev_sibling']))
            
            return node_id
        
        traverse(ast_node)
        return nodes, edges, node_types, positions
    
    def _create_edge_index(self, edges: List[Tuple], num_nodes: int) -> np.ndarray:
        """
        Create edge index matrix for GNN
        
        Args:
            edges: List of edges (source, target, type)
            num_nodes: Number of nodes
            
        Returns:
            Edge index matrix (2 x num_edges)
        """
        if not edges:
            return np.array([[], []], dtype=np.int64)
        
        edge_index = np.array([[e[0], e[1]] for e in edges], dtype=np.int64).T
        return edge_index
    
    def _create_node_features(self, nodes: List, node_types: List, 
                             positions: List, language: str) -> np.ndarray:
        """
        Create node feature matrix
        
        Args:
            nodes: List of AST nodes
            node_types: List of node type indices
            positions: List of node positions
            language: Programming language
            
        Returns:
            Node feature matrix (num_nodes x feature_dim)
        """
        num_nodes = len(nodes)
        
        # Feature dimensions
        type_embedding_dim = 64
        position_dim = 4
        language_dim = 8
        structural_dim = 16
        
        total_dim = type_embedding_dim + position_dim + language_dim + structural_dim
        features = np.zeros((num_nodes, total_dim), dtype=np.float32)
        
        for i, (node, node_type, position) in enumerate(zip(nodes, node_types, positions)):
            feat_idx = 0
            
            # Node type embedding (one-hot or learned embedding)
            type_feat = np.zeros(type_embedding_dim)
            if node_type < type_embedding_dim:
                type_feat[node_type] = 1.0
            else:
                # Hash to get pseudo-random features for unknown types
                type_hash = int(hashlib.md5(str(node_type).encode()).hexdigest(), 16)
                indices = [type_hash % type_embedding_dim, 
                          (type_hash // type_embedding_dim) % type_embedding_dim]
                for idx in indices[:2]:
                    type_feat[idx] = 0.5
            features[i, feat_idx:feat_idx+type_embedding_dim] = type_feat
            feat_idx += type_embedding_dim
            
            # Position features (normalized)
            pos_feat = np.array(position, dtype=np.float32)
            pos_feat = pos_feat / (pos_feat.max() + 1e-8)  # Normalize
            features[i, feat_idx:feat_idx+position_dim] = pos_feat
            feat_idx += position_dim
            
            # Language features (one-hot)
            lang_feat = np.zeros(language_dim)
            lang_map = {
                'python': 0, 'java': 1, 'cpp': 2, 'c': 3,
                'javascript': 4, 'go': 5, 'php': 6, 'csharp': 7
            }
            if language in lang_map:
                lang_feat[lang_map[language]] = 1.0
            features[i, feat_idx:feat_idx+language_dim] = lang_feat
            feat_idx += language_dim
            
            # Structural features
            struct_feat = self._compute_structural_features(node, i, nodes)
            features[i, feat_idx:feat_idx+structural_dim] = struct_feat[:structural_dim]
        
        return features
    
    def _compute_structural_features(self, node, node_idx: int, all_nodes: List) -> np.ndarray:
        """
        Compute structural features for a node
        
        Args:
            node: AST node
            node_idx: Index of the node
            all_nodes: List of all nodes
            
        Returns:
            Structural feature vector
        """
        features = np.zeros(16, dtype=np.float32)
        
        # Depth in tree (approximation based on index)
        features[0] = np.log1p(node_idx)
        
        # Number of children
        num_children = 0
        if hasattr(node, 'children'):
            num_children = len(node.children)
        elif isinstance(node, dict) and 'children' in node:
            num_children = len(node['children'])
        features[1] = np.log1p(num_children)
        
        # Is leaf node
        features[2] = 1.0 if num_children == 0 else 0.0
        
        # Node text length (if available)
        if hasattr(node, 'text'):
            features[3] = np.log1p(len(node.text))
        elif isinstance(node, dict) and 'text' in node:
            features[3] = np.log1p(len(node['text']))
        
        # Relative position in siblings (approximation)
        features[4] = node_idx / (len(all_nodes) + 1e-8)
        
        return features
    
    def _create_edge_features(self, edges: List[Tuple]) -> np.ndarray:
        """
        Create edge feature matrix
        
        Args:
            edges: List of edges with types
            
        Returns:
            Edge feature matrix (num_edges x edge_feature_dim)
        """
        if not edges:
            return np.array([], dtype=np.float32).reshape(0, len(self.edge_types))
        
        num_edges = len(edges)
        edge_features = np.zeros((num_edges, len(self.edge_types)), dtype=np.float32)
        
        for i, edge in enumerate(edges):
            if len(edge) >= 3:
                edge_type = edge[2]
                if edge_type < len(self.edge_types):
                    edge_features[i, edge_type] = 1.0
        
        return edge_features
    
    def _create_adjacency_list(self, edges: List[Tuple], num_nodes: int) -> Dict[int, List[int]]:
        """
        Create adjacency list representation
        
        Args:
            edges: List of edges
            num_nodes: Number of nodes
            
        Returns:
            Adjacency list
        """
        adj_list = defaultdict(list)
        
        for edge in edges:
            source, target = edge[0], edge[1]
            adj_list[source].append(target)
        
        # Ensure all nodes are in adjacency list
        for i in range(num_nodes):
            if i not in adj_list:
                adj_list[i] = []
        
        return dict(adj_list)
    
    def graph_to_sequence(self, graph_data: Dict) -> List[int]:
        """
        Convert graph to sequence representation (for baseline models)
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            Sequence of node types
        """
        return graph_data.get('node_types', [])