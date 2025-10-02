# feature_extractor.py
"""
Feature extractor module for extracting various code features from AST
"""

import numpy as np
from typing import Dict, List, Any, Optional
import re
import logging
from collections import Counter, defaultdict
import hashlib

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts various features from code and AST for detection"""
    
    def __init__(self):
        """Initialize feature extractor"""
        self.feature_dimensions = {
            'lexical': 50,
            'syntactic': 30,
            'semantic': 40,
            'statistical': 20,
            'structural': 30
        }
        
        # Common keywords per language
        self.language_keywords = {
            'python': ['def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'with', 'return', 'yield', 'lambda'],
            'java': ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'new', 'return', 'import'],
            'cpp': ['class', 'struct', 'namespace', 'using', 'template', 'public', 'private', 'protected', 'virtual', 'const', 'static', 'void', 'return'],
            'c': ['struct', 'typedef', 'enum', 'union', 'static', 'const', 'void', 'return', 'if', 'else', 'for', 'while', 'switch'],
            'javascript': ['function', 'const', 'let', 'var', 'class', 'import', 'export', 'async', 'await', 'return', 'if', 'else', 'for'],
            'go': ['func', 'package', 'import', 'type', 'struct', 'interface', 'return', 'if', 'else', 'for', 'range', 'defer', 'go'],
            'php': ['function', 'class', 'public', 'private', 'protected', 'static', 'return', 'if', 'else', 'foreach', 'while', 'require', 'include'],
            'csharp': ['class', 'interface', 'public', 'private', 'protected', 'static', 'void', 'return', 'using', 'namespace', 'async', 'await']
        }
        
        # Common patterns that might indicate AI generation
        self.ai_patterns = [
            r'# TODO:.*',
            r'# FIXME:.*',
            r'# NOTE:.*',
            r'"""[\s\S]*?"""',  # Docstrings
            r"'''[\s\S]*?'''",  # Triple quotes
            r'^\s*#.*$',  # Comments
            r'^\s*//.*$',  # C-style comments
            r'/\*[\s\S]*?\*/',  # Block comments
        ]
    
    def extract_features(self, ast_node, graph_data: Dict, code: str, language: str) -> Dict[str, np.ndarray]:
        """
        Extract all features from code
        
        Args:
            ast_node: AST root node
            graph_data: Graph representation
            code: Source code string
            language: Programming language
            
        Returns:
            Dictionary of feature arrays
        """
        features = {}
        
        # Extract different feature types
        features['lexical'] = self.extract_lexical_features(code, language)
        features['syntactic'] = self.extract_syntactic_features(ast_node, language)
        features['semantic'] = self.extract_semantic_features(ast_node, code, language)
        features['statistical'] = self.extract_statistical_features(code, ast_node)
        features['structural'] = self.extract_structural_features(graph_data)
        
        # Combine all features
        features['combined'] = self._combine_features(features)
        
        return features
    
    def extract_lexical_features(self, code: str, language: str) -> np.ndarray:
        """
        Extract lexical features from code
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            Lexical feature vector
        """
        features = np.zeros(self.feature_dimensions['lexical'], dtype=np.float32)
        
        # Basic statistics
        features[0] = len(code)  # Code length
        features[1] = len(code.split('\n'))  # Number of lines
        features[2] = len(code.split())  # Number of tokens
        
        # Character statistics
        features[3] = code.count(' ') / (len(code) + 1e-8)  # Space ratio
        features[4] = code.count('\n') / (len(code) + 1e-8)  # Newline ratio
        features[5] = code.count('\t') / (len(code) + 1e-8)  # Tab ratio
        
        # Keyword statistics
        keywords = self.language_keywords.get(language, [])
        keyword_counts = []
        for keyword in keywords[:10]:  # Top 10 keywords
            count = len(re.findall(r'\b' + keyword + r'\b', code))
            keyword_counts.append(count)
        
        if keyword_counts:
            features[6:16] = np.array(keyword_counts[:10]) / (len(code.split()) + 1e-8)
        
        # Operator statistics
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=']
        for i, op in enumerate(operators[:10]):
            features[16 + i] = code.count(op) / (len(code) + 1e-8)
        
        # Bracket statistics
        features[26] = code.count('(') / (len(code) + 1e-8)
        features[27] = code.count(')') / (len(code) + 1e-8)
        features[28] = code.count('[') / (len(code) + 1e-8)
        features[29] = code.count(']') / (len(code) + 1e-8)
        features[30] = code.count('{') / (len(code) + 1e-8)
        features[31] = code.count('}') / (len(code) + 1e-8)
        
        # Comment statistics
        single_comments = len(re.findall(r'#.*$|//.*$', code, re.MULTILINE))
        multi_comments = len(re.findall(r'/\*[\s\S]*?\*/', code))
        features[32] = single_comments / (len(code.split('\n')) + 1e-8)
        features[33] = multi_comments / (len(code.split('\n')) + 1e-8)
        
        # String literal statistics
        single_quotes = len(re.findall(r"'[^']*'", code))
        double_quotes = len(re.findall(r'"[^"]*"', code))
        features[34] = single_quotes / (len(code.split()) + 1e-8)
        features[35] = double_quotes / (len(code.split()) + 1e-8)
        
        # Indentation statistics
        lines = code.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
        
        if indent_levels:
            features[36] = np.mean(indent_levels)
            features[37] = np.std(indent_levels)
            features[38] = np.max(indent_levels)
        
        # Variable naming patterns
        camel_case = len(re.findall(r'\b[a-z][a-zA-Z]*\b', code))
        snake_case = len(re.findall(r'\b[a-z]+_[a-z]+\b', code))
        pascal_case = len(re.findall(r'\b[A-Z][a-zA-Z]*\b', code))
        
        features[39] = camel_case / (len(code.split()) + 1e-8)
        features[40] = snake_case / (len(code.split()) + 1e-8)
        features[41] = pascal_case / (len(code.split()) + 1e-8)
        
        # AI pattern detection
        ai_pattern_count = 0
        for pattern in self.ai_patterns:
            ai_pattern_count += len(re.findall(pattern, code, re.MULTILINE))
        features[42] = ai_pattern_count / (len(lines) + 1e-8)
        
        return features
    
    def extract_syntactic_features(self, ast_node, language: str) -> np.ndarray:
        """
        Extract syntactic features from AST
        
        Args:
            ast_node: AST root node
            language: Programming language
            
        Returns:
            Syntactic feature vector
        """
        features = np.zeros(self.feature_dimensions['syntactic'], dtype=np.float32)
        
        if ast_node is None:
            return features
        
        # Node type distribution
        node_types = self._collect_node_types(ast_node)
        type_counter = Counter(node_types)
        
        # Most common node types
        most_common = type_counter.most_common(10)
        for i, (node_type, count) in enumerate(most_common):
            if i < 10:
                features[i] = count / (len(node_types) + 1e-8)
        
        # Tree structure features
        features[10] = self._get_tree_depth(ast_node)
        features[11] = self._get_max_branching_factor(ast_node)
        features[12] = len(node_types)  # Total nodes
        features[13] = len(set(node_types))  # Unique node types
        
        # Balance metrics
        features[14] = self._calculate_tree_balance(ast_node)
        
        # Specific syntactic patterns
        features[15] = node_types.count('function_definition') / (len(node_types) + 1e-8)
        features[16] = node_types.count('class_definition') / (len(node_types) + 1e-8)
        features[17] = node_types.count('if_statement') / (len(node_types) + 1e-8)
        features[18] = node_types.count('for_statement') / (len(node_types) + 1e-8)
        features[19] = node_types.count('while_statement') / (len(node_types) + 1e-8)
        
        return features
    
    def extract_semantic_features(self, ast_node, code: str, language: str) -> np.ndarray:
        """
        Extract semantic features
        
        Args:
            ast_node: AST root node
            code: Source code
            language: Programming language
            
        Returns:
            Semantic feature vector
        """
        features = np.zeros(self.feature_dimensions['semantic'], dtype=np.float32)
        
        # Function complexity
        functions = self._extract_functions(ast_node, code)
        if functions:
            complexities = [self._calculate_complexity(f) for f in functions]
            features[0] = np.mean(complexities)
            features[1] = np.std(complexities)
            features[2] = np.max(complexities)
            features[3] = len(functions)
        
        # Variable usage patterns
        variables = self._extract_variables(ast_node, code)
        if variables:
            features[4] = len(variables)
            features[5] = len(set(variables))  # Unique variables
            features[6] = np.mean([len(v) for v in variables])  # Average variable name length
        
        # Import/dependency patterns
        imports = self._extract_imports(code, language)
        features[7] = len(imports)
        features[8] = len(set(imports))  # Unique imports
        
        # Control flow patterns
        features[9] = code.count('if ') + code.count('if(')
        features[10] = code.count('else')
        features[11] = code.count('for ') + code.count('for(')
        features[12] = code.count('while ') + code.count('while(')
        features[13] = code.count('return')
        
        # Exception handling
        if language in ['python', 'java', 'cpp', 'csharp']:
            features[14] = code.count('try')
            features[15] = code.count('catch') + code.count('except')
            features[16] = code.count('finally')
        
        return features
    
    def extract_statistical_features(self, code: str, ast_node) -> np.ndarray:
        """
        Extract statistical features
        
        Args:
            code: Source code
            ast_node: AST root node
            
        Returns:
            Statistical feature vector
        """
        features = np.zeros(self.feature_dimensions['statistical'], dtype=np.float32)
        
        # Token statistics
        tokens = code.split()
        if tokens:
            token_lengths = [len(t) for t in tokens]
            features[0] = np.mean(token_lengths)
            features[1] = np.std(token_lengths)
            features[2] = np.median(token_lengths)
            features[3] = np.max(token_lengths) if token_lengths else 0
            features[4] = np.min(token_lengths) if token_lengths else 0
        
        # Line statistics
        lines = code.split('\n')
        line_lengths = [len(line) for line in lines]
        if line_lengths:
            features[5] = np.mean(line_lengths)
            features[6] = np.std(line_lengths)
            features[7] = np.median(line_lengths)
            features[8] = np.max(line_lengths)
        
        # Entropy of character distribution
        char_counts = Counter(code)
        total_chars = len(code)
        if total_chars > 0:
            probabilities = [count/total_chars for count in char_counts.values()]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probabilities)
            features[9] = entropy
        
        # Compression ratio (approximation)
        unique_tokens = len(set(tokens))
        features[10] = unique_tokens / (len(tokens) + 1e-8)
        
        return features
    
    def extract_structural_features(self, graph_data: Dict) -> np.ndarray:
        """
        Extract structural features from graph representation
        
        Args:
            graph_data: Graph data dictionary
            
        Returns:
            Structural feature vector
        """
        features = np.zeros(self.feature_dimensions['structural'], dtype=np.float32)
        
        if not graph_data:
            return features
        
        # Graph statistics
        num_nodes = graph_data.get('num_nodes', 0)
        edge_index = graph_data.get('edge_index', np.array([[], []]))
        num_edges = edge_index.shape[1] if edge_index.size > 0 else 0
        
        features[0] = num_nodes
        features[1] = num_edges
        features[2] = num_edges / (num_nodes + 1e-8)  # Edge density
        
        # Degree statistics
        adj_list = graph_data.get('adjacency_list', {})
        if adj_list:
            degrees = [len(neighbors) for neighbors in adj_list.values()]
            if degrees:
                features[3] = np.mean(degrees)
                features[4] = np.std(degrees)
                features[5] = np.max(degrees)
                features[6] = np.min(degrees)
        
        # Connected components (approximation)
        features[7] = self._estimate_components(adj_list)
        
        # Clustering coefficient (local)
        features[8] = self._calculate_clustering_coefficient(adj_list)
        
        return features
    
    # Helper methods
    def _collect_node_types(self, node) -> List[str]:
        """Collect all node types in the AST"""
        types = []
        
        def traverse(n):
            if hasattr(n, 'type'):
                types.append(n.type)
            elif isinstance(n, dict) and 'type' in n:
                types.append(n['type'])
            
            children = []
            if hasattr(n, 'children'):
                children = n.children
            elif isinstance(n, dict) and 'children' in n:
                children = n['children']
            
            for child in children:
                traverse(child)
        
        traverse(node)
        return types
    
    def _get_tree_depth(self, node) -> int:
        """Get maximum depth of AST"""
        def get_depth(n):
            if not n:
                return 0
            
            children = []
            if hasattr(n, 'children'):
                children = n.children
            elif isinstance(n, dict) and 'children' in n:
                children = n['children']
            
            if not children:
                return 1
            
            return 1 + max(get_depth(child) for child in children)
        
        return get_depth(node)
    
    def _get_max_branching_factor(self, node) -> int:
        """Get maximum branching factor in AST"""
        max_branching = 0
        
        def traverse(n):
            nonlocal max_branching
            children = []
            if hasattr(n, 'children'):
                children = n.children
            elif isinstance(n, dict) and 'children' in n:
                children = n['children']
            
            max_branching = max(max_branching, len(children))
            for child in children:
                traverse(child)
        
        traverse(node)
        return max_branching
    
    def _calculate_tree_balance(self, node) -> float:
        """Calculate tree balance metric"""
        def get_subtree_size(n):
            if not n:
                return 0
            
            children = []
            if hasattr(n, 'children'):
                children = n.children
            elif isinstance(n, dict) and 'children' in n:
                children = n['children']
            
            if not children:
                return 1
            
            return 1 + sum(get_subtree_size(child) for child in children)
        
        def calculate_balance(n):
            children = []
            if hasattr(n, 'children'):
                children = n.children
            elif isinstance(n, dict) and 'children' in n:
                children = n['children']
            
            if not children:
                return 1.0
            
            subtree_sizes = [get_subtree_size(child) for child in children]
            if not subtree_sizes:
                return 1.0
            
            # Calculate variance in subtree sizes
            mean_size = np.mean(subtree_sizes)
            if mean_size == 0:
                return 1.0
            
            variance = np.var(subtree_sizes)
            # Normalize: lower variance means better balance
            return 1.0 / (1.0 + variance / (mean_size ** 2 + 1e-8))
        
        return calculate_balance(node)
    
    def _extract_functions(self, ast_node, code: str) -> List[Dict]:
        """Extract function definitions from AST"""
        functions = []
        
        def traverse(node):
            node_type = None
            if hasattr(node, 'type'):
                node_type = node.type
            elif isinstance(node, dict) and 'type' in node:
                node_type = node['type']
            
            # Check for function-like nodes
            if node_type and any(keyword in node_type.lower() for keyword in ['function', 'method', 'def']):
                func_info = {
                    'type': node_type,
                    'node': node,
                    'complexity': 1  # Base complexity
                }
                functions.append(func_info)
            
            children = []
            if hasattr(node, 'children'):
                children = node.children
            elif isinstance(node, dict) and 'children' in node:
                children = node['children']
            
            for child in children:
                traverse(child)
        
        if ast_node:
            traverse(ast_node)
        
        return functions
    
    def _calculate_complexity(self, function: Dict) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        def count_decision_points(node):
            nonlocal complexity
            
            node_type = None
            if hasattr(node, 'type'):
                node_type = node.type
            elif isinstance(node, dict) and 'type' in node:
                node_type = node['type']
            
            if node_type:
                # Decision point keywords
                decision_keywords = ['if', 'else', 'elif', 'for', 'while', 'case', 'catch', 'except']
                if any(keyword in node_type.lower() for keyword in decision_keywords):
                    complexity += 1
            
            children = []
            if hasattr(node, 'children'):
                children = node.children
            elif isinstance(node, dict) and 'children' in node:
                children = node['children']
            
            for child in children:
                count_decision_points(child)
        
        if 'node' in function:
            count_decision_points(function['node'])
        
        return complexity
    
    def _extract_variables(self, ast_node, code: str) -> List[str]:
        """Extract variable names from code"""
        variables = []
        
        # Simple regex-based extraction
        # This is language-agnostic but might not be perfect
        patterns = [
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=',  # Assignment
            r'\bvar\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # JavaScript var
            r'\blet\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # JavaScript let
            r'\bconst\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # JavaScript const
            r'\bint\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # C/C++/Java int
            r'\bfloat\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # C/C++/Java float
            r'\bdouble\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # C/C++/Java double
            r'\bString\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # Java String
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code)
            variables.extend(matches)
        
        return variables
    
    def _extract_imports(self, code: str, language: str) -> List[str]:
        """Extract import statements"""
        imports = []
        
        if language == 'python':
            imports.extend(re.findall(r'import\s+([\w.]+)', code))
            imports.extend(re.findall(r'from\s+([\w.]+)\s+import', code))
        elif language == 'java':
            imports.extend(re.findall(r'import\s+([\w.]+);', code))
        elif language == 'javascript':
            imports.extend(re.findall(r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]', code))
            imports.extend(re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', code))
        elif language == 'go':
            imports.extend(re.findall(r'import\s+"([^"]+)"', code))
        elif language in ['c', 'cpp', 'csharp']:
            imports.extend(re.findall(r'#include\s+[<"]([^>"]+)[>"]', code))
            imports.extend(re.findall(r'using\s+([\w.]+);', code))
        
        return imports
    
    def _estimate_components(self, adj_list: Dict) -> int:
        """Estimate number of connected components"""
        if not adj_list:
            return 0
        
        visited = set()
        components = 0
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in adj_list.get(node, []):
                dfs(neighbor)
        
        for node in adj_list:
            if node not in visited:
                components += 1
                dfs(node)
        
        return components
    
    def _calculate_clustering_coefficient(self, adj_list: Dict) -> float:
        """Calculate average clustering coefficient"""
        if not adj_list:
            return 0.0
        
        coefficients = []
        
        for node, neighbors in adj_list.items():
            if len(neighbors) < 2:
                coefficients.append(0.0)
                continue
            
            # Count edges between neighbors
            edges_between_neighbors = 0
            neighbor_list = list(neighbors)
            for i in range(len(neighbor_list)):
                for j in range(i + 1, len(neighbor_list)):
                    if neighbor_list[j] in adj_list.get(neighbor_list[i], []):
                        edges_between_neighbors += 1
            
            # Calculate clustering coefficient
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            if possible_edges > 0:
                coefficients.append(edges_between_neighbors / possible_edges)
            else:
                coefficients.append(0.0)
        
        return np.mean(coefficients) if coefficients else 0.0
    
    def _combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all feature types into single vector"""
        all_features = []
        
        for feat_type in ['lexical', 'syntactic', 'semantic', 'statistical', 'structural']:
            if feat_type in features:
                all_features.append(features[feat_type])
        
        if all_features:
            return np.concatenate(all_features)
        else:
            total_dim = sum(self.feature_dimensions.values())
            return np.zeros(total_dim, dtype=np.float32)