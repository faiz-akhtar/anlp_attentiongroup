# ast_constructor.py
"""
AST Constructor module using tree-sitter for multiple programming languages
"""

import tree_sitter
from tree_sitter import Node
from tree_sitter_languages import get_language
import ast as _py_ast

# Flag to detect incompatible tree_sitter_languages / tree_sitter API at runtime
TREE_SITTER_BAD_API = False
import logging
from typing import Optional, Dict, Any
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class ASTConstructor:
    """Constructs Abstract Syntax Trees using tree-sitter"""
    
    # Language to tree-sitter grammar mapping
    LANGUAGE_GRAMMARS = [
        'python',
        'java',
        'cpp',  # Note: 'cpp' is the correct name in tree-sitter-languages
        'c',
        'javascript',
        'go',
        'php',
        'csharp'
    ]
    
    def __init__(self):
        """
        Initialize AST Constructor
        
        Args:
            grammar_dir: Directory to store tree-sitter grammar files
        """
        self.parsers = {}
        self.languages = {}
        
        # Setup parsers for all languages
        self._setup_parsers()
    
    def _setup_parsers(self):
        """Setup tree-sitter parsers for all supported languages"""
        for lang in self.LANGUAGE_GRAMMARS:
            try:
                parser = self._get_parser(lang)
                if parser:
                    self.parsers[lang] = parser
                    logger.info(f"Successfully loaded parser for {lang}")
            except Exception as e:
                logger.warning(f"Failed to load parser for {lang}: {e}")
    
    def _get_parser(self, language: str):
        """
        Get or create parser for a language using pre-built tree-sitter libraries
        
        Args:
            language: Programming language name
            
        Returns:
            Parser object or None if failed
        """
        try:
            # Handle languages with different names in tree-sitter-languages
            language_mapping = {
                'csharp': 'c_sharp'
            }
            
            lang_name = language_mapping.get(language, language)
            
            # Try to obtain the language object first
            lang_obj = None
            try:
                lang_obj = get_language(lang_name)
            except Exception as e:
                raise RuntimeError(f"get_language failed for {lang_name}: {e}")

            from tree_sitter import Parser as TSParser
            parser = TSParser()
            parser.set_language(lang_obj)
            self.languages[language] = lang_obj
            return parser
            
        except Exception as e:
            logger.error(f"Error setting up parser for {language}: {e}")
            return None
    
    def parse_code(self, code: str, language: str) -> Optional[Dict[str, Any]]:
        """
        Parse code and return AST as serializable dictionary
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            AST as dictionary or None if parsing failed
        """
        # If parser is available, use it. Otherwise provide language-specific
        # fallbacks (currently Python via ast).
        parser = self.parsers.get(language)
        if parser is None:
           print(f"No parser available for {language}")
           return None
        try:
            tree = parser.parse(bytes(code, 'utf8'))

            # Convert to serializable dictionary immediately
            ast_dict = self.node_to_dict(tree.root_node, bytes(code, 'utf8'))
            return ast_dict

        except Exception as e:
            logger.error(f"Error parsing code for {language}: {e}")
            return None
    
    def node_to_dict(self, node: Node, code_bytes: bytes = None) -> Dict[str, Any]:
        """
        Convert tree-sitter node to dictionary representation
        
        Args:
            node: Tree-sitter node
            code_bytes: Original code as bytes
            
        Returns:
            Dictionary representation of the node
        """
        node_dict = {
            'type': node.type,
            'start_point': node.start_point,
            'end_point': node.end_point,
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'children': []
        }
        
        # Add text content for leaf nodes
        if not node.children and code_bytes:
            node_dict['text'] = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
        
        # Recursively process children
        for child in node.children:
            child_dict = self.node_to_dict(child, code_bytes)
            node_dict['children'].append(child_dict)
        
        return node_dict
    
    def get_ast_statistics(self, ast_dict: Dict[str, Any]) -> Dict[str, int]:
        """
        Get statistics about the AST
        
        Args:
            ast_dict: AST dictionary
            
        Returns:
            Dictionary with AST statistics
        """
        stats = {
            'total_nodes': 0,
            'max_depth': 0,
            'node_types': {}
        }
        
        def traverse(node, depth=0):
            stats['total_nodes'] += 1
            stats['max_depth'] = max(stats['max_depth'], depth)
            
            node_type = node['type']
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            
            for child in node['children']:
                traverse(child, depth + 1)
        
        traverse(ast_dict)
        return stats
    
    def extract_node_sequence(self, ast_dict: Dict[str, Any]) -> list:
        """
        Extract sequence of node types (pre-order traversal)
        
        Args:
            ast_dict: AST dictionary
            
        Returns:
            List of node types in pre-order
        """
        sequence = []
        
        def traverse(node):
            sequence.append(node['type'])
            for child in node['children']:
                traverse(child)
        
        traverse(ast_dict)
        return sequence
    
    def extract_paths(self, ast_dict: Dict[str, Any], max_length: int = 8) -> list:
        """
        Extract paths from leaves to root (for path-based representations)
        
        Args:
            ast_dict: AST dictionary
            max_length: Maximum path length
            
        Returns:
            List of paths
        """
        paths = []
        
        def traverse(node, current_path):
            current_path = current_path + [node['type']]
            
            if not node['children']:  # Leaf node
                # Reverse to get leaf-to-root path
                path = current_path[::-1][:max_length]
                paths.append(path)
            else:
                for child in node['children']:
                    traverse(child, current_path)
        
        traverse(ast_dict, [])
        return paths
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages with working parsers"""
        return list(self.parsers.keys())