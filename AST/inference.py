# inference.py
"""
Inference utilities for trained GNN models
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from gnn_model import create_model
from ast_constructor import ASTConstructor
from graph_converter import GraphConverter
from feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class CodeDetectorInference:
    """
    Inference class for code detection using trained GNN models
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize inference engine
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        self.model = create_model(self.checkpoint['config']['model_config'])
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessing components
        self.ast_constructor = ASTConstructor()
        self.graph_converter = GraphConverter()
        self.feature_extractor = FeatureExtractor()
        
        # Get task and class labels
        self.task = self.checkpoint['config'].get('task', 'subtask_a')
        self.num_classes = self.checkpoint['config']['model_config']['num_classes']
        
        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Task: {self.task}, Classes: {self.num_classes}")
    
    def preprocess_code(
        self,
        code: str,
        language: str = 'python'
    ) -> Optional[Data]:
        """
        Preprocess code into graph format
        
        Args:
            code: Source code string
            language: Programming language
            
        Returns:
            PyTorch Geometric Data object or None if preprocessing fails
        """
        try:
            # Parse code to AST
            ast_dict = self.ast_constructor.parse_code(code, language)
            if ast_dict is None:
                logger.warning("Failed to parse code")
                return None
            
            # Convert to graph
            graph_data = self.graph_converter.ast_to_graph(ast_dict, language)
            if graph_data is None:
                logger.warning("Failed to convert AST to graph")
                return None
            
            # Extract features
            features = self.feature_extractor.extract_features(
                ast_dict,
                graph_data,
                code,
                language
            )
            if features is None:
                logger.warning("Failed to extract features")
                return None
            
            # Create PyTorch Geometric Data object
            node_features = graph_data['node_features']
            edge_index = graph_data['edge_index']
            edge_features = graph_data.get('edge_features', None)
            
            x = torch.FloatTensor(np.array(node_features))
            edge_index = torch.LongTensor(np.array(edge_index))
            edge_attr = None
            if edge_features is not None:
                edge_attr = torch.FloatTensor(np.array(edge_features))
            
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=graph_data['num_nodes']
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error preprocessing code: {e}")
            return None
    
    def predict_single(
        self,
        code: str,
        language: str = 'python',
        return_probabilities: bool = True
    ) -> Dict:
        """
        Predict on a single code sample
        
        Args:
            code: Source code string
            language: Programming language
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        data = self.preprocess_code(code, language)
        if data is None:
            return {
                'success': False,
                'error': 'Failed to preprocess code'
            }
        
        # Move to device
        data = data.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(
                data.x,
                data.edge_index,
                data.edge_attr if hasattr(data, 'edge_attr') else None,
                None  # batch is None for single sample
            )
            
            # Get probabilities
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = int(output.argmax(dim=1).cpu().numpy()[0])
        
        result = {
            'success': True,
            'prediction': pred_class,
            'confidence': float(probs[pred_class]),
            'language': language
        }
        
        if return_probabilities:
            result['probabilities'] = probs.tolist()
        
        # Add class labels
        result['predicted_label'] = self._get_class_label(pred_class)
        
        return result
    
    def predict_batch(
        self,
        codes: List[str],
        languages: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict on a batch of code samples
        
        Args:
            codes: List of source code strings
            languages: List of programming languages (defaults to 'python' for all)
            batch_size: Batch size for inference
            
        Returns:
            List of prediction dictionaries
        """
        if languages is None:
            languages = ['python'] * len(codes)
        
        if len(codes) != len(languages):
            raise ValueError("Number of codes and languages must match")
        
        results = []
        
        # Preprocess all samples
        data_list = []
        valid_indices = []
        
        for idx, (code, lang) in enumerate(tqdm(zip(codes, languages), desc="Preprocessing")):
            data = self.preprocess_code(code, lang)
            if data is not None:
                data_list.append(data)
                valid_indices.append(idx)
            else:
                results.append({
                    'success': False,
                    'error': 'Failed to preprocess code',
                    'index': idx
                })
        
        # Batch predict
        for i in tqdm(range(0, len(data_list), batch_size), desc="Predicting"):
            batch_data = data_list[i:i+batch_size]
            batch = Batch.from_data_list(batch_data).to(self.device)
            
            with torch.no_grad():
                output = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr if hasattr(batch, 'edge_attr') else None,
                    batch.batch
                )
                
                probs = F.softmax(output, dim=1).cpu().numpy()
                preds = output.argmax(dim=1).cpu().numpy()
            
            # Store results
            for j, (pred, prob) in enumerate(zip(preds, probs)):
                orig_idx = valid_indices[i + j]
                results.append({
                    'success': True,
                    'prediction': int(pred),
                    'confidence': float(prob[pred]),
                    'probabilities': prob.tolist(),
                    'predicted_label': self._get_class_label(int(pred)),
                    'language': languages[orig_idx],
                    'index': orig_idx
                })
        
        # Sort by original index
        results.sort(key=lambda x: x.get('index', -1))
        
        return results
    
    def _get_class_label(self, class_idx: int) -> str:
        """
        Get human-readable class label
        
        Args:
            class_idx: Class index
            
        Returns:
            Class label string
        """
        if self.task == 'subtask_a':
            labels = {0: 'human', 1: 'machine'}
        elif self.task == 'subtask_b':
            labels = {
                0: 'human',
                1: 'gpt-3.5-turbo',
                2: 'gpt-4',
                3: 'claude',
                4: 'copilot',
                5: 'codex',
                6: 'starcoder',
                7: 'wizardcoder',
                8: 'phi-2',
                9: 'deepseek',
                10: 'other'
            }
        elif self.task == 'subtask_c':
            labels = {
                0: 'human',
                1: 'machine',
                2: 'hybrid',
                3: 'adversarial'
            }
        else:
            labels = {i: f'class_{i}' for i in range(self.num_classes)}
        
        return labels.get(class_idx, f'unknown_{class_idx}')
    
    def predict_file(
        self,
        file_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Predict on a code file
        
        Args:
            file_path: Path to code file
            language: Programming language (auto-detected if None)
            
        Returns:
            Prediction dictionary
        """
        file_path = Path(file_path)
        
        # Auto-detect language from extension
        if language is None:
            ext_to_lang = {
                '.py': 'python',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.js': 'javascript',
                '.ts': 'typescript'
            }
            language = ext_to_lang.get(file_path.suffix, 'python')
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to read file: {e}'
            }
        
        # Predict
        result = self.predict_single(code, language)
        result['file_path'] = str(file_path)
        
        return result
    
    def predict_directory(
        self,
        directory_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Dict]:
        """
        Predict on all code files in a directory
        
        Args:
            directory_path: Path to directory
            extensions: File extensions to process (defaults to common code extensions)
            recursive: Whether to search recursively
            
        Returns:
            List of prediction dictionaries
        """
        if extensions is None:
            extensions = ['.py', '.java', '.cpp', '.c', '.js', '.ts']
        
        directory_path = Path(directory_path)
        
        # Find all code files
        if recursive:
            files = []
            for ext in extensions:
                files.extend(directory_path.rglob(f'*{ext}'))
        else:
            files = []
            for ext in extensions:
                files.extend(directory_path.glob(f'*{ext}'))
        
        logger.info(f"Found {len(files)} code files")
        
        # Predict on each file
        results = []
        for file_path in tqdm(files, desc="Processing files"):
            result = self.predict_file(str(file_path))
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Save prediction results to JSON file
        
        Args:
            results: List of prediction dictionaries
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with trained GNN model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--code', type=str, help='Code string to predict')
    parser.add_argument('--file', type=str, help='Path to code file')
    parser.add_argument('--directory', type=str, help='Path to directory of code files')
    parser.add_argument('--language', type=str, default='python', help='Programming language')
    parser.add_argument('--output', type=str, help='Path to save results (for directory mode)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    detector = CodeDetectorInference(args.checkpoint)
    
    if args.code:
        # Predict on code string
        result = detector.predict_single(args.code, args.language)
        print(json.dumps(result, indent=2))
    
    elif args.file:
        # Predict on file
        result = detector.predict_file(args.file, args.language)
        print(json.dumps(result, indent=2))
    
    elif args.directory:
        # Predict on directory
        results = detector.predict_directory(args.directory)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        print(f"\nProcessed {len(results)} files ({successful} successful)")
        
        if args.output:
            detector.save_results(results, args.output)
    
    else:
        print("Please provide --code, --file, or --directory")


if __name__ == '__main__':
    main()