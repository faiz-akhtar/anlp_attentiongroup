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
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ASTCodeDetector:
    """Main class for AST-based code detection pipeline"""
    
    def __init__(self, task='subtask_a', languages=None):
        self.task = task
        self.languages = languages or ['python', 'java', 'cpp']
        self.data_loader = DataLoader(task=task)
        self.ast_constructor = ASTConstructor()
        self.graph_converter = GraphConverter()
        self.feature_extractor = FeatureExtractor()
        
    def process_dataset(self, split='train', limit=None, save_path=None):
        """
        Process dataset and extract AST features
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            limit: Limit number of samples (for testing)
            save_path: Path to save processed data
        """
        logger.info(f"Processing {split} dataset for {self.task}")
        
        # Load data
        data = self.data_loader.load_data(split=split, limit=limit)
        logger.info(f"Loaded {len(data)} samples")
        
        processed_data = []
        failed_count = 0
        
        for idx, sample in tqdm.tqdm(enumerate(data), total=len(data)):
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
        
        logger.info(f"Successfully processed {len(processed_data)} samples")
        logger.info(f"Failed to process {failed_count} samples")
        
        # Save processed data if path provided
        if save_path:
            self.save_processed_data(processed_data, save_path)
        
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
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test'],
        help='Dataset split to process'
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
        '--save_path',
        type=str,
        default='./processed_data/ast_features',
        help='Path to save processed data'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ASTCodeDetector(
        task=args.task,
        languages=args.languages
    )
    
    # Process dataset
    processed_data = detector.process_dataset(
        split=args.split,
        limit=args.limit,
        save_path=f"{args.save_path}_{args.task}_{args.split}"
    )
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    main()