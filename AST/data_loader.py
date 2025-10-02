# data_loader.py
"""
Data loader module for loading SemEval-2026 Task 13 dataset from HuggingFace
"""

from datasets import load_dataset
import pandas as pd
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading data from HuggingFace datasets"""
    
    DATASET_NAME = "DaniilOr/SemEval-2026-Task13"
    
    # Task configurations
    TASK_CONFIGS = {
        'subtask_a': {
            'train_languages': ['cpp', 'python', 'java'],
            'test_languages': ['cpp', 'python', 'java', 'go', 'php', 'csharp', 'c', 'javascript'],
            'domains': ['algorithmic', 'research', 'production']
        },
        'subtask_b': {
            'generators': [
                'DeepSeek-AI', 'Qwen', '01-ai', 'BigCode', 'Gemma',
                'Phi', 'Meta-LLaMA', 'IBM-Granite', 'Mistral', 'OpenAI'
            ]
        },
        'subtask_c': {
            'labels': ['human', 'machine', 'hybrid', 'adversarial']
        }
    }
    
    def __init__(self, task='subtask_a', cache_dir='./cache'):
        """
        Initialize data loader
        
        Args:
            task: Task name (subtask_a, subtask_b, or subtask_c)
            cache_dir: Directory to cache downloaded data
        """
        self.task = task
        self.cache_dir = cache_dir
        self.dataset = None
        
    def load_data(self, split='train', limit=None) -> List[Dict]:
        """
        Load data from HuggingFace
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            limit: Limit number of samples to load
            
        Returns:
            List of data samples
        """
        logger.info(f"Loading {self.task} dataset from HuggingFace...")
        Task_mapping = {
            'subtask_a': "A",
            'subtask_b': "B",
            'subtask_c': "C"
        }
        
        try:
            # Load dataset from HuggingFace
            # The dataset structure might vary, so we'll try different approaches
            dataset = load_dataset(
                self.DATASET_NAME,
                Task_mapping.get(self.task, 'A'),
                split=split,
                cache_dir=self.cache_dir,
            )
        except Exception as e2:
            logger.error(f"Failed to load dataset: {e2}")
            return []
        
        # Convert to list of dictionaries
        data = []
        for idx, item in enumerate(dataset):
            if limit and idx >= limit:
                break
            
            sample = {
                'code': item.get('code', ''),
                'label': item.get('label', -1),
                'language': item.get('language', 'python'),
                'generator': item.get('generator', 'unknown'),
                'domain': item.get('domain', 'algorithmic')
            }
            
            # Add any additional fields
            for key in item.keys():
                if key not in sample:
                    sample[key] = item[key]
            
            data.append(sample)
        
        logger.info(f"Loaded {len(data)} samples from {split} split")
        return data
    
    def get_label_mapping(self) -> Dict:
        """
        Get label mapping for the current task
        
        Returns:
            Dictionary with label mappings
        """
        mappings = {
            'subtask_a': {
                'id_to_label': {0: 'human', 1: 'machine'},
                'label_to_id': {'human': 0, 'machine': 1}
            },
            'subtask_b': {
                'id_to_label': {
                    0: 'human',
                    1: 'DeepSeek-AI',
                    2: 'Qwen',
                    3: '01-ai',
                    4: 'BigCode',
                    5: 'Gemma',
                    6: 'Phi',
                    7: 'Meta-LLaMA',
                    8: 'IBM-Granite',
                    9: 'Mistral',
                    10: 'OpenAI'
                },
                'label_to_id': {
                    'human': 0,
                    'DeepSeek-AI': 1,
                    'Qwen': 2,
                    '01-ai': 3,
                    'BigCode': 4,
                    'Gemma': 5,
                    'Phi': 6,
                    'Meta-LLaMA': 7,
                    'IBM-Granite': 8,
                    'Mistral': 9,
                    'OpenAI': 10
                }
            },
            'subtask_c': {
                'id_to_label': {
                    0: 'human',
                    1: 'machine',
                    2: 'hybrid',
                    3: 'adversarial'
                },
                'label_to_id': {
                    'human': 0,
                    'machine': 1,
                    'hybrid': 2,
                    'adversarial': 3
                }
            }
        }
        
        return mappings.get(self.task, {})