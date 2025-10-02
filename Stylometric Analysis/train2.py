import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
import warnings
import argparse
from io import StringIO
from collections import Counter

warnings.filterwarnings('ignore')

# Import feature extractors
from FeatureExtractor import FeatureExtractor
from FormattingMicroSignatureExtractor import FormattingMicroSignatureExtractor
from CommentParser import CommentFeatureProcessor
from VariableNamingEntropyExtractor import VariableNamingEntropyExtractor

# Global constants for data paths
SUBTASK_A_PATHS = {
    'train': '../SemEval-2026-Task13/task_a/task_a_training_set_1.parquet',
    'validation': '../SemEval-2026-Task13/task_a/task_a_validation_set.parquet',
    'test': '../SemEval-2026-Task13/task_a/task_a_test_set_sample.parquet'
}

SUBTASK_B_PATHS = {
    'train': '../SemEval-2026-Task13/task_b/task_b_training_set.parquet',
    'validation': '../SemEval-2026-Task13/task_b/task_b_validation_set.parquet',
    'test': '../SemEval-2026-Task13/task_b/task_b_test_set_sample.parquet'
}


class WeightedLoss(nn.Module):
    """Weighted CrossEntropyLoss for equal group importance."""
    def __init__(self, group_weights):
        super(WeightedLoss, self).__init__()
        self.group_weights = group_weights
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, outputs, targets, groups):
        losses = self.ce_loss(outputs, targets)
        weights = torch.tensor([self.group_weights[g.item()] for g in groups], 
                              device=outputs.device, dtype=torch.float32)
        weighted_loss = (losses * weights).mean()
        return weighted_loss


class LLMDetectionPipeline:
    """Complete pipeline for LLM-generated code detection."""
    
    def __init__(self, subtask: str, batch_size: int = 1000, feature_batch_size: int = 100,
                 pca_batch_size: int = 5000, random_state: int = 42, strategy: str = 'proportional'):
        self.subtask = subtask
        self.paths = SUBTASK_A_PATHS if subtask == 'a' else SUBTASK_B_PATHS
        self.batch_size = batch_size
        self.feature_batch_size = feature_batch_size
        self.pca_batch_size = pca_batch_size
        self.random_state = random_state
        self.strategy = strategy
        
        self.formatting_extractor = FormattingMicroSignatureExtractor()
        self.comment_processor = CommentFeatureProcessor()
        self.variable_extractor = VariableNamingEntropyExtractor()
        
        self.scaler = StandardScaler()
        self.pca = None
        self.model = None
        self.criterion = None
        self.group_weights = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_cols = None
        
        print(f"Using device: {self.device}")
        print(f"Sampling strategy: {self.strategy}")
    
    def load_stratified_sample(self, n_samples: int, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print(f"Loading data for Subtask {self.subtask.upper()}...")
        df_train_full = pd.read_parquet(self.paths['train'])
        df_val_full = pd.read_parquet(self.paths['validation'])
        
        print(f"Full training set: {len(df_train_full)} samples")
        print(f"Full validation set: {len(df_val_full)} samples")
        
        if n_samples == -1 or n_samples >= len(df_train_full):
            print(f"Using the entire training dataset of {len(df_train_full)} samples.")
            df_train = df_train_full
        else:
            print(f"Sampling {n_samples} training samples using {self.strategy} strategy...")
            df_train = self._sample_data(df_train_full, n_samples, is_train=True)
        
        val_samples = int(len(df_train) * 0.2)
        if val_samples >= len(df_val_full):
            print(f"Using the entire validation dataset of {len(df_val_full)} samples.")
            df_val = df_val_full
        else:
            print(f"Sampling {val_samples} validation samples...")
            df_val = self._sample_data(df_val_full, val_samples, is_train=False)
        
        df_train = df_train.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        df_val = df_val.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        self._save_sampling_stats(df_train_full, df_val_full, df_train, df_val, output_dir)
        
        return df_train, df_val
    
    def _sample_data(self, df: pd.DataFrame, n_samples: int, is_train: bool) -> pd.DataFrame:
        df['stratify_col'] = df['label'].astype(str) + '_' + df['language'].astype(str)
        
        # Handle cases where a stratification group has only one member, which breaks train_test_split
        group_counts = df['stratify_col'].value_counts()
        single_member_groups = group_counts[group_counts == 1].index
        
        if not single_member_groups.empty:
            num_removed = len(single_member_groups)
            print(f"Warning: Found {num_removed} stratification groups with only 1 member. "
                  f"These {num_removed} samples will be excluded from the dataset to allow for stable stratification.")
            df = df[~df['stratify_col'].isin(single_member_groups)]

        if self.strategy == 'proportional' or (self.strategy == 'weighted' and not is_train):
            df_sample, _ = train_test_split(df, train_size=n_samples, stratify=df['stratify_col'],
                                           random_state=self.random_state)
        elif self.strategy == 'equitable' and is_train:
            # Re-calculate group_counts on the potentially filtered dataframe
            group_counts = df['stratify_col'].value_counts()
            min_group_size = group_counts.min()
            n_groups = len(group_counts)
            samples_per_group = n_samples // n_groups
            
            if samples_per_group > min_group_size:
                raise ValueError(f"Cannot sample {samples_per_group} samples per group. "
                               f"Smallest group has only {min_group_size} samples. "
                               "Try reducing n_samples or using the 'proportional' strategy.")
            
            sampled_dfs = []
            for group in group_counts.index:
                group_df = df[df['stratify_col'] == group]
                sampled_group = group_df.sample(n=samples_per_group, random_state=self.random_state)
                sampled_dfs.append(sampled_group)
            df_sample = pd.concat(sampled_dfs, ignore_index=True)
        else: # This will now catch the 'weighted' strategy for training data
            df_sample, _ = train_test_split(df, train_size=n_samples, stratify=df['stratify_col'],
                                           random_state=self.random_state)
        
        return df_sample.drop(columns=['stratify_col'])
    
    def _save_sampling_stats(self, df_train_full, df_val_full, df_train, df_val, output_dir):
        summary_buffer = StringIO()
        summary_buffer.write("="*70 + "\n")
        summary_buffer.write(f"SUBTASK {self.subtask.upper()} - SAMPLING SUMMARY ({self.strategy.upper()} STRATEGY)\n")
        summary_buffer.write("="*70 + "\n")
        summary_buffer.write(f"Training samples: {len(df_train)} (from {len(df_train_full)} available)\n")
        summary_buffer.write(f"Validation samples: {len(df_val)} (from {len(df_val_full)} available)\n\n")
        
        summary_buffer.write("--- Sampled Training Dataset Distribution ---\n")
        summary_buffer.write("Label distribution:\n")
        summary_buffer.write(df_train['label'].value_counts(normalize=True).to_string() + "\n\n")
        summary_buffer.write("[Label, Language] distribution:\n")
        summary_buffer.write(df_train.groupby('label')['language'].value_counts(normalize=True).to_string() + "\n\n")
        
        summary_text = summary_buffer.getvalue()
        print(summary_text)
        
        stats_path = os.path.join(output_dir, 'sampling_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(summary_text)
        print(f"Sampling statistics saved to {stats_path}")
    
    def extract_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Extracting features in batches...")
        all_features = []
        total_batches = (len(df) + self.feature_batch_size - 1) // self.feature_batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.feature_batch_size
            end_idx = min((batch_idx + 1) * self.feature_batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} (samples {start_idx}-{end_idx})...")
            
            for idx, row in batch_df.iterrows():
                try:
                    features = {}
                    style_extractor = FeatureExtractor(row['code'], row['language'])
                    style_features = style_extractor.extract_all_features()
                    features.update({f'style_{k}': v for k, v in style_features.items()})
                    
                    format_features = self.formatting_extractor.extract_all_signatures(row['code'], row['language'])
                    features.update({f'format_{k}': v for k, v in format_features.items()})
                    
                    comment_features = self.comment_processor.extractor.extract_features(row['code'], row['language'])
                    features.update({f'comment_{k}': v for k, v in comment_features.items()})
                    
                    var_features = self.variable_extractor.extract_all_features(row['code'], row['language'])
                    features.update({f'var_{k}': v for k, v in var_features.items()})
                    
                    features['label'] = row['label']
                    features['language'] = row['language']
                    all_features.append(features)
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
        
        features_df = pd.DataFrame(all_features)
        print(f"Feature extraction complete. Shape: {features_df.shape}")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        return features_df
    
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple:
        print("Preparing data for training...")
        train_df['group'] = train_df['label'].astype(str) + '_' + train_df['language'].astype(str)
        val_df['group'] = val_df['label'].astype(str) + '_' + val_df['language'].astype(str)
        
        self.feature_cols = [col for col in train_df.columns if col not in ['label', 'language', 'group']]
        
        # Align columns
        missing_cols = [col for col in self.feature_cols if col not in val_df.columns]
        if missing_cols:
            print(f"Adding {len(missing_cols)} missing columns to validation set")
            for col in missing_cols:
                val_df[col] = 0.0
        
        extra_cols = [col for col in val_df.columns if col not in self.feature_cols + ['label', 'language', 'group']]
        if extra_cols:
            print(f"Removing {len(extra_cols)} extra columns from validation set")
            val_df = val_df.drop(columns=extra_cols)
        
        X_train = train_df[self.feature_cols].values
        y_train = train_df['label'].values
        train_groups = train_df['group'].values
        
        X_val = val_df[self.feature_cols].values
        y_val = val_df['label'].values
        val_groups = val_df['group'].values
        
        if self.strategy == 'weighted':
            group_counts = Counter(train_groups)
            total_samples = len(train_groups)
            n_groups = len(group_counts)
            self.group_weights = {group: (total_samples / n_groups) / count 
                                 for group, count in group_counts.items()}
            print(f"Calculated weights for {len(self.group_weights)} groups")
        
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
        return X_train, X_val, y_train, y_val, train_groups, val_groups
    
    def apply_pca(self, X_train: np.ndarray, X_val: np.ndarray, explained_variance: float = 0.95) -> Tuple:
        print(f"\nApplying PCA (target variance: {explained_variance})...")
        print(f"Features before PCA: {X_train.shape[1]}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_components = min(X_train.shape[0], X_train.shape[1])
        self.pca = IncrementalPCA(n_components=n_components, batch_size=self.pca_batch_size)
        
        n_batches = (len(X_train_scaled) + self.pca_batch_size - 1) // self.pca_batch_size
        for i in range(n_batches):
            start_idx = i * self.pca_batch_size
            end_idx = min((i + 1) * self.pca_batch_size, len(X_train_scaled))
            self.pca.partial_fit(X_train_scaled[start_idx:end_idx])
        
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components_final = np.argmax(cumsum_variance >= explained_variance) + 1
        
        print(f"Using {n_components_final} components (explained variance: {cumsum_variance[n_components_final-1]:.4f})")
        
        self.pca = IncrementalPCA(n_components=n_components_final, batch_size=self.pca_batch_size)
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        
        print(f"PCA complete. Final shape: {X_train_pca.shape}\n")
        return X_train_pca, X_val_pca
    
    def build_model(self, input_dim: int, num_classes: int, hidden_dims: List[int] = [256, 128, 64]) -> nn.Module:
        class FeedForwardNN(nn.Module):
            def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
                super(FeedForwardNN, self).__init__()
                layers = []
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    layers.extend([nn.Linear(prev_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(), nn.Dropout(dropout)])
                    prev_dim = hidden_dim
                layers.append(nn.Linear(prev_dim, num_classes))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return FeedForwardNN(input_dim, hidden_dims, num_classes).to(self.device)
    
    def train_model(self, X_train, y_train, X_val, y_val, train_groups, val_groups,
                   epochs, lr, patience, output_dir) -> Dict:
        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
        print(f"Training model for up to {epochs} epochs...")
        
        num_classes = len(np.unique(y_train))
        self.model = self.build_model(X_train.shape[1], num_classes)
        
        unique_groups = list(set(train_groups) | set(val_groups))
        group_to_idx = {group: idx for idx, group in enumerate(unique_groups)}
        train_group_indices = np.array([group_to_idx[g] for g in train_groups])
        val_group_indices = np.array([group_to_idx[g] for g in val_groups])
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train),
                                     torch.LongTensor(train_group_indices))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val),
                                   torch.LongTensor(val_group_indices))
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        if self.strategy == 'weighted':
            idx_to_group = {idx: group for group, idx in group_to_idx.items()}
            indexed_weights = {idx: self.group_weights.get(idx_to_group[idx], 1.0) 
                             for idx in range(len(unique_groups))}
            self.criterion = WeightedLoss(indexed_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                         patience=5, verbose=True)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for batch_X, batch_y, batch_groups in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_groups = batch_groups.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.strategy == 'weighted':
                    loss = self.criterion(outputs, batch_y, batch_groups)
                else:
                    loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            
            with torch.no_grad():
                for batch_X, batch_y, batch_groups in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    batch_groups = batch_groups.to(self.device)
                    
                    outputs = self.model(batch_X)
                    
                    if self.strategy == 'weighted':
                        loss = self.criterion(outputs, batch_y, batch_groups)
                    else:
                        loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(val_accuracy)
            
            scheduler.step(val_accuracy)
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state': self.scaler,
                    'pca_state': self.pca,
                    'feature_cols': self.feature_cols,
                    'val_loss': avg_val_loss,
                    'val_acc': val_accuracy,
                    'strategy': self.strategy
                }, checkpoint_path)
                print(f"Model checkpoint saved (val_acc: {val_accuracy:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("\nTraining complete!")
        return history
    
    def plot_training_history(self, history: Dict, output_dir: str):
        save_path = os.path.join(output_dir, 'training_plot.png')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
        plt.close()
    
    def evaluate_model(self, X_val, y_val, output_dir) -> Dict:
        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
        print(f"Evaluating model on validation set using checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
        
        report_buffer = StringIO()
        report_buffer.write("="*50 + "\nVALIDATION EVALUATION METRICS\n" + "="*50 + "\n")
        report_buffer.write(f"Accuracy:  {accuracy:.4f}\n")
        report_buffer.write(f"Precision: {precision:.4f}\n")
        report_buffer.write(f"Recall:    {recall:.4f}\n")
        report_buffer.write(f"F1 Score:  {f1:.4f}\n" + "="*50 + "\n")
        report_buffer.write("\nDetailed Classification Report:\n")
        report_buffer.write(classification_report(all_labels, all_preds, zero_division=0) + "\n")
        
        report_text = report_buffer.getvalue()
        print(report_text)
        
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Evaluation report saved to {report_path}")
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM Code Detection Pipeline")
    parser.add_argument('--subtask', type=str, required=True, choices=['a', 'b'])
    parser.add_argument('--n_samples', type=int, default=100000)
    parser.add_argument('--strategy', type=str, default='proportional',
                        choices=['proportional', 'equitable', 'weighted'])
    args = parser.parse_args()

    n_samples_str = "full" if args.n_samples == -1 else str(args.n_samples)
    output_dir = f"{n_samples_str}_subtask_{args.subtask}_{args.strategy}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {output_dir}")

    CONFIG = {
        'subtask': args.subtask, 'n_samples': args.n_samples, 'strategy': args.strategy,
        'output_dir': output_dir, 'explained_variance': 0.95, 'epochs': 100,
        'learning_rate': 0.001, 'patience': 20, 'batch_size': 1024,
        'feature_batch_size': 10000, 'pca_batch_size': 5000,
        'hidden_dims': [256, 128, 64], 'random_state': 42
    }
    
    pipeline = LLMDetectionPipeline(
        subtask=CONFIG['subtask'], batch_size=CONFIG['batch_size'],
        feature_batch_size=CONFIG['feature_batch_size'],
        pca_batch_size=CONFIG['pca_batch_size'],
        random_state=CONFIG['random_state'], strategy=CONFIG['strategy']
    )
    
    df_train, df_val = pipeline.load_stratified_sample(CONFIG['n_samples'], CONFIG['output_dir'])
    train_features_df = pipeline.extract_features_batch(df_train)
    val_features_df = pipeline.extract_features_batch(df_val)
    
    X_train, X_val, y_train, y_val, train_groups, val_groups = pipeline.prepare_data(
        train_features_df, val_features_df
    )
    
    X_train_pca, X_val_pca = pipeline.apply_pca(X_train, X_val, 
                                                 explained_variance=CONFIG['explained_variance'])
    
    history = pipeline.train_model(X_train_pca, y_train, X_val_pca, y_val,
                                   train_groups, val_groups, epochs=CONFIG['epochs'],
                                   lr=CONFIG['learning_rate'], patience=CONFIG['patience'],
                                   output_dir=CONFIG['output_dir'])
    
    pipeline.plot_training_history(history, CONFIG['output_dir'])
    metrics = pipeline.evaluate_model(X_val_pca, y_val, CONFIG['output_dir'])
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*50)
    print(f"All artifacts saved in directory: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()