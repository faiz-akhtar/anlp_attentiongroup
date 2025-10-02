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

warnings.filterwarnings('ignore')

# Import feature extractors
# Make sure these files (FeatureExtractor.py, etc.) are in the same directory
from FeatureExtractor import FeatureExtractor
from FormattingMicroSignatureExtractor import FormattingMicroSignatureExtractor
from CommentParser import CommentFeatureProcessor
from VariableNamingEntropyExtractor import VariableNamingEntropyExtractor

# Global constants for data paths
data_path_a = '../SemEval-2026-Task13/task_a/task_a_training_set_1.parquet'
data_path_b = '../SemEval-2026-Task13/task_b/task_b_training_set.parquet'


class LLMDetectionPipeline:
    """
    Complete pipeline for LLM-generated code detection with batch processing
    to handle large datasets efficiently.
    """
    
    def __init__(self, 
                 data_path: str,
                 batch_size: int = 1000,
                 feature_batch_size: int = 100,
                 pca_batch_size: int = 5000,
                 random_state: int = 42):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to the parquet file
            batch_size: Batch size for training
            feature_batch_size: Batch size for feature extraction
            pca_batch_size: Batch size for PCA fitting
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_batch_size = feature_batch_size
        self.pca_batch_size = pca_batch_size
        self.random_state = random_state
        
        # Feature extractors
        self.formatting_extractor = FormattingMicroSignatureExtractor()
        self.comment_processor = CommentFeatureProcessor()
        self.variable_extractor = VariableNamingEntropyExtractor()
        
        # Scalers and PCA
        self.scaler = StandardScaler()
        self.pca = None
        
        # Model
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
    
    def load_proportional_sample(self, n_samples: int, output_dir: str) -> pd.DataFrame:
        """
        Load a proportionally stratified sample from the dataset.
        This stratification ensures that each [label, generator] combination
        is represented in the sample in proportion to its presence in the full dataset.
        If n_samples is -1, the entire dataset is used.
        
        Args:
            n_samples: Total number of samples to load. -1 for the full dataset.
            output_dir: Directory to save sampling statistics.
            
        Returns:
            Proportionally stratified sample DataFrame
        """
        print("Loading full dataset for sampling...")
        df = pd.read_parquet(self.data_path)
        
        if n_samples == -1 or n_samples >= len(df):
            print(f"Using the entire dataset of {len(df)} samples.")
            df_sample = df
        else:
            print(f"Loading proportional sample of {n_samples} samples from {len(df)} total...")
            
            # Create a combined column for stratification
            df['stratify_col'] = df['label'].astype(str) + '_' + df['generator'].astype(str)
            
            # Use train_test_split to get a stratified sample
            # We only care about the 'train' part of the split, which is our sample.
            df_sample, _ = train_test_split(
                df,
                train_size=n_samples,
                stratify=df['stratify_col'],
                random_state=self.random_state
            )
            df_sample = df_sample.drop(columns=['stratify_col'])

        df_sample = df_sample.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Prepare summary for printing and saving
        summary_buffer = StringIO()
        summary_buffer.write("="*50 + "\n")
        summary_buffer.write("PROPORTIONAL SAMPLING SUMMARY\n")
        summary_buffer.write("="*50 + "\n")
        summary_buffer.write(f"Loaded a total of {len(df_sample)} samples.\n")
        
        summary_buffer.write("\n--- Full Dataset Distribution ---\n")
        summary_buffer.write("Label distribution:\n")
        summary_buffer.write(df['label'].value_counts(normalize=True).to_string() + "\n\n")
        summary_buffer.write("[Label, Generator] distribution:\n")
        summary_buffer.write(df.groupby('label')['generator'].value_counts(normalize=True).to_string() + "\n")
        
        summary_buffer.write("\n--- Sampled Dataset Distribution ---\n")
        summary_buffer.write("Label distribution:\n")
        summary_buffer.write(df_sample['label'].value_counts(normalize=True).to_string() + "\n\n")
        summary_buffer.write("[Label, Generator] distribution:\n")
        summary_buffer.write(df_sample.groupby('label')['generator'].value_counts(normalize=True).to_string() + "\n")
        summary_buffer.write("="*50 + "\n")

        summary_text = summary_buffer.getvalue()
        print(summary_text)

        # Save summary to file
        stats_path = os.path.join(output_dir, 'sampling_stats.txt')
        with open(stats_path, 'w') as f:
            f.write(summary_text)
        print(f"Sampling statistics saved to {stats_path}")
        
        return df_sample
    
    def extract_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from code samples in batches.
        
        Args:
            df: DataFrame with code samples
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting features in batches...")
        
        all_features = []
        total_batches = (len(df) + self.feature_batch_size - 1) // self.feature_batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.feature_batch_size
            end_idx = min((batch_idx + 1) * self.feature_batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} "
                  f"(samples {start_idx}-{end_idx})...")
            
            batch_features = []
            for idx, row in batch_df.iterrows():
                try:
                    features = {}
                    
                    # 1. Stylometric features
                    style_extractor = FeatureExtractor(row['code'], row['language'])
                    style_features = style_extractor.extract_all_features()
                    features.update({f'style_{k}': v for k, v in style_features.items()})
                    
                    # 2. Formatting micro-signatures
                    format_features = self.formatting_extractor.extract_all_signatures(
                        row['code'], row['language']
                    )
                    features.update({f'format_{k}': v for k, v in format_features.items()})
                    
                    # 3. Comment features
                    comment_features = self.comment_processor.extractor.extract_features(
                        row['code'], row['language']
                    )
                    features.update({f'comment_{k}': v for k, v in comment_features.items()})
                    
                    # 4. Variable naming entropy
                    var_features = self.variable_extractor.extract_all_features(
                        row['code'], row['language']
                    )
                    features.update({f'var_{k}': v for k, v in var_features.items()})
                    
                    # Add metadata
                    features['label'] = row['label']
                    features['generator'] = row['generator']
                    features['language'] = row['language']
                    
                    batch_features.append(features)
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
            
            all_features.extend(batch_features)
        
        features_df = pd.DataFrame(all_features)
        print(f"Feature extraction complete. Shape: {features_df.shape}")
        
        # Handle missing values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        return features_df
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
            """
            Split data into train and validation sets, stratified by [label, generator].
            Handles cases where some strata have only one member by adding them to the training set.
            
            Args:
                df: Features DataFrame
                test_size: Proportion of data for validation
                
            Returns:
                X_train, X_val, y_train, y_val
            """
            print(f"Splitting data (test_size={test_size}) with proportional stratification...")
            
            # Create a combined column for stratification
            df['stratify_col'] = df['label'].astype(str) + '_' + df['generator'].astype(str)
            
            # Identify strata counts
            strata_counts = df['stratify_col'].value_counts()
            
            # Identify strata with only one member
            single_member_strata = strata_counts[strata_counts == 1].index
            
            if len(single_member_strata) > 0:
                print(f"Found {len(single_member_strata)} strata with only 1 member each. "
                    f"These {len(single_member_strata)} samples will be added exclusively to the training set.")

            # Separate stratifiable data from single-member data
            df_stratifiable = df[~df['stratify_col'].isin(single_member_strata)]
            df_single_member = df[df['stratify_col'].isin(single_member_strata)]
            
            # Define feature columns
            feature_cols = [col for col in df.columns 
                        if col not in ['label', 'generator', 'language', 'stratify_col']]
            
            # Handle the stratifiable part
            if not df_stratifiable.empty:
                X_strat = df_stratifiable[feature_cols].values
                y_strat = df_stratifiable['label'].values
                stratify_col_strat = df_stratifiable['stratify_col']

                X_train, X_val, y_train, y_val = train_test_split(
                    X_strat, y_strat,
                    test_size=test_size, 
                    stratify=stratify_col_strat, 
                    random_state=self.random_state
                )
            else:
                # If no data is stratifiable, initialize empty sets for validation
                # and put everything into the single-member pool
                X_train, X_val, y_train, y_val = (
                    np.array([]).reshape(0, len(feature_cols)), 
                    np.array([]).reshape(0, len(feature_cols)), 
                    np.array([], dtype=int), 
                    np.array([], dtype=int)
                )
                df_single_member = df # All data is single-member strata
                print("Warning: No strata have more than 1 member. Cannot create a validation set. All data will be used for training.")

            # Handle the single-member part by adding it to the training set
            if not df_single_member.empty:
                X_single = df_single_member[feature_cols].values
                y_single = df_single_member['label'].values
                
                # Concatenate with the training set
                if X_train.size > 0:
                    X_train = np.vstack([X_train, X_single])
                    y_train = np.concatenate([y_train, y_single])
                else:
                    X_train = X_single
                    y_train = y_single
                    
            print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
            if y_train.size > 0:
                print(f"Train label distribution: {np.bincount(y_train)}")
            if y_val.size > 0:
                print(f"Val label distribution: {np.bincount(y_val)}")
            else:
                print("Validation set is empty.")
                
            return X_train, X_val, y_train, y_val
    
    def apply_pca(self, X_train: np.ndarray, X_val: np.ndarray, 
                  explained_variance: float = 0.95) -> Tuple:
        """
        Apply PCA with incremental fitting for large datasets.
        
        Args:
            X_train: Training features
            X_val: Validation features
            explained_variance: Target cumulative explained variance
            
        Returns:
            X_train_pca, X_val_pca
        """
        print(f"Applying PCA (target variance: {explained_variance})...")
        
        # Standardize features first
        print("Standardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Determine number of components
        n_components = min(X_train.shape[0], X_train.shape[1])
        
        # Use IncrementalPCA for large datasets
        print(f"Fitting IncrementalPCA with {n_components} components...")
        self.pca = IncrementalPCA(n_components=n_components, batch_size=self.pca_batch_size)
        
        # Fit PCA in batches
        n_batches = (len(X_train_scaled) + self.pca_batch_size - 1) // self.pca_batch_size
        for i in range(n_batches):
            start_idx = i * self.pca_batch_size
            end_idx = min((i + 1) * self.pca_batch_size, len(X_train_scaled))
            batch = X_train_scaled[start_idx:end_idx]
            self.pca.partial_fit(batch)
            if (i + 1) % 10 == 0:
                print(f"Fitted batch {i + 1}/{n_batches}")
        
        # Find number of components for target variance
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components_final = np.argmax(cumsum_variance >= explained_variance) + 1
        
        print(f"Using {n_components_final} components "
              f"(explained variance: {cumsum_variance[n_components_final-1]:.4f})")
        
        # Re-fit PCA with the optimal number of components for transformation
        print(f"Transforming data using {n_components_final} components...")
        self.pca = IncrementalPCA(n_components=n_components_final, batch_size=self.pca_batch_size)
        
        # Fit and transform the training data
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        # Transform the validation data
        X_val_pca = self.pca.transform(X_val_scaled)
        
        print(f"PCA complete. New shape: {X_train_pca.shape}")
        
        return X_train_pca, X_val_pca
    
    def build_model(self, input_dim: int, num_classes: int, 
                   hidden_dims: List[int] = [256, 128, 64]) -> nn.Module:
        """
        Build a feedforward neural network.
        
        Args:
            input_dim: Input dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            
        Returns:
            PyTorch model
        """
        class FeedForwardNN(nn.Module):
            def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
                super(FeedForwardNN, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, num_classes))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        model = FeedForwardNN(input_dim, hidden_dims, num_classes).to(self.device)
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int, lr: float, patience: int, 
                   output_dir: str) -> Dict:
        """
        Train the model with early stopping based on validation accuracy.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            output_dir: Directory to save the best model checkpoint
            
        Returns:
            Training history dictionary
        """
        checkpoint_path = os.path.join(output_dir, 'best_model.pth')
        print(f"Training model for up to {epochs} epochs...")
        
        # Determine number of classes
        num_classes = len(np.unique(y_train))
        
        # Build model
        self.model = self.build_model(X_train.shape[1], num_classes)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=0)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        history = { 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [] }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
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
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
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
                    'val_loss': avg_val_loss,
                    'val_acc': val_accuracy
                }, checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path} (val_acc: {val_accuracy:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("\nTraining complete!")
        return history
    
    def plot_training_history(self, history: Dict, output_dir: str):
        """
        Plot training and validation loss/accuracy.
        
        Args:
            history: Training history dictionary
            output_dir: Directory to save the plot
        """
        save_path = os.path.join(output_dir, 'training_plot.png')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
        plt.close()
    
    def evaluate_model(self, X_val: np.ndarray, y_val: np.ndarray, 
                      output_dir: str) -> Dict:
        """
        Evaluate the model on validation set and save results.
        
        Args:
            X_val, y_val: Validation data
            output_dir: Directory to load model and save results
            
        Returns:
            Dictionary with evaluation metrics
        """
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
        
        # Prepare evaluation report
        report_buffer = StringIO()
        report_buffer.write("="*50 + "\nEVALUATION METRICS\n" + "="*50 + "\n")
        report_buffer.write(f"Accuracy:  {accuracy:.4f}\n")
        report_buffer.write(f"Precision: {precision:.4f}\n")
        report_buffer.write(f"Recall:    {recall:.4f}\n")
        report_buffer.write(f"F1 Score:  {f1:.4f}\n" + "="*50 + "\n")
        
        report_buffer.write("\nDetailed Classification Report:\n")
        class_report = classification_report(all_labels, all_preds, zero_division=0)
        report_buffer.write(class_report + "\n")
        
        report_text = report_buffer.getvalue()
        print(report_text)
        
        # Save evaluation report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Evaluation report saved to {report_path}")
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()
        
        return metrics


def main():
    """
    Main execution function with configurable parameters from command line.
    """
    parser = argparse.ArgumentParser(description="LLM Code Detection Pipeline")
    parser.add_argument('--task', type=str, required=True, choices=['a', 'b'],
                        help="Task to run ('a' or 'b')")
    parser.add_argument('--n_samples', type=int, default=100000,
                        help="Number of samples to use. -1 for the entire dataset.")
    args = parser.parse_args()

    # --- Configuration ---
    data_path = data_path_a if args.task == 'a' else data_path_b
    n_samples_str = "full" if args.n_samples == -1 else str(args.n_samples)
    output_dir = f"{n_samples_str}_task_{args.task}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {output_dir}")

    CONFIG = {
        'data_path': data_path,
        'n_samples': args.n_samples,
        'output_dir': output_dir,
        'test_size': 0.2,
        'explained_variance': 0.95,
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 20,
        'batch_size': 1024,
        'feature_batch_size': 10000,
        'pca_batch_size': 5000,
        'hidden_dims': [256, 128, 64],
        'random_state': 42
    }
    
    # --- Pipeline Execution ---
    pipeline = LLMDetectionPipeline(
        data_path=CONFIG['data_path'],
        batch_size=CONFIG['batch_size'],
        feature_batch_size=CONFIG['feature_batch_size'],
        pca_batch_size=CONFIG['pca_batch_size'],
        random_state=CONFIG['random_state']
    )
    
    # 1. Load proportional sample
    df = pipeline.load_proportional_sample(CONFIG['n_samples'], CONFIG['output_dir'])
    
    # 2. Extract features
    features_df = pipeline.extract_features_batch(df)
    
    # 3. Split data
    X_train, X_val, y_train, y_val = pipeline.split_data(
        features_df, test_size=CONFIG['test_size']
    )
    
    # 4. Apply PCA
    X_train_pca, X_val_pca = pipeline.apply_pca(
        X_train, X_val, explained_variance=CONFIG['explained_variance']
    )
    
    # 5. Train model
    history = pipeline.train_model(
        X_train_pca, y_train, X_val_pca, y_val,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        patience=CONFIG['patience'],
        output_dir=CONFIG['output_dir']
    )
    
    # 6. Plot training history
    pipeline.plot_training_history(history, CONFIG['output_dir'])
    
    # 7. Evaluate model
    metrics = pipeline.evaluate_model(X_val_pca, y_val, CONFIG['output_dir'])
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*50)
    print(f"All artifacts saved in directory: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()