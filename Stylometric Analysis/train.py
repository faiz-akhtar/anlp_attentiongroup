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
warnings.filterwarnings('ignore')

# Import feature extractors
# Make sure these files (FeatureExtractor.py, etc.) are in the same directory
from FeatureExtractor import FeatureExtractor
from FormattingMicroSignatureExtractor import FormattingMicroSignatureExtractor
from CommentParser import CommentFeatureProcessor
from VariableNamingEntropyExtractor import VariableNamingEntropyExtractor


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
    
    def load_stratified_sample(self, n_samples: int) -> pd.DataFrame:
        """
        Load a stratified sample from the dataset.
        This stratification ensures that each label has an equal number of samples.
        Within each label, the samples are drawn equally from the available generators.
        
        Args:
            n_samples: Total number of samples to load
            
        Returns:
            Stratified sample DataFrame
        """
        print(f"Loading stratified sample of approximately {n_samples} samples...")
        
        # Read the full dataset to get metadata
        df = pd.read_parquet(self.data_path)
        
        unique_labels = df['label'].unique()
        n_labels = len(unique_labels)
        
        if n_labels == 0:
            print("No labels found in the dataset.")
            return pd.DataFrame()
            
        samples_per_label = n_samples // n_labels
        print(f"Targeting {samples_per_label} samples per label.")
        
        sampled_dfs = []
        
        for label in unique_labels:
            print(f"\nProcessing label: '{label}'")
            df_label = df[df['label'] == label]
            
            unique_generators = df_label['generator'].unique()
            n_generators = len(unique_generators)
            
            if n_generators == 0:
                print(f"  - No generators found for label '{label}'. Skipping.")
                continue
                
            samples_per_generator = samples_per_label // n_generators
            print(f"  - Found {n_generators} generator(s): {unique_generators}")
            print(f"  - Targeting {samples_per_generator} samples per generator.")

            # Group by generator within the current label and sample from each group
            label_sample_df = df_label.groupby('generator', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), samples_per_generator),
                    random_state=self.random_state
                )
            )
            sampled_dfs.append(label_sample_df)
        
        if not sampled_dfs:
            print("Could not sample any data.")
            return pd.DataFrame()
            
        # Combine all the sampled dataframes
        df_sample = pd.concat(sampled_dfs).reset_index(drop=True)
        
        # Shuffle the final dataframe to mix labels and generators
        df_sample = df_sample.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print("\n" + "="*50)
        print("STRATIFIED SAMPLING SUMMARY")
        print("="*50)
        print(f"Loaded a total of {len(df_sample)} samples.")
        print("\nFinal Label distribution:")
        print(df_sample['label'].value_counts())
        print("\nFinal Generator distribution (within each label):")
        print(df_sample.groupby('label')['generator'].value_counts())
        print("="*50 + "\n")
        
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
        Split data into train and validation sets.
        
        Args:
            df: Features DataFrame
            test_size: Proportion of data for validation
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        print(f"Splitting data (test_size={test_size})...")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns 
                       if col not in ['label', 'generator', 'language']]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=self.random_state
        )
        
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}")
        print(f"Train label distribution: {np.bincount(y_train)}")
        print(f"Val label distribution: {np.bincount(y_val)}")
        
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
                   epochs: int = 100, lr: float = 0.001,
                   patience: int = 10, checkpoint_path: str = 'best_model.pth') -> Dict:
        """
        Train the model with early stopping based on validation accuracy.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum number of epochs
            lr: Learning rate
            patience: Early stopping patience
            checkpoint_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
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
        # MODIFIED: Scheduler now monitors validation accuracy
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # MODIFIED: Track best validation accuracy instead of loss
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
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
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
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
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(val_accuracy)
            
            # MODIFIED: Learning rate scheduling based on validation accuracy
            scheduler.step(val_accuracy)
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            
            # MODIFIED: Early stopping and checkpointing based on validation accuracy
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
                print(f"Model checkpoint saved (val_acc: {val_accuracy:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("\nTraining complete!")
        return history
    
    def plot_training_history(self, history: Dict, save_path: str = 'training_plot.png'):
        """
        Plot training and validation loss/accuracy.
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
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
                      checkpoint_path: str = 'best_model.pth') -> Dict:
        """
        Evaluate the model on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            checkpoint_path: Path to best model checkpoint
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating model on validation set...")
        
        # Load best checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Predict
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                               shuffle=False, num_workers=0)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("="*50)
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to confusion_matrix.png")
        plt.close()
        
        return metrics


def main():
    """
    Main execution function with configurable parameters.
    """
    # Configuration
    CONFIG = {
        'data_path': '../SemEval-2026-Task13/task_b/task_b_training_set.parquet',
        'n_samples': 100000,  # Number of samples to use (increase as needed)
        'test_size': 0.2,    # Train/val split ratio
        'explained_variance': 0.95,  # PCA variance threshold
        'epochs': 100,
        'learning_rate': 0.001,
        'patience': 10,
        'batch_size': 256,
        'feature_batch_size': 100,
        'pca_batch_size': 5000,
        'hidden_dims': [256, 128, 64],
        'random_state': 42
    }
    
    # Initialize pipeline
    pipeline = LLMDetectionPipeline(
        data_path=CONFIG['data_path'],
        batch_size=CONFIG['batch_size'],
        feature_batch_size=CONFIG['feature_batch_size'],
        pca_batch_size=CONFIG['pca_batch_size'],
        random_state=CONFIG['random_state']
    )
    
    # 1. Load stratified sample
    df = pipeline.load_stratified_sample(CONFIG['n_samples'])
    
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
        patience=CONFIG['patience']
    )
    
    # 6. Plot training history
    pipeline.plot_training_history(history)
    
    # 7. Evaluate model
    metrics = pipeline.evaluate_model(X_val_pca, y_val)
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*50)
    print(f"Best model saved to: best_model.pth")
    print(f"Training plot saved to: training_plot.png")
    print(f"Confusion matrix saved to: confusion_matrix.png")


if __name__ == "__main__":
    main()