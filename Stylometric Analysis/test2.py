import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# Import feature extractors
from FeatureExtractor import FeatureExtractor
from FormattingMicroSignatureExtractor import FormattingMicroSignatureExtractor
from CommentParser import CommentFeatureProcessor
from VariableNamingEntropyExtractor import VariableNamingEntropyExtractor

# Global constants for data paths
SUBTASK_A_PATHS = {
    'test': '../SemEval-2026-Task13/task_a/task_a_test_set_sample.parquet'
}

SUBTASK_B_PATHS = {
    'test': '../SemEval-2026-Task13/task_b/task_b_test_set_sample.parquet'
}


class LLMDetectionTester:
    """Testing pipeline for LLM-generated code detection."""
    
    def __init__(self, subtask: str, model_dir: str, batch_size: int = 1024,
                 feature_batch_size: int = 10000):
        self.subtask = subtask
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.feature_batch_size = feature_batch_size
        self.paths = SUBTASK_A_PATHS if subtask == 'a' else SUBTASK_B_PATHS
        
        self.formatting_extractor = FormattingMicroSignatureExtractor()
        self.comment_processor = CommentFeatureProcessor()
        self.variable_extractor = VariableNamingEntropyExtractor()
        
        self.model = None
        self.scaler = None
        self.pca = None
        self.feature_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Testing Subtask {subtask.upper()}")
    
    def load_test_data(self) -> pd.DataFrame:
        print(f"Loading test data from: {self.paths['test']}")
        df_test = pd.read_parquet(self.paths['test'])
        print(f"Loaded {len(df_test)} test samples")
        
        print("\n--- Test Dataset Distribution ---")
        print("Label distribution:")
        print(df_test['label'].value_counts(normalize=True))
        print("\n[Label, Language] distribution:")
        print(df_test.groupby('label')['language'].value_counts(normalize=True))
        
        return df_test
    
    def extract_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nExtracting features from test data...")
        
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
                    
                    format_features = self.formatting_extractor.extract_all_signatures(
                        row['code'], row['language']
                    )
                    features.update({f'format_{k}': v for k, v in format_features.items()})
                    
                    comment_features = self.comment_processor.extractor.extract_features(
                        row['code'], row['language']
                    )
                    features.update({f'comment_{k}': v for k, v in comment_features.items()})
                    
                    var_features = self.variable_extractor.extract_all_features(
                        row['code'], row['language']
                    )
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
    
    def load_model(self):
        checkpoint_path = os.path.join(self.model_dir, 'best_model.pth')
        print(f"\nLoading model from: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.scaler = checkpoint['scaler_state']
        self.pca = checkpoint['pca_state']
        self.feature_cols = checkpoint.get('feature_cols', None)
        
        model_state = checkpoint['model_state_dict']
        
        # 1. Find all keys corresponding to Linear layer weights.
        linear_layer_keys = []
        for key in model_state.keys():
            if 'network' in key and '.weight' in key and len(model_state[key].shape) == 2:
                # This filter works because only Linear layers have 2D '.weight' tensors in this architecture
                linear_layer_keys.append(key)

        # 2. Sort the keys numerically based on their layer index.
        # e.g., 'network.12.weight' should come after 'network.8.weight'
        linear_layer_keys.sort(key=lambda x: int(x.split('.')[1]))

        # 3. Build the list of layer shapes from the correctly sorted keys.
        linear_layers = [(key, model_state[key].shape) for key in linear_layer_keys]

        # Now, infer the architecture from the correctly ordered layers
        input_dim = linear_layers[0][1][1]
        num_classes = linear_layers[-1][1][0]
        hidden_dims = [shape[0] for _, shape in linear_layers[:-1]]
        
        print(f"Model architecture: input_dim={input_dim}, hidden_dims={hidden_dims}, num_classes={num_classes}")
        
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
        
        self.model = FeedForwardNN(input_dim, hidden_dims, num_classes).to(self.device)
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def preprocess_features(self, features_df: pd.DataFrame) -> tuple:
        print("\nPreprocessing features...")
        
        # Align columns with training set
        if self.feature_cols is not None:
            print(f"Aligning test features with {len(self.feature_cols)} training features...")
            
            # Add missing columns with zeros
            missing_cols = [col for col in self.feature_cols if col not in features_df.columns]
            if missing_cols:
                print(f"Adding {len(missing_cols)} missing columns to test set")
                for col in missing_cols:
                    features_df[col] = 0.0
            
            # Remove extra columns
            extra_cols = [col for col in features_df.columns 
                         if col not in self.feature_cols + ['label', 'language']]
            if extra_cols:
                print(f"Removing {len(extra_cols)} extra columns from test set")
                features_df = features_df.drop(columns=extra_cols)
            
            # Ensure correct order
            feature_cols = self.feature_cols
        else:
            # Fallback if feature_cols not saved in checkpoint
            print("Warning: feature_cols not found in checkpoint, using all numeric columns")
            feature_cols = [col for col in features_df.columns 
                          if col not in ['label', 'language']]
        
        X_test = features_df[feature_cols].values
        y_test = features_df['label'].values
        
        print(f"Features before preprocessing: {X_test.shape[1]}")
        
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"Features after preprocessing: {X_test_pca.shape[1]}")
        
        return X_test_pca, y_test
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, output_dir: str):
        print("\nEvaluating model on test set...")
        
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate all metrics
        accuracy = accuracy_score(all_labels, all_preds)
        
        f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0)
        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0)
        recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Prepare report
        report_buffer = StringIO()
        report_buffer.write("="*70 + "\n")
        report_buffer.write(f"TEST SET EVALUATION - SUBTASK {self.subtask.upper()}\n")
        report_buffer.write("="*70 + "\n\n")
        
        report_buffer.write("SUMMARY METRICS\n")
        report_buffer.write("-"*70 + "\n")
        report_buffer.write(f"Accuracy:           {accuracy:.4f}\n\n")
        
        report_buffer.write("F1 SCORES:\n")
        report_buffer.write(f"  F1 Micro:         {f1_micro:.4f}\n")
        report_buffer.write(f"  F1 Macro:         {f1_macro:.4f}\n")
        report_buffer.write(f"  F1 Weighted:      {f1_weighted:.4f}\n\n")
        
        report_buffer.write("PRECISION SCORES:\n")
        report_buffer.write(f"  Precision Micro:  {precision_micro:.4f}\n")
        report_buffer.write(f"  Precision Macro:  {precision_macro:.4f}\n")
        report_buffer.write(f"  Precision Weighted: {precision_weighted:.4f}\n\n")
        
        report_buffer.write("RECALL SCORES:\n")
        report_buffer.write(f"  Recall Micro:     {recall_micro:.4f}\n")
        report_buffer.write(f"  Recall Macro:     {recall_macro:.4f}\n")
        report_buffer.write(f"  Recall Weighted:  {recall_weighted:.4f}\n")
        report_buffer.write("="*70 + "\n\n")
        
        report_buffer.write("DETAILED CLASSIFICATION REPORT\n")
        report_buffer.write("-"*70 + "\n")
        class_report = classification_report(all_labels, all_preds, zero_division=0)
        report_buffer.write(class_report + "\n")
        
        report_text = report_buffer.getvalue()
        print(report_text)
        
        report_path = os.path.join(output_dir, 'test_evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Test evaluation report saved to {report_path}")
        
        # Save metrics as CSV
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'F1_Micro', 'F1_Macro', 'F1_Weighted',
                      'Precision_Micro', 'Precision_Macro', 'Precision_Weighted',
                      'Recall_Micro', 'Recall_Macro', 'Recall_Weighted'],
            'Score': [accuracy, f1_micro, f1_macro, f1_weighted,
                     precision_micro, precision_macro, precision_weighted,
                     recall_micro, recall_macro, recall_weighted]
        })
        csv_path = os.path.join(output_dir, 'test_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Test metrics saved to {csv_path}")
        
        # Confusion matrix
        cm_path = os.path.join(output_dir, 'test_confusion_matrix.png')
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Test Set Confusion Matrix - Subtask {self.subtask.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Test confusion matrix saved to {cm_path}")
        plt.close()
        
        # Create metrics comparison plot
        self._plot_metrics_comparison(
            f1_micro, f1_macro, f1_weighted,
            precision_micro, precision_macro, precision_weighted,
            recall_micro, recall_macro, recall_weighted,
            output_dir
        )
        
        return {
            'accuracy': accuracy,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_micro': precision_micro,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted
        }
    
    def _plot_metrics_comparison(self, f1_micro, f1_macro, f1_weighted,
                                 prec_micro, prec_macro, prec_weighted,
                                 rec_micro, rec_macro, rec_weighted,
                                 output_dir):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        categories = ['Micro', 'Macro', 'Weighted']
        f1_scores = [f1_micro, f1_macro, f1_weighted]
        precision_scores = [prec_micro, prec_macro, prec_weighted]
        recall_scores = [rec_micro, rec_macro, rec_weighted]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax.bar(x - width, f1_scores, width, label='F1 Score', color='#2ecc71')
        ax.bar(x, precision_scores, width, label='Precision', color='#3498db')
        ax.bar(x + width, recall_scores, width, label='Recall', color='#e74c3c')
        
        ax.set_xlabel('Averaging Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Test Set Metrics Comparison - Subtask {self.subtask.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'test_metrics_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison plot saved to {plot_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="LLM Code Detection Testing Pipeline")
    parser.add_argument('--subtask', type=str, required=True, choices=['a', 'b'],
                        help="Subtask to test ('a' or 'b')")
    parser.add_argument('--model_dir', type=str, required=True,
                        help="Directory containing the trained model")
    args = parser.parse_args()
    
    output_dir = os.path.join(args.model_dir, 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Test results will be saved to: {output_dir}")
    
    tester = LLMDetectionTester(subtask=args.subtask, model_dir=args.model_dir)
    
    df_test = tester.load_test_data()
    test_features_df = tester.extract_features_batch(df_test)
    tester.load_model()
    X_test, y_test = tester.preprocess_features(test_features_df)
    metrics = tester.evaluate_model(X_test, y_test, output_dir)
    
    print("\n" + "="*70)
    print("TEST EVALUATION COMPLETE")
    print("="*70)
    print(f"All test results saved in: {output_dir}")
    print("\nKey Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Micro:    {metrics['f1_micro']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()