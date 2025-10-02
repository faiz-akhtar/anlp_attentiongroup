import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

# Import feature extractors (assuming they're in the same directory)
from FeatureExtractor import FeatureExtractor
from FormattingMicroSignatureExtractor import FormattingMicroSignatureExtractor, process_dataset
from CommentParser import CommentFeatureProcessor
from VariableNamingEntropyExtractor import VariableNamingEntropyExtractor


def sample_data(df, n_samples, labels=None):
    """Sample data stratified by generator/label"""
    if labels is not None:
        df = df[df['label'].isin(labels)]
    
    # Stratified sampling
    sampled_df = df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), n_samples // df['label'].nunique()), random_state=42)
    )
    
    return sampled_df.reset_index(drop=True)


def extract_all_features(df):
    """Extract features using all four extractors"""
    print("Extracting features...")
    all_features = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(df)}")
        
        features = {'index': idx}
        
        try:
            # 1. FeatureExtractor
            fe = FeatureExtractor(row['code'], row['language'])
            features.update(fe.extract_all_features())
            
            # 2. FormattingMicroSignatureExtractor
            fmse = FormattingMicroSignatureExtractor()
            formatting_features = fmse.extract_all_signatures(row['code'], row['language'])
            features.update({f'fmt_{k}': v for k, v in formatting_features.items()})
            
            # 3. CommentFeatureProcessor
            cfp = CommentFeatureProcessor()
            comment_df = pd.DataFrame([row])
            comment_features_df = cfp.extractor.extract_features(row['code'], row['language'])
            features.update({f'cmt_{k}': v for k, v in comment_features_df.items()})
            
            # 4. VariableNamingEntropyExtractor
            vnee = VariableNamingEntropyExtractor()
            var_features = vnee.extract_all_features(row['code'], row['language'])
            features.update({f'var_{k}': v for k, v in var_features.items()})
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
        
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Merge with original data
    result_df = df.copy()
    for col in features_df.columns:
        if col != 'index':
            result_df[col] = features_df[col].values
    
    return result_df


def get_legend_label(generators, task):
    """Generate legend label based on task and generators"""
    if task == 'a':
        return 'Human' if generators[0] == 'human' else 'LLM'
    else:  # task == 'b'
        if len(generators) == 1:
            return generators[0]
        # Find common prefix/substring
        common = generators[0]
        for gen in generators[1:]:
            # Find longest common substring
            common_parts = []
            words1 = common.lower().split()
            words2 = gen.lower().split()
            for w1 in words1:
                for w2 in words2:
                    if w1 in w2 or w2 in w1:
                        common_parts.append(w1 if len(w1) >= len(w2) else w2)
            if common_parts:
                common = max(common_parts, key=len)
        return common.capitalize()


def get_distinct_colors(n):
    """Generate maximally distinct colors"""
    if n == 2:
        return ['red', 'green']
    elif n <= 10:
        # Use tab10 colormap for up to 10 colors
        cmap = plt.cm.get_cmap('tab10')
        return [cmap(i) for i in range(n)]
    else:
        # Use hsv colormap for more colors
        cmap = plt.cm.get_cmap('hsv')
        return [cmap(i / n) for i in range(n)]


def plot_pca(df, task, labels_filter):
    """Perform PCA and plot results"""
    print("Performing PCA...")
    
    # Select only numeric feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['code', 'generator', 'label', 'language', 'index']]
    
    X = df[feature_cols].values
    
    # Handle NaN and inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # Remove outliers (keep within 2 std dev)
    mean_pc1, std_pc1 = np.mean(X_pca[:, 0]), np.std(X_pca[:, 0])
    mean_pc2, std_pc2 = np.mean(X_pca[:, 1]), np.std(X_pca[:, 1])
    
    mask = ((np.abs(X_pca[:, 0] - mean_pc1) <= 2 * std_pc1) & 
            (np.abs(X_pca[:, 1] - mean_pc2) <= 2 * std_pc2))
    
    X_pca_filtered = X_pca[mask]
    df_filtered = df[mask].reset_index(drop=True)
    
    print(f"Removed {np.sum(~mask)} outliers ({np.sum(~mask)/len(mask)*100:.1f}%)")
    
    # Create label to generator mapping
    label_to_generators = df_filtered.groupby('label')['generator'].apply(list).to_dict()
    label_to_legend = {}
    for label, gens in label_to_generators.items():
        unique_gens = list(set(gens))
        label_to_legend[label] = get_legend_label(unique_gens, task)
    
    # Get unique labels and colors
    unique_labels = sorted(df_filtered['label'].unique())
    colors = get_distinct_colors(len(unique_labels))
    label_to_color = dict(zip(unique_labels, colors))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    for label in unique_labels:
        mask_label = df_filtered['label'] == label
        plt.scatter(X_pca_filtered[mask_label, 0], 
                   X_pca_filtered[mask_label, 1],
                   c=[label_to_color[label]], 
                   label=label_to_legend[label],
                   alpha=0.6,
                   s=20,
                   edgecolors='none')
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title(f'PCA Visualization - Task {task.upper()}', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    filename = f'pca_visualization_task_{task}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Feature extraction and PCA visualization')
    parser.add_argument('n_samples', type=int, help='Number of samples to extract')
    parser.add_argument('task', choices=['a', 'b'], help='Task type (a or b)')
    parser.add_argument('--labels', type=int, nargs='+', default=None,
                       help='List of labels to filter (optional)')
    parser.add_argument('--data-path', type=str, 
                       default='./SemEval-2026-Task13/task_a/task_a_training_set_1.parquet',
                       help='Path to dataset')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    try:
        df = pd.read_parquet(args.data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Unique labels: {sorted(df['label'].unique())}")
    print(f"Unique generators: {df['generator'].unique()}")
    
    # Sample data
    print(f"\nSampling {args.n_samples} data points...")
    sampled_df = sample_data(df, args.n_samples, args.labels)
    print(f"Sampled shape: {sampled_df.shape}")
    print(f"Label distribution:\n{sampled_df['label'].value_counts().sort_index()}")
    
    # Extract features
    features_df = extract_all_features(sampled_df)
    print(f"\nFeatures extracted. Total columns: {len(features_df.columns)}")
    
    # Plot PCA
    plot_pca(features_df, args.task, args.labels)


if __name__ == '__main__':
    main()