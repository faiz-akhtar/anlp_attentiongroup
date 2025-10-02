import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import math
from typing import Dict, List, Tuple, Any
from scipy.stats import entropy
import ast
import tokenize
from io import StringIO

class FormattingMicroSignatureExtractor:
    """
    Extracts formatting micro-signatures from source code to distinguish
    human-written code from machine-generated code and identify LLM families.
    """
    
    def __init__(self):
        self.operators = ['=', '+', '-', '*', '/', '//', '%', '**', 
                         '==', '!=', '<', '>', '<=', '>=', '&&', '||', 
                         '&', '|', '^', '<<', '>>', '+=', '-=', '*=', '/=']
        
        self.delimiters = ['(', ')', '[', ']', '{', '}', ',', ';', ':']
        
        # Language-specific patterns
        self.language_patterns = {
            'Python': {
                'keywords': ['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except'],
                'operators': [':', 'lambda', 'in', 'not in', 'is', 'is not'],
                'comment_char': '#'
            },
            'Java': {
                'keywords': ['public', 'private', 'protected', 'class', 'interface', 'if', 'else'],
                'operators': ['instanceof'],
                'comment_char': '//'
            },
            'C++': {
                'keywords': ['class', 'struct', 'namespace', 'template', 'if', 'else'],
                'operators': ['::', '->'],
                'comment_char': '//'
            },
            'JavaScript': {
                'keywords': ['function', 'var', 'let', 'const', 'if', 'else'],
                'operators': ['===', '!==', '=>'],
                'comment_char': '//'
            },
            'Go': {
                'keywords': ['func', 'type', 'struct', 'interface', 'if', 'else'],
                'operators': [':=', '<-'],
                'comment_char': '//'
            },
            'PHP': {
                'keywords': ['function', 'class', 'if', 'else', 'foreach'],
                'operators': ['=>', '.='],
                'comment_char': '//'
            },
            'C': {
                'keywords': ['struct', 'typedef', 'if', 'else', 'for', 'while'],
                'operators': ['->'],
                'comment_char': '//'
            },
            'C#': {
                'keywords': ['class', 'interface', 'namespace', 'if', 'else'],
                'operators': ['??', '?.'],
                'comment_char': '//'
            }
        }

    def extract_all_signatures(self, code: str, language: str) -> Dict[str, Any]:
        """Extract all formatting micro-signatures from code."""
        signatures = {}
        
        # Core pattern extraction
        signatures.update(self.extract_spacing_patterns(code, language))
        signatures.update(self.extract_indentation_patterns(code, language))
        signatures.update(self.extract_line_patterns(code))
        signatures.update(self.extract_delimiter_patterns(code, language))
        signatures.update(self.extract_brace_patterns(code, language))
        signatures.update(self.extract_comment_patterns(code, language))
        signatures.update(self.extract_language_specific_patterns(code, language))
        
        # Statistical measures
        signatures.update(self.calculate_consistency_scores(signatures))
        
        return signatures

    def extract_spacing_patterns(self, code: str, language: str) -> Dict[str, float]:
        """Extract spacing patterns around operators and keywords."""
        patterns = {}
        
        # Get language-specific operators
        ops = self.operators.copy()
        if language in self.language_patterns:
            ops.extend(self.language_patterns[language]['operators'])
        
        for op in ops:
            spaces_before = []
            spaces_after = []
            
            # Escape special regex characters
            escaped_op = re.escape(op)
            pattern = rf'(\s*){escaped_op}(\s*)'
            
            for match in re.finditer(pattern, code):
                spaces_before.append(len(match.group(1)))
                spaces_after.append(len(match.group(2)))
            
            if spaces_before:
                op_clean = re.sub(r'[^\w]', '_', op)
                patterns[f'{op_clean}_before_mean'] = np.mean(spaces_before)
                patterns[f'{op_clean}_before_var'] = np.var(spaces_before)
                patterns[f'{op_clean}_after_mean'] = np.mean(spaces_after)
                patterns[f'{op_clean}_after_var'] = np.var(spaces_after)
                patterns[f'{op_clean}_spacing_consistency'] = self._calculate_consistency(
                    list(zip(spaces_before, spaces_after))
                )
        
        # Function call spacing patterns
        func_patterns = re.findall(r'(\w+)\s*(\()', code)
        if func_patterns:
            func_spaces = [len(match[0]) - len(match[0].rstrip()) for match in func_patterns]
            patterns['function_call_spacing_var'] = np.var(func_spaces) if func_spaces else 0
        
        return patterns

    def extract_indentation_patterns(self, code: str, language: str) -> Dict[str, float]:
        """Extract indentation micro-patterns."""
        lines = code.split('\n')
        indents = []
        indent_changes = []
        
        prev_indent = 0
        for line in lines:
            if line.strip():  # Non-empty lines
                leading_spaces = len(line) - len(line.lstrip())
                indents.append(leading_spaces)
                indent_changes.append(leading_spaces - prev_indent)
                prev_indent = leading_spaces
        
        patterns = {}
        if indents:
            patterns['indent_mean'] = np.mean(indents)
            patterns['indent_var'] = np.var(indents)
            patterns['indent_entropy'] = entropy(np.bincount(indents) + 1e-10)
            patterns['indent_gcd'] = math.gcd(*indents) if len(set(indents)) > 1 else 1
            patterns['mixed_indent_score'] = self._detect_mixed_indentation(lines)
            patterns['indent_change_var'] = np.var(indent_changes) if indent_changes else 0
            patterns['max_indent_level'] = max(indents) if indents else 0
        
        # Alignment patterns (continuation lines)
        patterns['alignment_consistency'] = self._measure_alignment_consistency(lines)
        
        return patterns

    def extract_line_patterns(self, code: str) -> Dict[str, float]:
        """Extract line-level formatting patterns."""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        patterns = {}
        if non_empty_lines:
            line_lengths = [len(line) for line in non_empty_lines]
            patterns['line_length_mean'] = np.mean(line_lengths)
            patterns['line_length_var'] = np.var(line_lengths)
            patterns['line_length_entropy'] = entropy(np.bincount(line_lengths) + 1e-10)
            patterns['max_line_length'] = max(line_lengths)
            
            # Trailing space patterns
            trailing_spaces = [len(line) - len(line.rstrip()) for line in lines]
            patterns['trailing_space_freq'] = sum(1 for x in trailing_spaces if x > 0) / len(lines)
            patterns['trailing_space_var'] = np.var(trailing_spaces)
            
            # Blank line patterns
            blank_line_groups = self._analyze_blank_line_groups(lines)
            patterns['blank_line_entropy'] = entropy(np.array(blank_line_groups) + 1)
            patterns['blank_line_consistency'] = self._calculate_consistency(blank_line_groups)
            
            # Line density
            patterns['code_density'] = len(non_empty_lines) / len(lines)
        
        return patterns

    def extract_delimiter_patterns(self, code: str, language: str) -> Dict[str, float]:
        """Extract delimiter usage patterns."""
        patterns = {}
        
        for delim in self.delimiters:
            # Spacing around delimiters
            before_pattern = rf'(\s*){re.escape(delim)}'
            after_pattern = rf'{re.escape(delim)}(\s*)'
            
            before_matches = [len(m.group(1)) for m in re.finditer(before_pattern, code)]
            after_matches = [len(m.group(1)) for m in re.finditer(after_pattern, code)]
            
            if before_matches:
                patterns[f'{delim}_before_spacing_var'] = np.var(before_matches)
            if after_matches:
                patterns[f'{delim}_after_spacing_var'] = np.var(after_matches)
        
        # Comma spacing specifically (very discriminative)
        comma_patterns = re.findall(r'(\s*),(\s*)', code)
        if comma_patterns:
            before_comma = [len(match[0]) for match in comma_patterns]
            after_comma = [len(match[1]) for match in comma_patterns]
            patterns['comma_before_consistency'] = self._calculate_consistency(before_comma)
            patterns['comma_after_consistency'] = self._calculate_consistency(after_comma)
        
        return patterns

    def extract_brace_patterns(self, code: str, language: str) -> Dict[str, float]:
        """Extract brace style patterns (for applicable languages)."""
        patterns = {}
        
        if language in ['Java', 'C++', 'C', 'C#', 'JavaScript', 'Go', 'PHP']:
            # Opening brace patterns
            same_line_braces = len(re.findall(r'.*{\s*$', code, re.MULTILINE))
            new_line_braces = len(re.findall(r'^\s*{\s*$', code, re.MULTILINE))
            
            total_braces = same_line_braces + new_line_braces
            if total_braces > 0:
                patterns['brace_same_line_ratio'] = same_line_braces / total_braces
                patterns['brace_style_consistency'] = max(same_line_braces, new_line_braces) / total_braces
            
            # Brace spacing
            opening_brace_spaces = re.findall(r'(\s*){', code)
            if opening_brace_spaces:
                spaces = [len(match) for match in opening_brace_spaces]
                patterns['opening_brace_spacing_var'] = np.var(spaces)
        
        return patterns

    def extract_comment_patterns(self, code: str, language: str) -> Dict[str, float]:
        """Extract comment formatting patterns."""
        patterns = {}
        
        if language not in self.language_patterns:
            return patterns
            
        comment_char = self.language_patterns[language]['comment_char']
        
        # Single line comments
        comment_lines = []
        for line in code.split('\n'):
            if comment_char in line:
                comment_pos = line.find(comment_char)
                if comment_pos >= 0:
                    # Spacing before comment
                    prefix = line[:comment_pos]
                    if prefix.strip():  # Inline comment
                        trailing_space = len(prefix) - len(prefix.rstrip())
                        comment_lines.append(('inline', trailing_space))
                    else:  # Full line comment
                        leading_space = len(prefix)
                        comment_lines.append(('full_line', leading_space))
        
        if comment_lines:
            inline_comments = [x[1] for x in comment_lines if x[0] == 'inline']
            full_line_comments = [x[1] for x in comment_lines if x[0] == 'full_line']
            
            if inline_comments:
                patterns['inline_comment_spacing_var'] = np.var(inline_comments)
            if full_line_comments:
                patterns['full_line_comment_indent_var'] = np.var(full_line_comments)
            
            patterns['comment_ratio'] = len(comment_lines) / len(code.split('\n'))
        
        return patterns

    def extract_language_specific_patterns(self, code: str, language: str) -> Dict[str, float]:
        """Extract language-specific formatting patterns."""
        patterns = {}
        
        if language == 'Python':
            patterns.update(self._extract_python_patterns(code))
        elif language in ['Java', 'C#']:
            patterns.update(self._extract_java_csharp_patterns(code))
        elif language in ['C++', 'C']:
            patterns.update(self._extract_cpp_c_patterns(code))
        elif language == 'JavaScript':
            patterns.update(self._extract_javascript_patterns(code))
        elif language == 'Go':
            patterns.update(self._extract_go_patterns(code))
        elif language == 'PHP':
            patterns.update(self._extract_php_patterns(code))
        
        return patterns

    def _extract_python_patterns(self, code: str) -> Dict[str, float]:
        """Python-specific patterns."""
        patterns = {}
        
        # Colon spacing
        colon_matches = re.findall(r'(\s*):(\s*)', code)
        if colon_matches:
            before_colon = [len(match[0]) for match in colon_matches]
            after_colon = [len(match[1]) for match in colon_matches]
            patterns['colon_before_var'] = np.var(before_colon)
            patterns['colon_after_var'] = np.var(after_colon)
        
        # Lambda formatting
        lambda_count = len(re.findall(r'lambda\s+', code))
        patterns['lambda_density'] = lambda_count / len(code.split('\n'))
        
        # Import statement formatting
        import_lines = [line for line in code.split('\n') if re.match(r'^\s*(import|from)', line)]
        if import_lines:
            import_indents = [len(line) - len(line.lstrip()) for line in import_lines]
            patterns['import_indent_consistency'] = self._calculate_consistency(import_indents)
        
        return patterns

    def _extract_java_csharp_patterns(self, code: str) -> Dict[str, float]:
        """Java/C# specific patterns."""
        patterns = {}
        
        # Access modifier spacing
        modifiers = ['public', 'private', 'protected', 'static', 'final', 'abstract']
        modifier_lines = []
        for line in code.split('\n'):
            for modifier in modifiers:
                if modifier in line:
                    modifier_lines.append(line)
                    break
        
        if modifier_lines:
            patterns['modifier_line_density'] = len(modifier_lines) / len(code.split('\n'))
        
        # Generic bracket spacing
        generic_patterns = re.findall(r'(\s*)<.*?>(\s*)', code)
        if generic_patterns:
            before_generic = [len(match[0]) for match in generic_patterns]
            after_generic = [len(match[1]) for match in generic_patterns]
            patterns['generic_before_var'] = np.var(before_generic)
            patterns['generic_after_var'] = np.var(after_generic)
        
        return patterns

    def _extract_cpp_c_patterns(self, code: str) -> Dict[str, float]:
        """C++/C specific patterns."""
        patterns = {}
        
        # Pointer/reference spacing
        pointer_patterns = re.findall(r'(\s*)\*(\s*)', code)
        reference_patterns = re.findall(r'(\s*)&(\s*)', code)
        
        if pointer_patterns:
            pointer_before = [len(match[0]) for match in pointer_patterns]
            pointer_after = [len(match[1]) for match in pointer_patterns]
            patterns['pointer_before_var'] = np.var(pointer_before)
            patterns['pointer_after_var'] = np.var(pointer_after)
        
        # Include statement formatting
        include_lines = [line for line in code.split('\n') if line.strip().startswith('#include')]
        if include_lines:
            patterns['include_density'] = len(include_lines) / len(code.split('\n'))
        
        return patterns

    def _extract_javascript_patterns(self, code: str) -> Dict[str, float]:
        """JavaScript specific patterns."""
        patterns = {}
        
        # Semicolon usage (optional in JS)
        lines_with_semicolon = len([line for line in code.split('\n') 
                                   if line.strip().endswith(';')])
        total_statement_lines = len([line for line in code.split('\n') 
                                   if line.strip() and not line.strip().startswith('//')])
        
        if total_statement_lines > 0:
            patterns['semicolon_usage_ratio'] = lines_with_semicolon / total_statement_lines
        
        # Arrow function spacing
        arrow_patterns = re.findall(r'(\s*)=>(\s*)', code)
        if arrow_patterns:
            arrow_before = [len(match[0]) for match in arrow_patterns]
            arrow_after = [len(match[1]) for match in arrow_patterns]
            patterns['arrow_before_var'] = np.var(arrow_before)
            patterns['arrow_after_var'] = np.var(arrow_after)
        
        return patterns

    def _extract_go_patterns(self, code: str) -> Dict[str, float]:
        """Go specific patterns."""
        patterns = {}
        
        # Short variable declaration spacing
        short_decl_patterns = re.findall(r'(\s*):=(\s*)', code)
        if short_decl_patterns:
            before_decl = [len(match[0]) for match in short_decl_patterns]
            after_decl = [len(match[1]) for match in short_decl_patterns]
            patterns['short_decl_before_var'] = np.var(before_decl)
            patterns['short_decl_after_var'] = np.var(after_decl)
        
        return patterns

    def _extract_php_patterns(self, code: str) -> Dict[str, float]:
        """PHP specific patterns."""
        patterns = {}
        
        # Variable prefix consistency ($)
        var_patterns = re.findall(r'(\s*)\$\w+', code)
        if var_patterns:
            var_spacing = [len(match) for match in var_patterns]
            patterns['php_var_spacing_var'] = np.var(var_spacing)
        
        return patterns

    def calculate_consistency_scores(self, signatures: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall consistency metrics."""
        variance_keys = [k for k in signatures.keys() if '_var' in k]
        consistency_keys = [k for k in signatures.keys() if '_consistency' in k]
        
        consistency_scores = {}
        
        if variance_keys:
            variances = [signatures[k] for k in variance_keys if isinstance(signatures[k], (int, float))]
            if variances:
                consistency_scores['overall_formatting_variance'] = np.mean(variances)
                consistency_scores['formatting_variance_consistency'] = np.var(variances)
        
        if consistency_keys:
            consistencies = [signatures[k] for k in consistency_keys if isinstance(signatures[k], (int, float))]
            if consistencies:
                consistency_scores['overall_consistency_score'] = np.mean(consistencies)
        
        return consistency_scores

    # Helper methods
    def _calculate_consistency(self, values: List) -> float:
        """Calculate consistency score (1 - normalized entropy)."""
        if not values:
            return 0.0
        
        counter = Counter(values)
        total = len(values)
        probs = [count / total for count in counter.values()]
        
        if len(probs) == 1:
            return 1.0
        
        max_entropy = math.log(len(probs))
        actual_entropy = entropy(probs)
        
        return 1.0 - (actual_entropy / max_entropy) if max_entropy > 0 else 0.0

    def _detect_mixed_indentation(self, lines: List[str]) -> float:
        """Detect mixed tab/space indentation."""
        tab_lines = 0
        space_lines = 0
        
        for line in lines:
            if line.strip():
                if line.startswith('\t'):
                    tab_lines += 1
                elif line.startswith(' '):
                    space_lines += 1
        
        total_indented = tab_lines + space_lines
        if total_indented == 0:
            return 0.0
        
        return min(tab_lines, space_lines) / total_indented

    def _measure_alignment_consistency(self, lines: List[str]) -> float:
        """Measure consistency in continuation line alignment."""
        continuation_indents = []
        
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith(('(', '[', '{')):
                next_line = lines[i + 1]
                if next_line.strip():
                    continuation_indent = len(next_line) - len(next_line.lstrip())
                    base_indent = len(line) - len(line.lstrip())
                    continuation_indents.append(continuation_indent - base_indent)
        
        return self._calculate_consistency(continuation_indents)

    def _analyze_blank_line_groups(self, lines: List[str]) -> List[int]:
        """Analyze patterns in blank line groupings."""
        blank_groups = []
        current_blank_count = 0
        
        for line in lines:
            if not line.strip():
                current_blank_count += 1
            else:
                if current_blank_count > 0:
                    blank_groups.append(current_blank_count)
                    current_blank_count = 0
        
        return blank_groups if blank_groups else [0]


def process_dataset(df: pd.DataFrame, handle_nans: str = 'fill') -> pd.DataFrame:
    """
    Process a dataset and extract formatting signatures for all code samples.
    
    Args:
        df: Input dataframe with code, generator, label, language columns
        handle_nans: Strategy for NaN handling ('fill', 'drop', 'keep')
                    'fill' - Replace NaNs with appropriate defaults
                    'drop' - Remove columns with too many NaNs
                    'keep' - Keep NaNs as-is
    """
    extractor = FormattingMicroSignatureExtractor()
    
    all_signatures = []
    
    for idx, row in df.iterrows():
        try:
            signatures = extractor.extract_all_signatures(row['code'], row['language'])
            signatures['original_index'] = idx
            signatures['generator'] = row['generator']
            signatures['label'] = row['label']
            signatures['language'] = row['language']
            all_signatures.append(signatures)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    result_df = pd.DataFrame(all_signatures)
    
    if handle_nans == 'fill':
        result_df = _handle_nans_with_fill(result_df)
    elif handle_nans == 'drop':
        result_df = _handle_nans_with_drop(result_df)
    # 'keep' does nothing
    
    return result_df


def _handle_nans_with_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs with appropriate default values based on feature type."""
    df_filled = df.copy()
    
    # Identify feature types and fill appropriately
    for col in df_filled.columns:
        if col in ['original_index', 'generator', 'label', 'language']:
            continue
            
        if df_filled[col].isna().all():
            # If entire column is NaN, fill with 0
            df_filled[col] = 0.0
        elif df_filled[col].isna().any():
            if 'var' in col or 'variance' in col:
                # Variance features: NaN likely means no variation (single value)
                df_filled[col] = df_filled[col].fillna(0.0)
            elif 'consistency' in col or 'ratio' in col:
                # Consistency/ratio features: NaN means perfect consistency or no applicable pattern
                df_filled[col] = df_filled[col].fillna(1.0)
            elif 'entropy' in col:
                # Entropy features: NaN means no entropy (single value)
                df_filled[col] = df_filled[col].fillna(0.0)
            elif 'density' in col or 'freq' in col:
                # Density/frequency features: NaN means no occurrence
                df_filled[col] = df_filled[col].fillna(0.0)
            elif 'mean' in col:
                # Mean features: use overall mean or 0
                overall_mean = df_filled[col].mean()
                df_filled[col] = df_filled[col].fillna(overall_mean if not pd.isna(overall_mean) else 0.0)
            else:
                # Default: fill with 0
                df_filled[col] = df_filled[col].fillna(0.0)
    
    return df_filled


def _handle_nans_with_drop(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Drop columns with too many NaNs (above threshold)."""
    df_cleaned = df.copy()
    
    for col in df_cleaned.columns:
        if col in ['original_index', 'generator', 'label', 'language']:
            continue
            
        nan_ratio = df_cleaned[col].isna().sum() / len(df_cleaned)
        if nan_ratio > threshold:
            df_cleaned = df_cleaned.drop(columns=[col])
            print(f"Dropped column '{col}' (NaN ratio: {nan_ratio:.2f})")
    
    # Fill remaining NaNs
    df_cleaned = _handle_nans_with_fill(df_cleaned)
    
    return df_cleaned


# Example usage and testing
if __name__ == "__main__":
    # Sample data for testing
    sample_data = {
        'code': [
            # Human-written Python (inconsistent spacing)
            """def calculate_sum(a,b):
    if a>0 and b >0:
        result = a+ b
        return result
    else:
        return 0
""",
            # LLM-generated Python (consistent spacing)
            """def calculate_sum(a, b):
    if a > 0 and b > 0:
        result = a + b
        return result
    else:
        return 0
""",
            # Human-written Java (mixed brace style)
            """public class Calculator {
    public int add(int a, int b) 
    {
        if(a>0&&b>0){
            return a+b;
        }
        return 0;
    }
}""",
            # LLM-generated Java (consistent style)
            """public class Calculator {
    public int add(int a, int b) {
        if (a > 0 && b > 0) {
            return a + b;
        }
        return 0;
    }
}"""
        ],
        'generator': ['human', 'gpt-4', 'human', 'claude'],
        'label': [0, 1, 0, 1],
        'language': ['Python', 'Python', 'Java', 'Java']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Extract signatures with different NaN handling strategies
    print("=== Testing different NaN handling strategies ===")
    
    # Strategy 1: Fill NaNs with appropriate defaults
    signature_df_filled = process_dataset(df, handle_nans='fill')
    print(f"\nWith NaN filling:")
    print(f"Shape: {signature_df_filled.shape}")
    print(f"NaN count: {signature_df_filled.isna().sum().sum()}")
    
    # Strategy 2: Drop columns with too many NaNs
    signature_df_dropped = process_dataset(df, handle_nans='drop')
    print(f"\nWith column dropping:")
    print(f"Shape: {signature_df_dropped.shape}")
    print(f"NaN count: {signature_df_dropped.isna().sum().sum()}")
    
    # Strategy 3: Keep NaNs as-is
    signature_df_raw = process_dataset(df, handle_nans='keep')
    print(f"\nWith NaNs kept:")
    print(f"Shape: {signature_df_raw.shape}")
    print(f"NaN count: {signature_df_raw.isna().sum().sum()}")
    
    # Use filled version for analysis
    signature_df = signature_df_filled
    
    # Display results
    print("\n=== Extracted formatting signatures (filled) ===")
    print(signature_df.head())
    
    # Show some key discriminative features
    feature_cols = [col for col in signature_df.columns 
                   if col not in ['original_index', 'generator', 'label', 'language']]
    
    print(f"\nExtracted {len(feature_cols)} formatting features")
    print("Sample features:", feature_cols[:10])
    
    # Analyze NaN patterns by language and feature type
    print("\n=== NaN Analysis (before filling) ===")
    nan_summary = signature_df_raw.isna().sum()
    nan_summary = nan_summary[nan_summary > 0].sort_values(ascending=False)
    if len(nan_summary) > 0:
        print("Features with most NaNs:")
        print(nan_summary.head(10))
        
        # Show which features are language-specific
        print("\nLanguage-specific feature patterns:")
        for lang in signature_df_raw['language'].unique():
            lang_data = signature_df_raw[signature_df_raw['language'] == lang]
            lang_nans = lang_data.isna().sum()
            lang_specific = lang_nans[lang_nans == 0]  # Features with no NaNs in this language
            print(f"{lang}: {len(lang_specific)} features applicable")
    
    # Basic analysis
    print("\n=== Formatting variance comparison (Human vs LLM) ===")
    for lang in signature_df['language'].unique():
        lang_data = signature_df[signature_df['language'] == lang]
        if 'overall_formatting_variance' in lang_data.columns:
            human_var = lang_data[lang_data['label'] == 0]['overall_formatting_variance'].mean()
            llm_var = lang_data[lang_data['label'] == 1]['overall_formatting_variance'].mean()
            print(f"{lang}: Human variance = {human_var:.4f}, LLM variance = {llm_var:.4f}")
    
    # Show most discriminative features
    print("\n=== Most Discriminative Features ===")
    numeric_cols = signature_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['original_index', 'label']]
    
    if len(numeric_cols) > 0:
        human_data = signature_df[signature_df['label'] == 0][numeric_cols]
        llm_data = signature_df[signature_df['label'] == 1][numeric_cols]
        
        # Calculate mean differences
        if len(human_data) > 0 and len(llm_data) > 0:
            mean_diff = abs(human_data.mean() - llm_data.mean()).sort_values(ascending=False)
            print("Top 10 features by mean difference:")
            for feature, diff in mean_diff.head(10).items():
                human_mean = human_data[feature].mean()
                llm_mean = llm_data[feature].mean()
                print(f"  {feature}: Human={human_mean:.3f}, LLM={llm_mean:.3f}, Diff={diff:.3f}")