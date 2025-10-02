import re
import tokenize
import io
from collections import Counter
import math

# --- Language-Specific Definitions ---
# Storing keywords and operators for each language makes the system extensible.
LANGUAGE_DEFINITIONS = {
    'python': {
        'keywords': {'for', 'while', 'if', 'else', 'elif', 'def', 'class', 'import', 'from', 'as', 'try', 'except', 'finally', 'with', 'return', 'yield', 'lambda', 'and', 'or', 'not', 'in', 'is'},
        'operators': {'+', '-', '*', '/', '%', '**', '//', '==', '!=', '>', '<', '>=', '<=', '=', '+=', '-=', '*=', '/=', '&=', '|=', '^=', '>>=', '<<='}
    },
    'c++': {
        'keywords': {'for', 'while', 'if', 'else', 'do', 'class', 'struct', 'int', 'double', 'char', 'bool', 'void', 'return', 'const', 'static', 'public', 'private', 'protected', 'namespace', 'using', 'template', 'typename'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '=', '+=', '-=', '*=', '/='}
    },
    'java': {
        'keywords': {'for', 'while', 'if', 'else', 'do', 'class', 'interface', 'int', 'double', 'char', 'boolean', 'void', 'return', 'final', 'static', 'public', 'private', 'protected', 'import', 'package', 'try', 'catch', 'finally', 'throw', 'throws'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '=', '+=', '-=', '*=', '/='}
    },
    # Simplified definitions for other languages to demonstrate the structure
    'javascript': {
        'keywords': {'for', 'while', 'if', 'else', 'function', 'class', 'const', 'let', 'var', 'return', 'import', 'export', 'try', 'catch', 'async', 'await'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '===', '!==', '>', '<', '>=', '<=', '&&', '||', '!', '=', '+=', '-=', '*=', '/='}
    },
    'c': {
        'keywords': {'for', 'while', 'if', 'else', 'do', 'struct', 'int', 'double', 'char', 'void', 'return', 'const', 'static', 'extern'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '=', '+=', '-=', '*=', '/='}
    },
     'c#': {
        'keywords': {'for', 'while', 'if', 'else', 'do', 'class', 'struct', 'int', 'double', 'char', 'bool', 'void', 'return', 'const', 'static', 'public', 'private', 'protected', 'namespace', 'using'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '=', '+=', '-=', '*=', '/='}
    },
    'go': {
        'keywords': {'for', 'if', 'else', 'switch', 'case', 'func', 'struct', 'package', 'import', 'return', 'var', 'const', 'range'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '&', '|', '^', '<<', '>>', ':=', '=', '+=', '-=', '*=', '/='}
    },
    'php': {
        'keywords': {'for', 'while', 'if', 'else', 'elseif', 'do', 'function', 'class', 'public', 'private', 'protected', 'return', 'echo', 'const'},
        'operators': {'+', '-', '*', '/', '%', '++', '--', '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '.', '=', '+=', '-=', '*=', '/='}
    }
}


def general_tokenizer(code, language):
    """
    A robust regex-based tokenizer for languages other than Python.
    This is a simplified approach. For production systems, a proper parser
    like tree-sitter would be more accurate.
    """
    # Regex for common tokens: identifiers, keywords, numbers, operators, strings, comments
    token_specification = [
        ('COMMENT',   r'//[^\n]*|/\*[\s\S]*?\*/|#[^\n]*'),  # Comments for C-style and scripts
        ('STRING',    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''),  # String literals
        ('NUMBER',    r'\b\d+(\.\d*)?([eE][+-]?\d+)?\b'), # Numbers
        ('OP',        r'->|>>=|<<=|===|!==|==|!=|>=|<=|&&|\|\||\+\+|--|[\+\-\*/%&|\^~<>=!.,;:\?\[\]\(\)\{\}]'), # Operators & Separators
        ('ID',        r'[A-Za-z_][A-Za-z0-9_]*'),    # Identifiers
        ('NEWLINE',   r'\n'),                      # Line endings
        ('SKIP',      r'[ \t]+'),                  # Skip over spaces and tabs
        ('MISMATCH',  r'.'),                       # Any other character
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    tokens = []
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind not in ['SKIP', 'MISMATCH', 'COMMENT', 'NEWLINE']:
             tokens.append(value)
    return tokens

def python_tokenizer(code):
    """Uses Python's built-in tokenizer for accurate tokenization."""
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type not in [tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING]:
                tokens.append(tok.string)
    # --- FIX: Added IndentationError to the except block ---
    except (tokenize.TokenError, IndentationError):
        # Fallback for syntactically incorrect code
        return general_tokenizer(code, 'python')
    return tokens


class FeatureExtractor:
    def __init__(self, code, language):
        self.code = code
        self.language = language.lower()
        if self.language not in LANGUAGE_DEFINITIONS:
            raise ValueError(f"Language '{language}' is not supported.")

        if self.language == 'python':
            self.tokens = python_tokenizer(self.code)
        else:
            self.tokens = general_tokenizer(self.code, self.language)
        
        self.token_counts = Counter(self.tokens)
        self.total_tokens = len(self.tokens) if self.tokens else 1

    def _get_ngrams(self, n):
        """Helper to generate n-grams from tokens."""
        return [tuple(self.tokens[i:i+n]) for i in range(len(self.tokens)-n+1)]

    def extract_all_features(self):
        """Run all feature extraction methods and combine results."""
        features = {}
        features.update(self.get_frequency_patterns())
        features.update(self.get_formatting_signatures())
        # features.update(self.get_sequential_patterns())
        features.update(self.get_choice_patterns())
        return features

    def get_frequency_patterns(self):
        """(i) Token-level patterns: Keyword and Operator Frequencies."""
        defs = LANGUAGE_DEFINITIONS[self.language]
        features = {}
        
        # Keyword Frequencies
        total_keywords = 0
        for keyword in defs['keywords']:
            count = self.token_counts.get(keyword, 0)
            features[f'freq_kw_{keyword}'] = count / self.total_tokens
            total_keywords += count
        features['freq_kw_total'] = total_keywords / self.total_tokens
            
        # Operator Frequencies
        total_operators = 0
        for op in defs['operators']:
            count = self.token_counts.get(op, 0)
            features[f'freq_op_{op}'] = count / self.total_tokens
            total_operators += count
        features['freq_op_total'] = total_operators / self.total_tokens
        
        return features

    def get_formatting_signatures(self):
        """(iii) Formatting micro-signatures: Quotes, number separators."""
        features = {}
        # Single vs. Double quotes
        single_quotes = self.code.count("'")
        double_quotes = self.code.count('"')
        total_quotes = single_quotes + double_quotes if (single_quotes + double_quotes) > 0 else 1
        features['ratio_single_quotes'] = single_quotes / total_quotes

        # Use of number separators (e.g., 1_000_000)
        features['uses_number_separator'] = 1 if re.search(r'\d+_\d+', self.code) else 0
        
        # Use of leading dot for floats (e.g., .5)
        features['uses_leading_dot_float'] = 1 if re.search(r'[^\w\d]\.\d+', self.code) else 0

        return features

    def get_sequential_patterns(self):
        """(i) Token-level patterns: N-Grams and specific sequences."""
        features = {}
        
        # Common Bigrams
        # This is a sample. In a real scenario, you'd select the most informative n-grams.
        bigrams = self._get_ngrams(2)
        bigram_counts = Counter(bigrams)
        
        # Example: C-style increment patterns
        features['count_bigram_x_plus_plus'] = bigram_counts.get(('x', '++'), 0) # Placeholder for any identifier
        features['count_bigram_plus_plus_x'] = bigram_counts.get(('++', 'x'), 0)
        
        # Example: Python C-style loop
        if self.language == 'python':
            c_style_loop = 0
            for i, bigram in enumerate(bigrams):
                if bigram == ('in', 'range') and i + 2 < len(self.tokens) and self.tokens[i+2] == 'len':
                    c_style_loop += 1
            features['count_python_c_style_loop'] = c_style_loop

        # Top 5 trigram frequencies (as a simplified example of n-gram features)
        trigrams = self._get_ngrams(3)
        top_5_trigrams = Counter(trigrams).most_common(5)
        for i, (trigram, count) in enumerate(top_5_trigrams):
            features[f'top_trigram_{i+1}_is_{"_".join(trigram)}'] = count / len(trigrams) if trigrams else 0
            
        return features
        
    def get_choice_patterns(self):
        """Patterns related to choosing between semantically equivalent forms."""
        features = {}

        # Redundant boolean checks: `if (var == true)` vs `if (var)`
        features['count_redundant_true_check'] = len(re.findall(r'==\s*(true|True)\b', self.code))
        features['count_redundant_false_check'] = len(re.findall(r'==\s*(false|False)\b', self.code))

        # Redundant parentheses in return statements: `return (x);`
        features['count_redundant_return_parens'] = len(re.findall(r'return\s*\(.*\)\s*;?', self.code))
        
        # Choice of increment operator
        # This is a simplified regex search on code; token sequence analysis is more robust.
        features['count_op_plus_equals_one'] = len(re.findall(r'\+=\s*1\b', self.code))
        features['count_op_equals_plus_one'] = len(re.findall(r'=\s*.*\s*\+\s*1\b', self.code))
        features['count_op_plus_plus'] = self.code.count('++')

        return features


# --- Example Usage ---
if __name__ == '__main__':
    # --- Sample 1: Human-written Python (might be slightly idiosyncratic) ---
    python_code_human = """
import numpy as np

def calculate_average(numbers):
    # simple function to get mean
    total = 0
    for x in numbers:
        total += x
    
    # avoid division by zero
    if len(numbers) == 0:
        return 0
    else:
        return (total / len(numbers)) # redundant parens
    """

    # --- Sample 2: LLM-generated Python (often more "by-the-book") ---
    python_code_llm = """
import numpy as np

def calculate_average(numbers: list) -> float:
    \"\"\"
    Calculates the average of a list of numbers.
    
    Args:
        numbers: A list of numbers.
        
    Returns:
        The average of the numbers, or 0 if the list is empty.
    \"\"\"
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
    """

    # --- Sample 3: C++ code ---
    cpp_code = """
    #include <iostream>
    #include <vector>

    int main() {
        std::vector<int> my_vec;
        for (int i = 0; i < 10; ++i) {
            my_vec.push_back(i);
        }

        if (my_vec.size() > 0 == true) { // redundant boolean check
            std::cout << "Vector is not empty!" << std::endl;
        }
        return(0); // redundant return parens
    }
    """
    
    print("--- Analyzing Human-written Python Code ---")
    extractor_human = FeatureExtractor(python_code_human, 'python')
    features_human = extractor_human.extract_all_features()
    # Print a subset of interesting features
    for key in ['freq_kw_for', 'freq_kw_else', 'count_redundant_return_parens', 'ratio_single_quotes']:
        if key in features_human:
            print(f"{key}: {features_human[key]:.4f}")
    
    print("\n--- Analyzing LLM-generated Python Code ---")
    extractor_llm = FeatureExtractor(python_code_llm, 'python')
    features_llm = extractor_llm.extract_all_features()
    for key in ['freq_kw_for', 'freq_kw_else', 'count_redundant_return_parens', 'ratio_single_quotes']:
        if key in features_llm:
            print(f"{key}: {features_llm[key]:.4f}")

    print("\n--- Analyzing C++ Code ---")
    extractor_cpp = FeatureExtractor(cpp_code, 'c++')
    features_cpp = extractor_cpp.extract_all_features()
    for key in ['freq_kw_for', 'count_redundant_true_check', 'count_redundant_return_parens', 'freq_op_++']:
        if key in features_cpp:
            print(f"{key}: {features_cpp[key]:.4f}")

    # You can get all features as a dictionary like this:
    # all_cpp_features = extractor_cpp.extract_all_features()
    # print(f"\nTotal features extracted for C++ code: {len(all_cpp_features)}")
