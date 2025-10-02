import re
import ast
import math
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple, Set
import javalang
import keyword
import nltk
from nltk.corpus import words
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

class VariableNamingEntropyExtractor:
    def __init__(self):
        # Initialize word corpus for semantic analysis
        try:
            self.english_words = set(words.words())
        except:
            self.english_words = set()
            
        # Common programming abbreviations
        self.common_abbrevs = {
            'str': 'string', 'num': 'number', 'cnt': 'count', 'tmp': 'temporary',
            'usr': 'user', 'pwd': 'password', 'cfg': 'config', 'db': 'database',
            'ctx': 'context', 'auth': 'authentication', 'admin': 'administrator',
            'util': 'utility', 'mgr': 'manager', 'ctrl': 'controller',
            'val': 'value', 'var': 'variable', 'obj': 'object', 'arr': 'array',
            'len': 'length', 'max': 'maximum', 'min': 'minimum', 'avg': 'average'
        }
        
        # Technical programming terms
        self.technical_terms = {
            'buffer', 'parser', 'handler', 'factory', 'builder', 'adapter',
            'iterator', 'validator', 'serializer', 'deserializer', 'mapper',
            'repository', 'service', 'controller', 'model', 'view', 'dto',
            'entity', 'exception', 'error', 'response', 'request', 'client',
            'server', 'connection', 'session', 'token', 'cache', 'queue'
        }

    def extract_variables_python(self, code: str) -> Dict[str, List[str]]:
        """Extract variables from Python code using AST"""
        try:
            tree = ast.parse(code)
            variables = {
                'local': [],
                'global': [],
                'parameters': [],
                'class_members': [],
                'function_names': [],
                'class_names': []
            }
            
            class VariableVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.scope_stack = []
                    
                def visit_FunctionDef(self, node):
                    variables['function_names'].append(node.name)
                    # Function parameters
                    for arg in node.args.args:
                        variables['parameters'].append(arg.arg)
                    self.scope_stack.append('function')
                    self.generic_visit(node)
                    self.scope_stack.pop()
                    
                def visit_ClassDef(self, node):
                    variables['class_names'].append(node.name)
                    self.scope_stack.append('class')
                    self.generic_visit(node)
                    self.scope_stack.pop()
                    
                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            if self.scope_stack and self.scope_stack[-1] == 'class':
                                variables['class_members'].append(target.id)
                            elif self.scope_stack:
                                variables['local'].append(target.id)
                            else:
                                variables['global'].append(target.id)
                    self.generic_visit(node)
                    
                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        if not any(node.id in variables[key] for key in variables):
                            if self.scope_stack:
                                variables['local'].append(node.id)
                            else:
                                variables['global'].append(node.id)
            
            visitor = VariableVisitor()
            visitor.visit(tree)
            return variables
            
        except Exception as e:
            # Fallback to regex-based extraction
            return self._regex_extract_variables(code, 'python')

    def extract_variables_java(self, code: str) -> Dict[str, List[str]]:
        """Extract variables from Java code"""
        try:
            # Try using javalang parser
            tree = javalang.parse.parse(code)
            variables = {
                'local': [],
                'global': [],
                'parameters': [],
                'class_members': [],
                'function_names': [],
                'class_names': []
            }
            
            # Extract class names
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                variables['class_names'].append(node.name)
                
            # Extract method names and parameters
            for path, node in tree.filter(javalang.tree.MethodDeclaration):
                variables['function_names'].append(node.name)
                if node.parameters:
                    for param in node.parameters:
                        variables['parameters'].append(param.name)
                        
            # Extract field declarations
            for path, node in tree.filter(javalang.tree.FieldDeclaration):
                for declarator in node.declarators:
                    variables['class_members'].append(declarator.name)
                    
            # Extract local variable declarations
            for path, node in tree.filter(javalang.tree.LocalVariableDeclaration):
                for declarator in node.declarators:
                    variables['local'].append(declarator.name)
                    
            return variables
            
        except Exception as e:
            # Fallback to regex-based extraction
            return self._regex_extract_variables(code, 'java')

    def _regex_extract_variables(self, code: str, language: str) -> Dict[str, List[str]]:
        """Regex-based variable extraction as fallback"""
        variables = {
            'local': [],
            'global': [],
            'parameters': [],
            'class_members': [],
            'function_names': [],
            'class_names': []
        }
        
        # Language-specific patterns
        patterns = {
            'python': {
                'variable': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
                'function': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'class': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]',
                'parameter': r'def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(([^)]*)\)'
            },
            'java': {
                'variable': r'\b(?:int|long|double|float|boolean|String|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'function': r'(?:public|private|protected|static)?\s*(?:void|int|String|boolean|[A-Z][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'class': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'parameter': r'\(([^)]*)\)'
            },
            'javascript': {
                'variable': r'\b(?:var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                'function': r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
                'class': r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                'parameter': r'function\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(([^)]*)\)'
            },
            'cpp': {
                'variable': r'\b(?:int|long|double|float|bool|char|string|auto)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'function': r'(?:\w+\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{',
                'class': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'parameter': r'\(([^)]*)\)'
            }
        }
        
        # Use appropriate patterns based on language
        lang_patterns = patterns.get(language, patterns['python'])
        
        # Extract variables
        if 'variable' in lang_patterns:
            var_matches = re.findall(lang_patterns['variable'], code, re.MULTILINE)
            variables['local'].extend(var_matches)
            
        # Extract function names
        if 'function' in lang_patterns:
            func_matches = re.findall(lang_patterns['function'], code, re.MULTILINE)
            variables['function_names'].extend(func_matches)
            
        # Extract class names
        if 'class' in lang_patterns:
            class_matches = re.findall(lang_patterns['class'], code, re.MULTILINE)
            variables['class_names'].extend(class_matches)
            
        return variables

    def shannon_entropy(self, items: List[str]) -> float:
        """Calculate Shannon entropy of a list of items"""
        if not items:
            return 0.0
        
        counts = Counter(items)
        total = sum(counts.values())
        if total <= 1:
            return 0.0
            
        entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
        return entropy

    def char_level_entropy(self, variable_names: List[str]) -> Dict[str, float]:
        """Calculate character-level entropy within variable names"""
        if not variable_names:
            return {'char_entropy': 0.0, 'char_transition_entropy': 0.0}
            
        all_chars = ''.join(variable_names)
        char_entropy = self.shannon_entropy(list(all_chars))
        
        # Character transition entropy
        transitions = []
        for name in variable_names:
            for i in range(len(name) - 1):
                transitions.append(name[i:i+2])
        
        transition_entropy = self.shannon_entropy(transitions)
        
        return {
            'char_entropy': char_entropy,
            'char_transition_entropy': transition_entropy
        }

    def length_entropy(self, variable_names: List[str]) -> Dict[str, float]:
        """Calculate entropy based on variable name lengths"""
        if not variable_names:
            return {'length_entropy': 0.0, 'avg_length': 0.0, 'length_std': 0.0}
            
        lengths = [len(name) for name in variable_names]
        length_entropy = self.shannon_entropy([str(l) for l in lengths])
        
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_std = math.sqrt(length_variance)
        
        return {
            'length_entropy': length_entropy,
            'avg_length': avg_length,
            'length_std': length_std
        }

    def semantic_category_entropy(self, variable_names: List[str]) -> Dict[str, float]:
        """Classify variables into semantic categories and measure entropy"""
        categories = {
            'descriptive': [],
            'abbreviated': [],
            'single_letter': [],
            'technical': [],
            'domain_specific': [],
            'mixed_case': [],
            'snake_case': [],
            'camel_case': []
        }
        
        for name in variable_names:
            # Length-based categories
            if len(name) == 1:
                categories['single_letter'].append(name)
            elif len(name) <= 3:
                categories['abbreviated'].append(name)
            else:
                categories['descriptive'].append(name)
            
            # Technical terms
            if name.lower() in self.technical_terms:
                categories['technical'].append(name)
            
            # Abbreviations
            if name.lower() in self.common_abbrevs:
                categories['domain_specific'].append(name)
            
            # Naming convention categories
            if '_' in name and name.islower():
                categories['snake_case'].append(name)
            elif re.match(r'^[a-z][a-zA-Z0-9]*$', name) and any(c.isupper() for c in name):
                categories['camel_case'].append(name)
            elif any(c.isupper() for c in name) and any(c.islower() for c in name):
                categories['mixed_case'].append(name)
        
        # Calculate entropy for each category
        category_counts = [len(cat_list) for cat_list in categories.values() if cat_list]
        category_entropy = self.shannon_entropy([str(c) for c in category_counts])
        
        # Calculate proportions
        total = len(variable_names)
        proportions = {}
        for cat_name, cat_list in categories.items():
            proportions[f'{cat_name}_ratio'] = len(cat_list) / total if total > 0 else 0
        
        proportions['semantic_category_entropy'] = category_entropy
        return proportions

    def naming_convention_entropy(self, variable_names: List[str]) -> Dict[str, float]:
        """Analyze naming convention consistency"""
        if not variable_names:
            return {'convention_entropy': 0.0}
            
        conventions = []
        for name in variable_names:
            if re.match(r'^[a-z]+(_[a-z0-9]+)*$', name):
                conventions.append('snake_case')
            elif re.match(r'^[a-z][a-zA-Z0-9]*$', name):
                conventions.append('camelCase')
            elif re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
                conventions.append('PascalCase')
            elif re.match(r'^[A-Z_]+$', name):
                conventions.append('SCREAMING_SNAKE_CASE')
            elif re.match(r'^[a-z]$', name):
                conventions.append('single_lowercase')
            elif re.match(r'^[A-Z]$', name):
                conventions.append('single_uppercase')
            else:
                conventions.append('mixed_or_other')
        
        convention_entropy = self.shannon_entropy(conventions)
        
        # Convention consistency (inverse of entropy, normalized)
        max_entropy = math.log2(len(set(conventions))) if len(set(conventions)) > 1 else 1
        consistency = 1 - (convention_entropy / max_entropy) if max_entropy > 0 else 1
        
        return {
            'convention_entropy': convention_entropy,
            'convention_consistency': consistency
        }

    def abbreviation_pattern_entropy(self, variable_names: List[str]) -> Dict[str, float]:
        """Analyze abbreviation patterns"""
        if not variable_names:
            return {'abbreviation_entropy': 0.0, 'abbreviation_ratio': 0.0}
            
        abbreviation_patterns = []
        abbreviation_count = 0
        
        for name in variable_names:
            # Check if it's a known abbreviation
            if name.lower() in self.common_abbrevs:
                abbreviation_patterns.append('known_abbrev')
                abbreviation_count += 1
            # Check for vowel removal pattern
            elif len([c for c in name if c.lower() in 'aeiou']) < len(name) * 0.3:
                abbreviation_patterns.append('vowel_removal')
                abbreviation_count += 1
            # Check for initialization pattern
            elif len(name) <= 3 and name.isupper():
                abbreviation_patterns.append('initialism')
                abbreviation_count += 1
            # Check for truncation pattern
            elif len(name) < 5 and not name.lower() in self.english_words:
                abbreviation_patterns.append('truncation')
                abbreviation_count += 1
            else:
                abbreviation_patterns.append('full_word')
        
        abbrev_entropy = self.shannon_entropy(abbreviation_patterns)
        abbrev_ratio = abbreviation_count / len(variable_names) if variable_names else 0
        
        return {
            'abbreviation_entropy': abbrev_entropy,
            'abbreviation_ratio': abbrev_ratio
        }

    def scope_based_entropy(self, variables_by_scope: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate entropy patterns across different scopes"""
        scope_entropies = {}
        
        for scope, vars_list in variables_by_scope.items():
            if vars_list:
                scope_entropies[f'{scope}_entropy'] = self.shannon_entropy(vars_list)
                scope_entropies[f'{scope}_count'] = len(vars_list)
            else:
                scope_entropies[f'{scope}_entropy'] = 0.0
                scope_entropies[f'{scope}_count'] = 0
        
        # Cross-scope entropy (how different scopes compare)
        all_scope_entropies = [entropy for key, entropy in scope_entropies.items() 
                              if key.endswith('_entropy')]
        if all_scope_entropies:
            scope_entropies['cross_scope_entropy'] = self.shannon_entropy(
                [str(round(e, 2)) for e in all_scope_entropies]
            )
        else:
            scope_entropies['cross_scope_entropy'] = 0.0
            
        return scope_entropies

    def extract_all_features(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """Extract all variable naming entropy features from code"""
        
        # Extract variables based on language
        if language.lower() == 'python':
            variables_by_scope = self.extract_variables_python(code)
        elif language.lower() in ['java', 'c#']:
            variables_by_scope = self.extract_variables_java(code)
        else:
            # Use regex-based extraction for other languages
            variables_by_scope = self._regex_extract_variables(code, language.lower())
        
        # Collect all variables
        all_variables = []
        for scope_vars in variables_by_scope.values():
            all_variables.extend(scope_vars)
        
        if not all_variables:
            # Return zero features if no variables found
            return self._get_zero_features()
        
        # Extract all features
        features = {}
        
        # 1. Overall entropy
        features['overall_entropy'] = self.shannon_entropy(all_variables)
        features['unique_variables'] = len(set(all_variables))
        features['total_variables'] = len(all_variables)
        features['variable_diversity'] = len(set(all_variables)) / len(all_variables) if all_variables else 0
        
        # 2. Character-level entropy
        char_features = self.char_level_entropy(all_variables)
        features.update(char_features)
        
        # 3. Length-based entropy
        length_features = self.length_entropy(all_variables)
        features.update(length_features)
        
        # 4. Semantic category entropy
        semantic_features = self.semantic_category_entropy(all_variables)
        features.update(semantic_features)
        
        # 5. Naming convention entropy
        convention_features = self.naming_convention_entropy(all_variables)
        features.update(convention_features)
        
        # 6. Abbreviation pattern entropy
        abbrev_features = self.abbreviation_pattern_entropy(all_variables)
        features.update(abbrev_features)
        
        # 7. Scope-based entropy
        scope_features = self.scope_based_entropy(variables_by_scope)
        features.update(scope_features)
        
        return features

    def _get_zero_features(self) -> Dict[str, Any]:
        """Return zero/default features when no variables are found"""
        return {
            'overall_entropy': 0.0,
            'unique_variables': 0,
            'total_variables': 0,
            'variable_diversity': 0.0,
            'char_entropy': 0.0,
            'char_transition_entropy': 0.0,
            'length_entropy': 0.0,
            'avg_length': 0.0,
            'length_std': 0.0,
            'semantic_category_entropy': 0.0,
            'descriptive_ratio': 0.0,
            'abbreviated_ratio': 0.0,
            'single_letter_ratio': 0.0,
            'technical_ratio': 0.0,
            'domain_specific_ratio': 0.0,
            'mixed_case_ratio': 0.0,
            'snake_case_ratio': 0.0,
            'camel_case_ratio': 0.0,
            'convention_entropy': 0.0,
            'convention_consistency': 1.0,
            'abbreviation_entropy': 0.0,
            'abbreviation_ratio': 0.0,
            'local_entropy': 0.0,
            'local_count': 0,
            'global_entropy': 0.0,
            'global_count': 0,
            'parameters_entropy': 0.0,
            'parameters_count': 0,
            'class_members_entropy': 0.0,
            'class_members_count': 0,
            'function_names_entropy': 0.0,
            'function_names_count': 0,
            'class_names_entropy': 0.0,
            'class_names_count': 0,
            'cross_scope_entropy': 0.0
        }

# Example usage and testing
def main():
    extractor = VariableNamingEntropyExtractor()
    
    # Test with Python code
    python_code = """
def calculate_user_score(user_data, config_settings):
    total_points = 0
    bonus_multiplier = 1.5
    user_level = user_data.get('level', 1)
    
    for achievement in user_data['achievements']:
        pts = achievement.score * bonus_multiplier
        total_points += pts
    
    return total_points

class UserManager:
    def __init__(self):
        self.active_users = {}
        self.session_timeout = 3600
        
    def authenticate_user(self, username, pwd):
        # Authentication logic here
        auth_result = self._validate_credentials(username, pwd)
        return auth_result
"""
    
    # Test with Java code
    java_code = """
public class DataProcessor {
    private String configPath;
    private int maxRetries;
    
    public void processUserData(List<User> userList, Configuration config) {
        int processedCount = 0;
        StringBuilder errorLog = new StringBuilder();
        
        for (User u : userList) {
            try {
                String result = processUser(u, config);
                processedCount++;
            } catch (Exception e) {
                errorLog.append(e.getMessage());
            }
        }
    }
}
"""
    
    print("Extracting features from Python code:")
    python_features = extractor.extract_all_features(python_code, 'python')
    for feature, value in python_features.items():
        print(f"{feature}: {value:.4f}" if isinstance(value, float) else f"{feature}: {value}")
    
    print("\n" + "="*50 + "\n")
    
    print("Extracting features from Java code:")
    java_features = extractor.extract_all_features(java_code, 'java')
    for feature, value in java_features.items():
        print(f"{feature}: {value:.4f}" if isinstance(value, float) else f"{feature}: {value}")

if __name__ == "__main__":
    main()