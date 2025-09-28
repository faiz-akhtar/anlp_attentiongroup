import re
import ast
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
from textblob import TextBlob
import javalang
from tree_sitter import Language, Parser
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CommentInfo:
    """Container for parsed comment information"""
    text: str
    line_number: int
    comment_type: str  # 'inline', 'block', 'docstring', 'header'
    indentation: int
    length: int

class CommentParser:
    """Multi-language comment parser"""
    
    def __init__(self):
        # Language-specific comment patterns
        self.patterns = {
            'python': {
                'single': r'#.*?$',
                'multi': r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'',
                'docstring': r'(?:def|class).*?:\s*(?:\n\s*)?(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
            },
            'java': {
                'single': r'//.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'/\*\*[\s\S]*?\*/'
            },
            'cpp': {
                'single': r'//.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'/\*\*[\s\S]*?\*/'
            },
            'c': {
                'single': r'//.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'/\*\*[\s\S]*?\*/'
            },
            'csharp': {
                'single': r'//.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'///.*?$|/\*\*[\s\S]*?\*/'
            },
            'javascript': {
                'single': r'//.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'/\*\*[\s\S]*?\*/'
            },
            'php': {
                'single': r'//.*?$|#.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'/\*\*[\s\S]*?\*/'
            },
            'go': {
                'single': r'//.*?$',
                'multi': r'/\*[\s\S]*?\*/',
                'docstring': r'//.*?(?:\n//.*?)*'
            }
        }
    
    def parse_comments(self, code: str, language: str) -> List[CommentInfo]:
        """Extract all comments from code"""
        language = language.lower()
        if language not in self.patterns:
            raise ValueError(f"Unsupported language: {language}")
        
        comments = []
        lines = code.split('\n')
        
        # Find single-line comments
        single_pattern = self.patterns[language]['single']
        for i, line in enumerate(lines):
            matches = re.finditer(single_pattern, line, re.MULTILINE)
            for match in matches:
                comments.append(CommentInfo(
                    text=match.group().strip(),
                    line_number=i + 1,
                    comment_type='inline' if line.strip() and not line.strip().startswith(('//','#')) else 'single',
                    indentation=len(line) - len(line.lstrip()),
                    length=len(match.group().strip())
                ))
        
        # Find multi-line comments
        multi_pattern = self.patterns[language]['multi']
        for match in re.finditer(multi_pattern, code, re.MULTILINE | re.DOTALL):
            start_line = code[:match.start()].count('\n') + 1
            comment_text = match.group()
            
            # Determine if it's a docstring
            is_docstring = False
            if language == 'python':
                # Check if it's after a function/class definition
                lines_before = code[:match.start()].split('\n')
                for line in reversed(lines_before[-5:]):  # Check last 5 lines
                    if re.match(r'\s*(def|class)\s+', line):
                        is_docstring = True
                        break
                    elif line.strip() and not line.strip().startswith('#'):
                        break
            elif 'docstring' in self.patterns[language]:
                is_docstring = bool(re.match(self.patterns[language]['docstring'], match.group()))
            
            comments.append(CommentInfo(
                text=comment_text.strip(),
                line_number=start_line,
                comment_type='docstring' if is_docstring else 'block',
                indentation=self._get_indentation_from_match(code, match),
                length=len(comment_text.strip())
            ))
        
        return comments
    
    def _get_indentation_from_match(self, code: str, match) -> int:
        """Get indentation level of a matched comment"""
        lines_before = code[:match.start()].split('\n')
        if lines_before:
            last_line = lines_before[-1]
            return len(last_line) - len(last_line.lstrip())
        return 0

class CommentIntegrationExtractor:
    """Extract comment integration characteristics for ML detection"""
    
    def __init__(self):
        self.parser = CommentParser()
        self.feature_names = []
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Initialize feature names for consistent output"""
        self.feature_names = [
            # Density features
            'comment_density', 'inline_comment_density', 'block_comment_density',
            'docstring_density', 'function_comment_coverage', 'class_comment_coverage',
            
            # Distribution features
            'comment_clustering_coefficient', 'avg_comment_distance',
            'comment_position_variance', 'header_comment_ratio',
            
            # Length and verbosity features
            'avg_comment_length', 'comment_length_std', 'comment_to_code_ratio',
            'max_comment_length', 'min_comment_length', 'median_comment_length',
            
            # Placement patterns
            'end_of_line_ratio', 'standalone_comment_ratio', 'indentation_consistency',
            'comment_code_alignment_score', 'proximity_to_complexity_score',
            
            # Content characteristics
            'avg_words_per_comment', 'sentence_completeness_ratio', 'personal_pronoun_ratio',
            'technical_term_density', 'redundancy_score', 'abstraction_level_score',
            
            # Language-specific features
            'docstring_format_compliance', 'todo_fixme_ratio', 'external_reference_ratio',
            'ascii_art_ratio', 'comment_hierarchy_score', 'temporal_marker_ratio',
            
            # Semantic features
            'sentiment_polarity_avg', 'sentiment_polarity_std', 'semantic_consistency_score',
            'explanation_depth_score', 'code_comment_similarity_avg'
        ]
    
    def extract_features(self, code: str, language: str) -> Dict[str, float]:
        """Extract all comment integration features"""
        try:
            comments = self.parser.parse_comments(code, language)
            lines = code.split('\n')
            
            features = {}
            
            # Basic counts
            total_lines = len(lines)
            code_lines = self._count_code_lines(lines, language)
            total_comments = len(comments)
            
            if total_comments == 0:
                return self._zero_features()
            
            # Extract all feature categories
            features.update(self._extract_density_features(comments, total_lines, code_lines, code, language))
            features.update(self._extract_distribution_features(comments, total_lines))
            features.update(self._extract_length_features(comments))
            features.update(self._extract_placement_features(comments, lines))
            features.update(self._extract_content_features(comments))
            features.update(self._extract_language_specific_features(comments, language))
            features.update(self._extract_semantic_features(comments, code))
            
            return features
            
        except Exception as e:
            print(f"Error processing code: {e}")
            return self._zero_features()
    
    def _zero_features(self) -> Dict[str, float]:
        """Return zero-filled feature dictionary"""
        return {name: 0.0 for name in self.feature_names}
    
    def _count_code_lines(self, lines: List[str], language: str) -> int:
        """Count actual code lines (excluding comments and empty lines)"""
        code_lines = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not self._is_comment_line(stripped, language):
                code_lines += 1
        return code_lines
    
    def _is_comment_line(self, line: str, language: str) -> bool:
        """Check if a line is primarily a comment"""
        language = language.lower()
        if language == 'python':
            return line.startswith('#')
        elif language in ['java', 'cpp', 'c', 'csharp', 'javascript']:
            return line.startswith('//') or line.startswith('/*') or line.startswith('*')
        elif language == 'php':
            return line.startswith('//') or line.startswith('#') or line.startswith('/*')
        elif language == 'go':
            return line.startswith('//') or line.startswith('/*')
        return False
    
    def _extract_density_features(self, comments: List[CommentInfo], total_lines: int, code_lines: int, code: str, language: str) -> Dict[str, float]:
        """Extract comment density features"""
        features = {}
        
        # Basic densities
        features['comment_density'] = len(comments) / max(total_lines, 1)
        
        # Type-specific densities
        inline_comments = [c for c in comments if c.comment_type == 'inline']
        block_comments = [c for c in comments if c.comment_type == 'block']
        docstring_comments = [c for c in comments if c.comment_type == 'docstring']
        
        features['inline_comment_density'] = len(inline_comments) / max(code_lines, 1)
        features['block_comment_density'] = len(block_comments) / max(total_lines, 1)
        features['docstring_density'] = len(docstring_comments) / max(total_lines, 1)
        
        # Function/class coverage
        function_count = self._count_functions(code, language)
        class_count = self._count_classes(code, language)
        
        features['function_comment_coverage'] = len(docstring_comments) / max(function_count, 1)
        features['class_comment_coverage'] = len(docstring_comments) / max(class_count, 1)
        
        return features
    
    def _extract_distribution_features(self, comments: List[CommentInfo], total_lines: int) -> Dict[str, float]:
        """Extract comment distribution features"""
        features = {}
        
        if not comments:
            features.update({
                'comment_clustering_coefficient': 0.0,
                'avg_comment_distance': 0.0,
                'comment_position_variance': 0.0,
                'header_comment_ratio': 0.0
            })
            return features
        
        # Comment line positions
        positions = [c.line_number for c in comments]
        
        # Clustering coefficient
        features['comment_clustering_coefficient'] = self._calculate_clustering_coefficient(positions)
        
        # Average distance between consecutive comments
        if len(positions) > 1:
            distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            features['avg_comment_distance'] = np.mean(distances)
        else:
            features['avg_comment_distance'] = total_lines
        
        # Position variance (normalized)
        features['comment_position_variance'] = np.var([p/total_lines for p in positions])
        
        # Header comments (first 10 lines)
        header_comments = len([c for c in comments if c.line_number <= 10])
        features['header_comment_ratio'] = header_comments / len(comments)
        
        return features
    
    def _extract_length_features(self, comments: List[CommentInfo]) -> Dict[str, float]:
        """Extract comment length and verbosity features"""
        features = {}
        
        if not comments:
            return {
                'avg_comment_length': 0.0,
                'comment_length_std': 0.0,
                'comment_to_code_ratio': 0.0,
                'max_comment_length': 0.0,
                'min_comment_length': 0.0,
                'median_comment_length': 0.0
            }
        
        lengths = [c.length for c in comments]
        
        features['avg_comment_length'] = np.mean(lengths)
        features['comment_length_std'] = np.std(lengths)
        features['max_comment_length'] = max(lengths)
        features['min_comment_length'] = min(lengths)
        features['median_comment_length'] = np.median(lengths)
        
        # Comment to code ratio (total comment chars / total code chars)
        total_comment_chars = sum(lengths)
        # Approximate code characters (this could be improved)
        features['comment_to_code_ratio'] = total_comment_chars / max(total_comment_chars + 1000, 1000)
        
        return features
    
    def _extract_placement_features(self, comments: List[CommentInfo], lines: List[str]) -> Dict[str, float]:
        """Extract comment placement pattern features"""
        features = {}
        
        if not comments:
            return {
                'end_of_line_ratio': 0.0,
                'standalone_comment_ratio': 0.0,
                'indentation_consistency': 0.0,
                'comment_code_alignment_score': 0.0,
                'proximity_to_complexity_score': 0.0
            }
        
        # End-of-line vs standalone comments
        inline_count = len([c for c in comments if c.comment_type == 'inline'])
        features['end_of_line_ratio'] = inline_count / len(comments)
        features['standalone_comment_ratio'] = 1.0 - features['end_of_line_ratio']
        
        # Indentation consistency
        indentations = [c.indentation for c in comments if c.comment_type != 'inline']
        if indentations:
            features['indentation_consistency'] = 1.0 / (1.0 + np.std(indentations))
        else:
            features['indentation_consistency'] = 1.0
        
        # Code alignment score (how well comments align with code structure)
        features['comment_code_alignment_score'] = self._calculate_alignment_score(comments, lines)
        
        # Proximity to complexity score
        features['proximity_to_complexity_score'] = self._calculate_complexity_proximity_score(comments, lines)
        
        return features
    
    def _extract_content_features(self, comments: List[CommentInfo]) -> Dict[str, float]:
        """Extract comment content characteristics"""
        features = {}
        
        if not comments:
            return {
                'avg_words_per_comment': 0.0,
                'sentence_completeness_ratio': 0.0,
                'personal_pronoun_ratio': 0.0,
                'technical_term_density': 0.0,
                'redundancy_score': 0.0,
                'abstraction_level_score': 0.0
            }
        
        all_text = ' '.join([c.text for c in comments])
        comment_texts = [self._clean_comment_text(c.text) for c in comments]
        
        # Word count features
        word_counts = [len(text.split()) for text in comment_texts if text]
        features['avg_words_per_comment'] = np.mean(word_counts) if word_counts else 0.0
        
        # Sentence completeness (comments ending with proper punctuation)
        complete_sentences = sum(1 for text in comment_texts 
                               if text and text.strip().endswith(('.', '!', '?')))
        features['sentence_completeness_ratio'] = complete_sentences / len(comments)
        
        # Personal pronouns
        personal_pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
        pronoun_count = sum(all_text.lower().count(pronoun) for pronoun in personal_pronouns)
        total_words = len(all_text.split())
        features['personal_pronoun_ratio'] = pronoun_count / max(total_words, 1)
        
        # Technical term density (approximate)
        technical_terms = ['function', 'method', 'class', 'variable', 'parameter', 'return',
                          'algorithm', 'implementation', 'optimization', 'complexity']
        tech_count = sum(all_text.lower().count(term) for term in technical_terms)
        features['technical_term_density'] = tech_count / max(total_words, 1)
        
        # Redundancy score (placeholder - could be improved with NLP)
        features['redundancy_score'] = self._calculate_redundancy_score(comment_texts)
        
        # Abstraction level score
        features['abstraction_level_score'] = self._calculate_abstraction_score(comment_texts)
        
        return features
    
    def _extract_language_specific_features(self, comments: List[CommentInfo], language: str) -> Dict[str, float]:
        """Extract language-specific comment features"""
        features = {}
        
        if not comments:
            return {
                'docstring_format_compliance': 0.0,
                'todo_fixme_ratio': 0.0,
                'external_reference_ratio': 0.0,
                'ascii_art_ratio': 0.0,
                'comment_hierarchy_score': 0.0,
                'temporal_marker_ratio': 0.0
            }
        
        all_text = ' '.join([c.text for c in comments])
        
        # Docstring format compliance
        features['docstring_format_compliance'] = self._check_docstring_compliance(comments, language)
        
        # TODO/FIXME markers
        todo_markers = ['todo', 'fixme', 'hack', 'bug', 'note', 'warning']
        marker_count = sum(all_text.lower().count(marker) for marker in todo_markers)
        features['todo_fixme_ratio'] = marker_count / len(comments)
        
        # External references (URLs, issue numbers, etc.)
        url_pattern = r'https?://|www\.|github\.com|stackoverflow\.com'
        issue_pattern = r'#\d+|issue\s*\d+|ticket\s*\d+'
        external_refs = len(re.findall(url_pattern, all_text, re.IGNORECASE)) + \
                       len(re.findall(issue_pattern, all_text, re.IGNORECASE))
        features['external_reference_ratio'] = external_refs / len(comments)
        
        # ASCII art / decorative comments
        ascii_patterns = [r'[=\-_\*]{3,}', r'\/\*+', r'\*+\/', r'#+']
        ascii_count = sum(len(re.findall(pattern, c.text)) for c in comments for pattern in ascii_patterns)
        features['ascii_art_ratio'] = ascii_count / len(comments)
        
        # Comment hierarchy (different comment styles)
        comment_types = set(c.comment_type for c in comments)
        features['comment_hierarchy_score'] = len(comment_types) / 4.0  # Max 4 types
        
        # Temporal markers
        temporal_patterns = ['yesterday', 'today', 'tomorrow', 'last week', 'next week', 
                           r'\d{4}-\d{2}-\d{2}', r'\d{1,2}/\d{1,2}/\d{2,4}']
        temporal_count = sum(len(re.findall(pattern, all_text, re.IGNORECASE)) 
                           for pattern in temporal_patterns)
        features['temporal_marker_ratio'] = temporal_count / len(comments)
        
        return features
    
    def _extract_semantic_features(self, comments: List[CommentInfo], code: str) -> Dict[str, float]:
        """Extract semantic features using basic NLP"""
        features = {}
        
        if not comments:
            return {
                'sentiment_polarity_avg': 0.0,
                'sentiment_polarity_std': 0.0,
                'semantic_consistency_score': 0.0,
                'explanation_depth_score': 0.0,
                'code_comment_similarity_avg': 0.0
            }
        
        try:
            comment_texts = [self._clean_comment_text(c.text) for c in comments if c.text.strip()]
            
            # Sentiment analysis
            sentiments = []
            for text in comment_texts:
                if text:
                    blob = TextBlob(text)
                    sentiments.append(blob.sentiment.polarity)
            
            if sentiments:
                features['sentiment_polarity_avg'] = np.mean(sentiments)
                features['sentiment_polarity_std'] = np.std(sentiments)
            else:
                features['sentiment_polarity_avg'] = 0.0
                features['sentiment_polarity_std'] = 0.0
            
            # Semantic consistency (placeholder)
            features['semantic_consistency_score'] = self._calculate_semantic_consistency(comment_texts)
            
            # Explanation depth
            features['explanation_depth_score'] = self._calculate_explanation_depth(comment_texts)
            
            # Code-comment similarity (simplified)
            features['code_comment_similarity_avg'] = self._calculate_code_comment_similarity(comment_texts, code)
            
        except Exception:
            # Fallback if TextBlob fails
            features.update({
                'sentiment_polarity_avg': 0.0,
                'sentiment_polarity_std': 0.0,
                'semantic_consistency_score': 0.0,
                'explanation_depth_score': 0.0,
                'code_comment_similarity_avg': 0.0
            })
        
        return features
    
    # Helper methods
    def _count_functions(self, code: str, language: str) -> int:
        """Count functions in code"""
        language = language.lower()
        if language == 'python':
            return len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
        elif language == 'java':
            return len(re.findall(r'(?:public|private|protected)?\s*\w+\s+\w+\s*\([^)]*\)\s*{', code))
        elif language in ['cpp', 'c', 'csharp']:
            return len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*{', code))
        elif language == 'javascript':
            return len(re.findall(r'function\s+\w+|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\(', code))
        elif language == 'php':
            return len(re.findall(r'function\s+\w+', code))
        elif language == 'go':
            return len(re.findall(r'func\s+\w+', code))
        return 1
    
    def _count_classes(self, code: str, language: str) -> int:
        """Count classes in code"""
        language = language.lower()
        if language == 'python':
            return len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
        elif language == 'java':
            return len(re.findall(r'(?:public|private)?\s*class\s+\w+', code))
        elif language in ['cpp', 'csharp']:
            return len(re.findall(r'class\s+\w+', code))
        elif language == 'javascript':
            return len(re.findall(r'class\s+\w+', code))
        elif language == 'php':
            return len(re.findall(r'class\s+\w+', code))
        return max(1, len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE)))
    
    def _clean_comment_text(self, comment: str) -> str:
        """Clean comment text by removing comment markers"""
        # Remove common comment markers
        comment = re.sub(r'^//+\s*', '', comment)
        comment = re.sub(r'^#+\s*', '', comment)
        comment = re.sub(r'^/\*+\s*|\s*\*+/$', '', comment)
        comment = re.sub(r'^\*+\s*', '', comment, flags=re.MULTILINE)
        comment = re.sub(r'^"""|"""$', '', comment, flags=re.MULTILINE)
        comment = re.sub(r"^'''|'''$", '', comment, flags=re.MULTILINE)
        return comment.strip()
    
    def _calculate_clustering_coefficient(self, positions: List[int]) -> float:
        """Calculate how clustered comments are"""
        if len(positions) < 2:
            return 0.0
        
        positions = sorted(positions)
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        
        # Clustering coefficient based on gap variance
        if len(gaps) > 1:
            return 1.0 / (1.0 + np.std(gaps) / np.mean(gaps))
        return 1.0
    
    def _calculate_alignment_score(self, comments: List[CommentInfo], lines: List[str]) -> float:
        """Calculate how well comments align with code structure"""
        if not comments:
            return 0.0
        
        aligned_count = 0
        for comment in comments:
            # Check if comment is near structural elements
            line_idx = comment.line_number - 1
            if 0 <= line_idx < len(lines):
                context_lines = lines[max(0, line_idx-2):min(len(lines), line_idx+3)]
                context = ' '.join(context_lines).lower()
                
                # Check for structural keywords
                if any(keyword in context for keyword in 
                       ['def ', 'class ', 'function', 'if ', 'for ', 'while ', 'try ', 'catch']):
                    aligned_count += 1
        
        return aligned_count / len(comments)
    
    def _calculate_complexity_proximity_score(self, comments: List[CommentInfo], lines: List[str]) -> float:
        """Calculate proximity of comments to complex code structures"""
        if not comments:
            return 0.0
        
        complexity_keywords = ['for', 'while', 'if', 'try', 'catch', 'switch', 'case']
        proximity_scores = []
        
        for comment in comments:
            line_idx = comment.line_number - 1
            min_distance = float('inf')
            
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in complexity_keywords):
                    distance = abs(i - line_idx)
                    min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                proximity_scores.append(1.0 / (1.0 + min_distance))
            else:
                proximity_scores.append(0.0)
        
        return np.mean(proximity_scores) if proximity_scores else 0.0
    
    def _calculate_redundancy_score(self, comment_texts: List[str]) -> float:
        """Calculate comment redundancy score"""
        if not comment_texts:
            return 0.0
        
        # Simple redundancy based on repeated phrases
        all_words = []
        for text in comment_texts:
            if text:
                words = text.lower().split()
                all_words.extend(words)
        
        if not all_words:
            return 0.0
        
        word_counts = Counter(all_words)
        total_words = len(all_words)
        unique_words = len(word_counts)
        
        # Redundancy as inverse of vocabulary diversity
        return 1.0 - (unique_words / max(total_words, 1))
    
    def _calculate_abstraction_score(self, comment_texts: List[str]) -> float:
        """Calculate abstraction level of comments"""
        if not comment_texts:
            return 0.0
        
        high_level_words = ['purpose', 'goal', 'intent', 'design', 'architecture', 'concept',
                           'approach', 'strategy', 'overview', 'summary', 'why', 'rationale']
        low_level_words = ['variable', 'increment', 'decrement', 'assign', 'initialize',
                          'loop', 'iterate', 'index', 'counter', 'pointer']
        
        all_text = ' '.join(comment_texts).lower()
        
        high_level_count = sum(all_text.count(word) for word in high_level_words)
        low_level_count = sum(all_text.count(word) for word in low_level_words)
        
        total_indicator_words = high_level_count + low_level_count
        if total_indicator_words == 0:
            return 0.5  # Neutral
        
        return high_level_count / total_indicator_words
    
    def _check_docstring_compliance(self, comments: List[CommentInfo], language: str) -> float:
        """Check docstring format compliance"""
        docstrings = [c for c in comments if c.comment_type == 'docstring']
        if not docstrings:
            return 0.0
        
        compliant_count = 0
        for docstring in docstrings:
            text = docstring.text.lower()
            
            # Basic compliance checks
            has_description = len(text.strip()) > 10
            has_parameters = 'param' in text or 'arg' in text or '@param' in text
            has_return = 'return' in text or '@return' in text
            
            compliance_score = sum([has_description, has_parameters, has_return]) / 3.0
            if compliance_score > 0.33:  # At least one criteria met
                compliant_count += 1
        
        return compliant_count / len(docstrings)
    
    def _calculate_semantic_consistency(self, comment_texts: List[str]) -> float:
        """Calculate semantic consistency across comments"""
        if len(comment_texts) < 2:
            return 1.0
        
        # Simple consistency based on vocabulary overlap
        vocabularies = []
        for text in comment_texts:
            if text:
                words = set(text.lower().split())
                vocabularies.append(words)
        
        if not vocabularies:
            return 0.0
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(vocabularies)):
            for j in range(i + 1, len(vocabularies)):
                intersection = len(vocabularies[i] & vocabularies[j])
                union = len(vocabularies[i] | vocabularies[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_explanation_depth(self, comment_texts: List[str]) -> float:
        """Calculate depth of explanation in comments"""
        if not comment_texts:
            return 0.0
        
        depth_indicators = ['because', 'since', 'due to', 'reason', 'why', 'how', 'what',
                           'algorithm', 'approach', 'method', 'technique', 'strategy',
                           'implementation', 'detail', 'step', 'process', 'procedure']
        
        depth_scores = []
        for text in comment_texts:
            if text:
                words = text.lower().split()
                depth_count = sum(1 for word in words if word in depth_indicators)
                depth_scores.append(depth_count / max(len(words), 1))
        
        return np.mean(depth_scores) if depth_scores else 0.0
    
    def _calculate_code_comment_similarity(self, comment_texts: List[str], code: str) -> float:
        """Calculate similarity between comments and code"""
        if not comment_texts:
            return 0.0
        
        # Extract identifiers from code (simplified)
        code_words = set()
        # Basic identifier extraction
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        code_words.update(word.lower() for word in identifiers)
        
        # Extract words from comments
        comment_words = set()
        for text in comment_texts:
            if text:
                words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text)
                comment_words.update(word.lower() for word in words)
        
        # Calculate Jaccard similarity
        if not code_words or not comment_words:
            return 0.0
        
        intersection = len(code_words & comment_words)
        union = len(code_words | comment_words)
        
        return intersection / union if union > 0 else 0.0


class CommentFeatureProcessor:
    """Process multiple code samples and extract features for ML training"""
    
    def __init__(self):
        self.extractor = CommentIntegrationExtractor()
    
    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a pandas DataFrame with code samples"""
        if not all(col in df.columns for col in ['code', 'language']):
            raise ValueError("DataFrame must contain 'code' and 'language' columns")
        
        features_list = []
        
        for idx, row in df.iterrows():
            print(f"Processing sample {idx + 1}/{len(df)}")
            
            try:
                features = self.extractor.extract_features(row['code'], row['language'])
                features['sample_id'] = idx
                features_list.append(features)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                # Add zero features for failed samples
                zero_features = self.extractor._zero_features()
                zero_features['sample_id'] = idx
                features_list.append(zero_features)
        
        features_df = pd.DataFrame(features_list)
        
        # Merge with original data
        result_df = df.copy()
        for col in features_df.columns:
            if col != 'sample_id':
                result_df[f'comment_{col}'] = features_df[col]
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return [f'comment_{name}' for name in self.extractor.feature_names]
    
    def save_features(self, df: pd.DataFrame, filepath: str):
        """Save processed features to file"""
        df.to_csv(filepath, index=False)
        print(f"Features saved to {filepath}")
    
    def load_features(self, filepath: str) -> pd.DataFrame:
        """Load processed features from file"""
        return pd.read_csv(filepath)


# Example usage and testing
def main():
    """Example usage of the Comment Feature Extractor"""
    
    # Sample data
    sample_data = {
        'code': [
            '''
def calculate_fibonacci(n):
    """
    Calculate the nth Fibonacci number using dynamic programming.
    
    Args:
        n (int): The position in the Fibonacci sequence
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")  # Input validation
    
    # Base cases
    if n <= 1:
        return n
    
    # Dynamic programming approach for efficiency
    dp = [0] * (n + 1)
    dp[1] = 1
    
    # Fill the dp array
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]  # Fibonacci recurrence
    
    return dp[n]
            ''',
            '''
public class QuickSort {
    public static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }
    
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = (low - 1);
        
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        
        return i + 1;
    }
}
            ''',
            '''
# Machine learning model for text classification
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class TextClassifier:
    def __init__(self):
        # Initialize the classification pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
            ('classifier', MultinomialNB(alpha=1.0))
        ])
        self.is_trained = False
    
    def train(self, texts, labels):
        # Train the model on the provided data
        self.pipeline.fit(texts, labels)
        self.is_trained = True
    
    def predict(self, texts):
        # Make predictions on new texts
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.pipeline.predict(texts)
            '''
        ],
        'language': ['python', 'java', 'python'],
        'generator': ['human', 'llm_gpt4', 'human'],
        'label': [0, 1, 0]
    }
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Initialize processor
    processor = CommentFeatureProcessor()
    
    # Process the dataset
    print("Processing dataset...")
    result_df = processor.process_dataset(df)
    
    # Display results
    print("\nFeature extraction completed!")
    print(f"Original columns: {len(df.columns)}")
    print(f"Total columns after feature extraction: {len(result_df.columns)}")
    
    # Show some key features
    feature_cols = processor.get_feature_names()
    print(f"\nExtracted {len(feature_cols)} comment features:")
    
    # Display first few features for each sample
    key_features = ['comment_comment_density', 'comment_avg_comment_length', 
                   'comment_docstring_density', 'comment_technical_term_density',
                   'comment_personal_pronoun_ratio']
    
    print("\nKey features for each sample:")
    for i, row in result_df.iterrows():
        print(f"\nSample {i+1} ({row['generator']}):")
        for feature in key_features:
            if feature in result_df.columns:
                print(f"  {feature}: {row[feature]:.4f}")
    
    # Save results
    output_file = "comment_features_output.csv"
    processor.save_features(result_df, output_file)
    
    print(f"\nComplete feature set saved to {output_file}")
    print("Feature extraction pipeline ready for ML training!")

if __name__ == "__main__":
    main()