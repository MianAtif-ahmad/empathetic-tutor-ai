# config_loader.py - Configuration Management System
import json
import os
from typing import Dict, List, Any
from pathlib import Path

class FrustrationConfigLoader:
    """
    Loads and manages frustration detection configurations from JSON files.
    Optimized for performance with caching and lazy loading.
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._cache = {}
        self._ensure_config_files()
    
    def _manage_cache_size(self):
        """Manage cache size to prevent memory leaks"""
        if len(self._cache) > 50:  # Max 50 cached items
            # Keep only the 25 most recently used items
            cache_items = list(self._cache.items())
            self._cache = dict(cache_items[-25:])
            print(f"🧹 Cache cleaned: kept 25 most recent items")
    
    def clear_cache(self):
        """Manually clear configuration cache"""
        self._cache.clear()
        print("🧹 Configuration cache cleared")
    
    def _ensure_config_files(self):
        """Create default config files if they don't exist"""
        default_configs = {
            "emotional_patterns.json": self._get_default_emotional_patterns(),
            "keyword_weights.json": self._get_default_keyword_weights(),
            "concept_keywords.json": self._get_default_concept_keywords(),
            "detection_settings.json": self._get_default_detection_settings()
        }
        
        for filename, default_data in default_configs.items():
            filepath = self.config_dir / filename
            if not filepath.exists():
                self._save_config(filepath, default_data)
                print(f"✅ Created default config: {filepath}")
    
    def _save_config(self, filepath: Path, data: Dict):
        """Save configuration to JSON file with pretty formatting"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_config(self, filename: str) -> Dict:
        """Load configuration from JSON file with caching"""
        if filename in self._cache:
            return self._cache[filename]
        
        filepath = self.config_dir / filename
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._cache[filename] = data
                self._manage_cache_size()
                return data
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {filepath}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {filepath}: {e}")
            return {}
    
    def reload_config(self, filename: str = None):
        """Reload configuration from disk (clears cache)"""
        if filename:
            self._cache.pop(filename, None)
        else:
            self._cache.clear()
        print(f"🔄 Reloaded config: {filename or 'all files'}")
    
    def get_emotional_patterns(self) -> Dict[str, float]:
        """Get emotional patterns with their frustration weights"""
        return self._load_config("emotional_patterns.json")
    
    def get_keyword_weights(self) -> Dict[str, Dict[str, Any]]:
        """Get keyword categories with weights and patterns"""
        return self._load_config("keyword_weights.json")
    
    def get_concept_keywords(self) -> Dict[str, List[str]]:
        """Get programming concept detection keywords"""
        return self._load_config("concept_keywords.json")
    
    def get_detection_settings(self) -> Dict[str, Any]:
        """Get detection algorithm settings and thresholds"""
        return self._load_config("detection_settings.json")
    
    def update_emotional_pattern(self, pattern: str, weight: float):
        """Add or update an emotional pattern"""
        patterns = self.get_emotional_patterns()
        patterns[pattern] = weight
        self._save_config(self.config_dir / "emotional_patterns.json", patterns)
        self.reload_config("emotional_patterns.json")
        print(f"✅ Updated emotional pattern: '{pattern}' = {weight}")
    
    def add_concept_keyword(self, concept: str, keyword: str):
        """Add a keyword to a concept category"""
        concepts = self.get_concept_keywords()
        if concept not in concepts:
            concepts[concept] = []
        if keyword not in concepts[concept]:
            concepts[concept].append(keyword)
            self._save_config(self.config_dir / "concept_keywords.json", concepts)
            self.reload_config("concept_keywords.json")
            print(f"✅ Added keyword '{keyword}' to concept '{concept}'")
    
    def get_stats(self) -> Dict[str, int]:
        """Get configuration statistics"""
        return {
            "emotional_patterns": len(self.get_emotional_patterns()),
            "keyword_categories": len(self.get_keyword_weights()),
            "concept_categories": len(self.get_concept_keywords()),
            "total_concept_keywords": sum(len(keywords) for keywords in self.get_concept_keywords().values())
        }
    
    # Default configuration data
    def _get_default_emotional_patterns(self) -> Dict[str, float]:
        return {
            "this is impossible": 4.0,
            "i can't do this": 3.5,
            "nothing works": 3.0,
            "i hate this": 3.0,
            "so frustrated": 2.5,
            "giving up": 4.0,
            "want to quit": 3.5,
            "this sucks": 2.5,
            "i'm lost": 2.0,
            "completely confused": 2.5,
            "makes no sense": 2.0,
            "why won't this work": 2.0,
            "driving me crazy": 3.0,
            "so annoying": 2.0,
            "i'm stuck": 1.5,
            "need help": 1.0,
            "don't get it": 1.5,
            "feel overwhelmed": 5.0,
            "feeling overwhelmed": 5.0,
            "not smart enough": 6.0,
            "feel stupid": 5.0,
            "feeling stupid": 5.0,
            "makes me feel": 3.0,
            "feel like": 2.5,
            "self doubt": 4.0,
            "not working": 2.0,
            "ain't working": 2.5,
            "totally broken": 4.0,
            "runs forever": 3.0,
            "loop runs": 2.0,
            "returns none": 2.5,
            "undefined but": 3.0,
            "never executes": 2.5,
            "out of range": 2.5,
            "doesn't make sense": 3.0,
            "missing colon": 2.0,
            "missing quote": 2.0,
            "syntax error": 3.0,
            "exception handling": 2.0,
            "class definition": 2.0,
            "function definition": 2.0
        }
    
    def _get_default_keyword_weights(self) -> Dict[str, Dict[str, Any]]:
        return {
            "high_impact": {
                "weight": 3.0,
                "keywords": ["fuck", "shit", "damn", "hell", "impossible", "hate", "stupid", "giving up", "quit", "worst", "terrible"]
            },
            "medium_impact": {
                "weight": 2.0,
                "keywords": ["stuck", "frustrated", "angry", "annoying", "difficult", "hard", "broken"]
            },
            "mild_impact": {
                "weight": 1.0,
                "keywords": ["confused", "don't understand", "lost", "help", "struggling", "unclear"]
            },
            "error_phrases": {
                "weight": 1.5,
                "keywords": ["not working", "doesn't work", "won't work", "isn't working", "doesn't working"]
            },
            "error_words": {
                "weight": 1.5,
                "keywords": ["error", "exception", "crash", "bug", "broken", "failed", "traceback"]
            },
            "syntax_indicators": {
                "weight": 2.5,
                "keywords": ["syntax", "colon", "parentheses", "braces", "indentation", "=", "==", "missing", "quote", "definition", "statement"]
            },
            "logic_errors": {
                "weight": 2.5,
                "keywords": ["returns none", "never executes", "runs forever", "infinite loop", "out of range", "undefined but"]
            },
            "emotional_distress": {
                "weight": 3.0,
                "keywords": ["overwhelmed", "not smart enough", "feel stupid", "feel like", "makes me feel", "self doubt", "not good at"]
            },
            "gratitude": {
                "weight": -2.0,
                "keywords": ["thank", "thanks", "appreciate", "grateful", "best tutor", "you're great"]
            },
            "positive_emotions": {
                "weight": -1.0,
                "keywords": ["excited", "proud", "love", "enjoy", "fantastic", "great"]
            }
        }
    
    def _get_default_concept_keywords(self) -> Dict[str, List[str]]:
        return {
            "loops": ["for", "while", "loop", "iterate", "iteration", "range", "infinite", "forever"],
            "functions": ["function", "def", "return", "parameter", "argument", "call", "decorator", "lambda", "print"],
            "variables": ["variable", "var", "assignment", "value", "store", "undefined", "declare"],
            "conditionals": ["if", "else", "elif", "condition", "boolean", "true", "false", "comparison"],
            "recursion": ["recursion", "recursive", "base case", "call itself"],
            "errors": ["error", "exception", "traceback", "bug", "debug", "syntax", "broken", "help", "issue", "problem", "fix"],
            "classes": ["class", "object", "method", "self", "init", "instance", "oop"],
            "lists": ["list", "array", "index", "append", "element", "comprehension"],
            "dictionaries": ["dict", "dictionary", "key", "value", "hash"],
            "strings": ["string", "str", "text", "character", "concatenate", "quote", "print", "hello"]
        }
    
    def _get_default_detection_settings(self) -> Dict[str, Any]:
        return {
            "frustration_caps": {
                "max_score": 10.0,
                "min_score": 0.0
            },
            "punctuation_weights": {
                "exclamation_weight": 0.5,
                "exclamation_max": 2.0,
                "question_weight": 0.3,
                "question_max": 1.0,
                "caps_weight": 3.0
            },
            "empathy_thresholds": {
                "high": 7.0,
                "medium": 4.0,
                "minimal": 2.0
            },
            "intervention_levels": {
                "level_1_max": 3.0,
                "level_2_max": 7.0,
                "level_3_min": 7.0
            },
            "response_settings": {
                "short_response_max_tokens": 80,
                "long_response_max_tokens": 150,
                "complex_code_threshold": 100
            }
        }

# Global instance
config_loader = FrustrationConfigLoader()