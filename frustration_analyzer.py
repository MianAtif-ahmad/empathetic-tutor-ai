# frustration_analyzer.py - Improved with Smart Scoring Logic
from typing import Dict, List, Tuple, Set
from attrib_loader import config_loader

class FrustrationAnalyzer:
    """
    Analyzes student messages for frustration levels and programming concepts.
    Uses smart scoring to prevent over-inflation from multiple pattern matches.
    """
    
    def __init__(self):
        self.config = config_loader
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all configurations from config files"""
        self.emotional_patterns = self.config.get_emotional_patterns()
        self.keyword_weights = self.config.get_keyword_weights()
        self.concept_keywords = self.config.get_concept_keywords()
        self.settings = self.config.get_detection_settings()
    
    def reload_configurations(self):
        """Reload configurations from disk"""
        self.config.reload_config()
        self._load_configurations()
        print("🔄 Frustration analyzer configurations reloaded")
    
    def analyze_message(self, message: str) -> Dict:
        """
        Analyze a student message for frustration and concepts with smart scoring.
        
        Returns:
            Dict containing frustration_score, empathy_level, concepts, and debug_info
        """
        message_lower = message.lower()
        debug_info = []
        
        # Step 1: Collect all matches with their sources
        all_matches = self._collect_all_matches(message_lower, debug_info)
        
        # Step 2: Apply smart scoring logic to prevent over-inflation
        frustration_score = self._calculate_smart_score(all_matches, debug_info)
        
        # Step 3: Add punctuation-based frustration
        frustration_score += self._analyze_punctuation(message, debug_info)
        
        # Step 4: Apply message type adjustments
        frustration_score = self._apply_message_type_adjustments(message_lower, frustration_score, debug_info)
        
        # Step 5: Normalize score
        frustration_score = self._normalize_score(frustration_score)
        
        # Step 6: Detect programming concepts
        concepts = self._detect_concepts(message_lower)
        
        # Step 7: Determine empathy level
        empathy_level = self._determine_empathy_level(frustration_score)
        
        return {
            "frustration_score": frustration_score,
            "empathy_level": empathy_level,
            "concepts": concepts,
            "debug_info": debug_info,
            "message_stats": {
                "length": len(message),
                "words": len(message.split()),
                "caps_ratio": sum(1 for c in message if c.isupper()) / max(len(message), 1)
            }
        }
    
    def _collect_all_matches(self, message_lower: str, debug_info: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Collect all pattern matches organized by category"""
        matches = {
            "emotional": [],
            "high_impact": [],
            "medium_impact": [],
            "mild_impact": [],
            "error_related": [],
            "syntax": [],
            "logic_errors": [],
            "emotional_distress": [],
            "positive": []
        }
        
        # Emotional patterns (highest priority)
        for pattern, weight in self.emotional_patterns.items():
            if pattern in message_lower:
                matches["emotional"].append((pattern, weight))
                debug_info.append(f"EMOTIONAL: '{pattern}' (+{weight})")
        
        # Keyword categories
        for category, config in self.keyword_weights.items():
            weight = config["weight"]
            keywords = config["keywords"]
            
            for keyword in keywords:
                if keyword in message_lower:
                    if category == "high_impact":
                        matches["high_impact"].append((keyword, weight))
                    elif category == "medium_impact":
                        matches["medium_impact"].append((keyword, weight))
                    elif category == "mild_impact":
                        matches["mild_impact"].append((keyword, weight))
                    elif category in ["error_phrases", "error_words"]:
                        matches["error_related"].append((keyword, weight))
                    elif category == "syntax_indicators":
                        matches["syntax"].append((keyword, weight))
                    elif category == "logic_errors":
                        matches["logic_errors"].append((keyword, weight))
                    elif category == "emotional_distress":
                        matches["emotional_distress"].append((keyword, weight))
                    elif category in ["gratitude", "positive_emotions"]:
                        matches["positive"].append((keyword, weight))
                    
                    debug_info.append(f"{category.upper()}: '{keyword}' ({weight:+.1f})")
        
        # Special pattern checks
        if ("isnt" in message_lower or "isn't" in message_lower) and "working" in message_lower:
            matches["error_related"].append(("isn't...working pattern", 1.5))
            debug_info.append("PATTERN: 'isn't...working' (+1.5)")
        
        return matches
    
    def _calculate_smart_score(self, matches: Dict[str, List], debug_info: List[str]) -> float:
        """Calculate frustration score using smart logic to prevent over-inflation"""
        score = 0.0
        
        # Message type detection
        message_type = self._detect_message_type(matches)
        debug_info.append(f"MESSAGE_TYPE: {message_type}")
        
        if message_type == "pure_syntax_error":
            # For pure syntax errors, use moderate scoring
            score += self._calculate_syntax_score(matches, debug_info)
            
        elif message_type == "logic_error":
            # For logic errors, use moderate scoring
            score += self._calculate_logic_score(matches, debug_info)
            
        elif message_type == "emotional_distress":
            # For emotional distress, use higher scoring
            score += self._calculate_emotional_score(matches, debug_info)
            
        else:
            # For mixed or general frustration, use standard scoring
            score += self._calculate_standard_score(matches, debug_info)
        
        return score
    
    def _detect_message_type(self, matches: Dict[str, List]) -> str:
        """Detect the primary type of the message"""
        has_emotional = len(matches["emotional"]) > 0 or len(matches["emotional_distress"]) > 0
        has_syntax = len(matches["syntax"]) > 0
        has_logic = len(matches["logic_errors"]) > 0
        has_high_impact = len(matches["high_impact"]) > 0
        
        # Priority order for classification
        if has_emotional and (has_high_impact or len(matches["emotional_distress"]) > 0):
            return "emotional_distress"
        elif has_syntax and not has_high_impact and not has_emotional:
            return "pure_syntax_error"
        elif has_logic and not has_high_impact and not has_emotional:
            return "logic_error"
        else:
            return "mixed_frustration"
    
    def _calculate_syntax_score(self, matches: Dict[str, List], debug_info: List[str]) -> float:
        """Calculate score for pure syntax errors (should be moderate)"""
        score = 0.0
        
        # Base syntax score (reduced)
        if matches["syntax"]:
            score += 1.5  # Reduced from 2.5
            debug_info.append("SYNTAX_BASE: +1.5")
        
        # Add error phrases but limit impact
        error_score = min(len(matches["error_related"]) * 1.0, 2.0)
        if error_score > 0:
            score += error_score
            debug_info.append(f"ERROR_LIMITED: +{error_score}")
        
        # Add mild frustration words with limit
        mild_score = min(len(matches["mild_impact"]) * 0.5, 1.0)
        if mild_score > 0:
            score += mild_score
            debug_info.append(f"MILD_LIMITED: +{mild_score}")
        
        return score
    
    def _calculate_logic_score(self, matches: Dict[str, List], debug_info: List[str]) -> float:
        """Calculate score for logic errors (should be moderate)"""
        score = 0.0
        
        # Base logic error score
        if matches["logic_errors"]:
            score += 2.0
            debug_info.append("LOGIC_BASE: +2.0")
        
        # Add error-related terms with limit
        error_score = min(len(matches["error_related"]) * 0.8, 1.5)
        if error_score > 0:
            score += error_score
            debug_info.append(f"ERROR_LOGIC: +{error_score}")
        
        # Add mild impact with limit
        mild_score = min(len(matches["mild_impact"]) * 0.5, 1.0)
        if mild_score > 0:
            score += mild_score
            debug_info.append(f"MILD_LOGIC: +{mild_score}")
        
        return score
    
    def _calculate_emotional_score(self, matches: Dict[str, List], debug_info: List[str]) -> float:
        """Calculate score for emotional distress (can be higher)"""
        score = 0.0
        
        # Emotional patterns (full weight)
        for pattern, weight in matches["emotional"]:
            score += weight
        
        # Emotional distress (full weight)
        for pattern, weight in matches["emotional_distress"]:
            score += weight
        
        # High impact words (full weight)
        for pattern, weight in matches["high_impact"]:
            score += weight
        
        # Medium impact (reduced to avoid over-inflation)
        medium_score = min(sum(w for _, w in matches["medium_impact"]), 4.0)
        if medium_score > 0:
            score += medium_score
            debug_info.append(f"MEDIUM_CAPPED: +{medium_score}")
        
        return score
    
    def _calculate_standard_score(self, matches: Dict[str, List], debug_info: List[str]) -> float:
        """Calculate score for mixed/general frustration"""
        score = 0.0
        
        # Emotional patterns (full weight)
        for pattern, weight in matches["emotional"]:
            score += weight
        
        # Impact categories with smart limits
        high_score = min(sum(w for _, w in matches["high_impact"]), 6.0)
        medium_score = min(sum(w for _, w in matches["medium_impact"]), 4.0)
        mild_score = min(sum(w for _, w in matches["mild_impact"]), 2.0)
        
        score += high_score + medium_score + mild_score
        
        if high_score > 0:
            debug_info.append(f"HIGH_CAPPED: +{high_score}")
        if medium_score > 0:
            debug_info.append(f"MEDIUM_CAPPED: +{medium_score}")
        if mild_score > 0:
            debug_info.append(f"MILD_CAPPED: +{mild_score}")
        
        # Error-related terms (limited)
        error_score = min(sum(w for _, w in matches["error_related"]), 3.0)
        if error_score > 0:
            score += error_score
            debug_info.append(f"ERROR_CAPPED: +{error_score}")
        
        # Syntax and logic (limited when mixed with other frustration)
        syntax_score = min(sum(w for _, w in matches["syntax"]), 2.0)
        logic_score = min(sum(w for _, w in matches["logic_errors"]), 2.0)
        
        if syntax_score > 0:
            score += syntax_score
            debug_info.append(f"SYNTAX_MIXED: +{syntax_score}")
        if logic_score > 0:
            score += logic_score
            debug_info.append(f"LOGIC_MIXED: +{logic_score}")
        
        return score
    
    def _apply_message_type_adjustments(self, message_lower: str, score: float, debug_info: List[str]) -> float:
        """Apply final adjustments based on message characteristics"""
        
        # Positive emotions (reduce score)
        gratitude_phrases = ["thank", "thanks", "appreciate", "grateful", "best tutor", "you're great"]
        positive_emotions = ["excited", "proud", "love", "enjoy", "fantastic", "great"]
        
        if any(phrase in message_lower for phrase in gratitude_phrases):
            score = max(0, score - 2.0)
            debug_info.append("GRATITUDE: -2.0")
        
        if any(emotion in message_lower for emotion in positive_emotions):
            score = max(0, score - 1.0)
            debug_info.append("POSITIVE: -1.0")
        
        # Length-based adjustments for very short messages
        if len(message_lower.split()) <= 3:
            score *= 0.8
            debug_info.append("SHORT_MSG: *0.8")
        
        return score
    
    def _analyze_punctuation(self, message: str, debug_info: List[str]) -> float:
        """Analyze punctuation for frustration indicators"""
        punct_settings = self.settings["punctuation_weights"]
        score = 0.0
        
        # Count punctuation
        exclamation_count = message.count('!')
        question_count = message.count('?')
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        
        # Apply weights with smart limits
        exclamation_score = min(exclamation_count * punct_settings["exclamation_weight"], 
                               punct_settings["exclamation_max"])
        question_score = min(question_count * punct_settings["question_weight"], 
                            punct_settings["question_max"])
        caps_score = min(caps_ratio * punct_settings["caps_weight"], 2.0)  # Cap at 2.0
        
        score += exclamation_score + question_score + caps_score
        
        if exclamation_score > 0:
            debug_info.append(f"PUNCT_EXCL: +{exclamation_score:.1f}")
        if question_score > 0:
            debug_info.append(f"PUNCT_QUEST: +{question_score:.1f}")
        if caps_score > 0:
            debug_info.append(f"PUNCT_CAPS: +{caps_score:.1f}")
        
        return score
    
    def _normalize_score(self, score: float) -> float:
        """Normalize frustration score to configured range"""
        caps = self.settings["frustration_caps"]
        return max(caps["min_score"], min(score, caps["max_score"]))
    
    def _detect_concepts(self, message_lower: str) -> List[str]:
        """Detect programming concepts mentioned in the message"""
        concepts = []
        
        for concept, keywords in self.concept_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _determine_empathy_level(self, frustration_score: float) -> str:
        """Determine appropriate empathy level based on frustration score"""
        thresholds = self.settings["empathy_thresholds"]
        
        if frustration_score >= thresholds["high"]:
            return "high"
        elif frustration_score >= thresholds["medium"]:
            return "medium"
        elif frustration_score >= thresholds["minimal"]:
            return "standard"
        else:
            return "minimal"
    
    def get_intervention_level(self, frustration_score: float) -> int:
        """Get intervention level for database storage"""
        levels = self.settings["intervention_levels"]
        
        if frustration_score < levels["level_1_max"]:
            return 1
        elif frustration_score < levels["level_2_max"]:
            return 2
        else:
            return 3
    
    def should_use_long_response(self, message: str) -> bool:
        """Determine if message requires a longer response"""
        threshold = self.settings["response_settings"]["complex_code_threshold"]
        return len(message) > threshold
    
    def get_max_tokens(self, message: str) -> int:
        """Get appropriate max tokens for response generation"""
        response_settings = self.settings["response_settings"]
        
        if self.should_use_long_response(message):
            return response_settings["long_response_max_tokens"]
        else:
            return response_settings["short_response_max_tokens"]
    
    def add_emotional_pattern(self, pattern: str, weight: float):
        """Add a new emotional pattern"""
        self.config.update_emotional_pattern(pattern, weight)
        self.emotional_patterns[pattern] = weight
    
    def get_statistics(self) -> Dict:
        """Get analyzer statistics"""
        return {
            "config_stats": self.config.get_stats(),
            "loaded_patterns": len(self.emotional_patterns),
            "loaded_categories": len(self.keyword_weights),
            "loaded_concepts": len(self.concept_keywords)
        }

# Global instance
frustration_analyzer = FrustrationAnalyzer()