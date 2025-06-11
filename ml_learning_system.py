# ml_learning_system.py - Complete Self-Learning ML Components
import numpy as np
import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
from dataclasses import dataclass

@dataclass
class LearningFeedback:
    interaction_id: str
    student_id: str
    prediction_error: float
    actual_frustration: float
    predicted_frustration: float
    feature_importance: Dict[str, float]
    response_helpful: bool
    empathy_appropriate: bool

class MLFeatureExtractor:
    """Enhanced feature extraction for ML learning"""
    
    def __init__(self):
        self.emotion_words = self._load_emotion_lexicon()
        
    def extract_features(self, message: str, student_history: List[Dict] = None) -> Dict[str, float]:
        """Extract comprehensive features for ML learning"""
        features = {}
        message_lower = message.lower()
        
        # Basic linguistic features
        features.update(self._extract_basic_features(message, message_lower))
        
        # Emotional intensity features
        features.update(self._extract_emotional_features(message_lower))
        
        # Contextual features (based on history)
        if student_history:
            features.update(self._extract_contextual_features(message_lower, student_history))
        
        # Programming-specific features
        features.update(self._extract_programming_features(message_lower))
        
        return features
    
    def _extract_basic_features(self, message: str, message_lower: str) -> Dict[str, float]:
        """Basic linguistic features"""
        words = message.split()
        return {
            'message_length': len(message),
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'caps_ratio': sum(1 for c in message if c.isupper()) / max(len(message), 1),
            'exclamation_count': message.count('!'),
            'question_count': message.count('?'),
            'repetition_score': self._calculate_repetition(message_lower),
            'urgency_words': sum(1 for word in ['urgent', 'asap', 'quick', 'fast'] if word in message_lower)
        }
    
    def _extract_emotional_features(self, message_lower: str) -> Dict[str, float]:
        """Extract emotional intensity features"""
        features = {}
        
        # Sentiment intensity
        positive_words = sum(1 for word in self.emotion_words['positive'] if word in message_lower)
        negative_words = sum(1 for word in self.emotion_words['negative'] if word in message_lower)
        
        features.update({
            'positive_emotion_count': positive_words,
            'negative_emotion_count': negative_words,
            'emotion_ratio': (positive_words - negative_words) / max(positive_words + negative_words, 1),
            'frustration_markers': self._count_frustration_markers(message_lower),
            'desperation_level': self._calculate_desperation(message_lower),
            'confidence_level': self._calculate_confidence_indicators(message_lower)
        })
        
        return features
    
    def _extract_contextual_features(self, message_lower: str, history: List[Dict]) -> Dict[str, float]:
        """Extract features based on student interaction history"""
        recent_history = history[-5:] if len(history) > 5 else history
        
        return {
            'avg_historical_frustration': np.mean([h.get('frustration_score', 0) for h in recent_history]),
            'frustration_trend': self._calculate_trend([h.get('frustration_score', 0) for h in recent_history]),
            'session_message_count': len([h for h in recent_history if self._is_same_session(h)]),
            'topic_consistency': self._calculate_topic_consistency(message_lower, recent_history),
            'help_seeking_frequency': sum(1 for h in recent_history if 'help' in h.get('message', '').lower())
        }
    
    def _extract_programming_features(self, message_lower: str) -> Dict[str, float]:
        """Programming-specific feature extraction"""
        return {
            'code_present': 1 if any(marker in message_lower for marker in ['def ', 'print(', 'if ', 'for ', '=']) else 0,
            'error_keywords': sum(1 for word in ['error', 'exception', 'traceback', 'bug'] if word in message_lower),
            'syntax_indicators': sum(1 for marker in ['syntax', 'colon', 'indent', 'quote'] if marker in message_lower),
            'concept_difficulty': self._assess_concept_difficulty(message_lower),
            'code_complexity': self._estimate_code_complexity(message_lower)
        }
    
    def _load_emotion_lexicon(self) -> Dict[str, List[str]]:
        """Load emotion word lexicon"""
        return {
            'positive': ['good', 'great', 'excellent', 'amazing', 'love', 'enjoy', 'happy', 'excited'],
            'negative': ['bad', 'terrible', 'awful', 'hate', 'frustrated', 'angry', 'confused', 'lost'],
            'desperation': ['impossible', 'can\'t', 'give up', 'quit', 'hopeless', 'stupid', 'never']
        }
    
    def _calculate_repetition(self, message: str) -> float:
        """Calculate word repetition score"""
        words = message.split()
        if len(words) < 2:
            return 0
        word_counts = Counter(words)
        max_count = max(word_counts.values())
        return (max_count - 1) / max(len(words) - 1, 1)
    
    def _count_frustration_markers(self, message: str) -> float:
        """Count frustration-specific markers"""
        markers = ['!!!', '???', 'wtf', 'omg', 'argh', 'ugh', 'dammit']
        return sum(message.count(marker) for marker in markers)
    
    def _calculate_desperation(self, message: str) -> float:
        """Calculate desperation level"""
        desperation_phrases = [
            'this is impossible', 'i can\'t do this', 'i give up', 
            'i\'m too stupid', 'this makes no sense', 'i hate this'
        ]
        return sum(1 for phrase in desperation_phrases if phrase in message)
    
    def _calculate_confidence_indicators(self, message: str) -> float:
        """Calculate confidence level indicators"""
        confidence_words = ['think', 'maybe', 'probably', 'sure', 'certain', 'know']
        uncertainty_words = ['confused', 'unsure', 'don\'t know', 'not sure']
        
        confidence = sum(1 for word in confidence_words if word in message)
        uncertainty = sum(1 for word in uncertainty_words if word in message)
        
        return (confidence - uncertainty) / max(confidence + uncertainty, 1)
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate trend in frustration scores"""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(scores)
        x_sum = sum(range(n))
        y_sum = sum(scores)
        xy_sum = sum(i * score for i, score in enumerate(scores))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def _is_same_session(self, interaction: Dict) -> bool:
        """Check if interaction is from the same session (within last hour)"""
        try:
            if 'created_at' not in interaction:
                return False
            
            interaction_time = datetime.fromisoformat(str(interaction['created_at']).replace('Z', '+00:00'))
            current_time = datetime.now()
            time_diff = (current_time - interaction_time).total_seconds()
            
            return time_diff < 3600  # 1 hour
            
        except Exception:
            return False
    
    def _calculate_topic_consistency(self, message: str, history: List[Dict]) -> float:
        """Calculate how consistent the current message is with recent topics"""
        if not history:
            return 0.5
        
        # Simple keyword overlap approach
        current_words = set(message.lower().split())
        
        overlap_scores = []
        for interaction in history[-3:]:  # Check last 3 interactions
            hist_words = set(interaction.get('message', '').lower().split())
            if hist_words:
                overlap = len(current_words & hist_words) / len(current_words | hist_words)
                overlap_scores.append(overlap)
        
        return sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.5
    
    def _assess_concept_difficulty(self, message: str) -> float:
        """Assess the difficulty level of programming concepts mentioned"""
        difficulty_keywords = {
            'easy': ['print', 'variable', 'string', 'number'],
            'medium': ['loop', 'function', 'list', 'dictionary'],
            'hard': ['class', 'inheritance', 'recursion', 'algorithm'],
            'very_hard': ['metaclass', 'decorator', 'generator', 'async']
        }
        
        difficulty_scores = {'easy': 1, 'medium': 2, 'hard': 3, 'very_hard': 4}
        
        max_difficulty = 0
        for level, keywords in difficulty_keywords.items():
            if any(keyword in message.lower() for keyword in keywords):
                max_difficulty = max(max_difficulty, difficulty_scores[level])
        
        return float(max_difficulty)
    
    def _estimate_code_complexity(self, message: str) -> float:
        """Estimate complexity of code mentioned in message"""
        complexity_indicators = [
            ('def ', 1.0),
            ('class ', 1.5),
            ('for ', 1.0),
            ('while ', 1.0),
            ('if ', 0.5),
            ('try:', 1.5),
            ('lambda ', 2.0),
            ('import ', 0.5)
        ]
        
        complexity = 0.0
        for indicator, weight in complexity_indicators:
            complexity += message.lower().count(indicator) * weight
        
        return min(complexity, 5.0)  # Cap at 5.0

class AdaptiveWeightManager:
    """Manages adaptive weights for personalized frustration estimation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.global_weights = self._load_global_weights()
        self.student_weights = self._load_student_weights()
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.001
        
    def get_weights(self, student_id: str) -> Dict[str, float]:
        """Get personalized weights for a student"""
        if student_id in self.student_weights:
            return self.student_weights[student_id]['weights']
        else:
            # Initialize with global weights
            self.student_weights[student_id] = {
                'weights': self.global_weights.copy(),
                'momentum': defaultdict(float),
                'update_count': 0
            }
            return self.global_weights.copy()
    
    def update_weights(self, student_id: str, features: Dict[str, float], 
                      prediction_error: float, feedback: LearningFeedback):
        """Update weights based on prediction error and feedback"""
        if student_id not in self.student_weights:
            self.get_weights(student_id)  # Initialize
        
        student_data = self.student_weights[student_id]
        weights = student_data['weights']
        momentum = student_data['momentum']
        
        # Calculate gradients using prediction error
        for feature, value in features.items():
            if feature in weights:
                # Gradient calculation
                gradient = prediction_error * value
                
                # Apply momentum
                momentum[feature] = self.momentum * momentum[feature] + self.learning_rate * gradient
                
                # Update weight with momentum and decay
                weights[feature] += momentum[feature] - self.weight_decay * weights[feature]
        
        # Additional updates based on qualitative feedback
        if feedback:
            self._apply_feedback_adjustments(student_id, features, feedback)
        
        student_data['update_count'] += 1
        self._save_student_weights()
        
        # Update global weights if this student has enough data
        if student_data['update_count'] % 10 == 0:
            self._update_global_weights(student_id)
    
    def _apply_feedback_adjustments(self, student_id: str, features: Dict[str, float], 
                                  feedback: LearningFeedback):
        """Apply adjustments based on qualitative feedback"""
        weights = self.student_weights[student_id]['weights']
        adjustment_strength = 0.05
        
        # If response wasn't helpful, adjust empathy-related features
        if not feedback.response_helpful:
            for feature in ['negative_emotion_count', 'desperation_level', 'frustration_markers']:
                if feature in weights:
                    weights[feature] += adjustment_strength
        
        # If empathy was inappropriate, adjust emotional features
        if not feedback.empathy_appropriate:
            for feature in ['emotion_ratio', 'confidence_level']:
                if feature in weights:
                    weights[feature] *= 0.95  # Reduce influence
    
    def _load_global_weights(self) -> Dict[str, float]:
        """Load global default weights"""
        default_weights = {
            'message_length': -0.001,
            'word_count': 0.01,
            'caps_ratio': 2.0,
            'exclamation_count': 0.5,
            'question_count': 0.3,
            'negative_emotion_count': 1.5,
            'frustration_markers': 2.0,
            'desperation_level': 3.0,
            'code_present': 0.5,
            'error_keywords': 1.2,
            'syntax_indicators': 1.0,
            'repetition_score': 1.5,
            'avg_historical_frustration': 0.8,
            'frustration_trend': 1.2
        }
        
        # Try to load from database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT weights FROM global_weights ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                saved_weights = json.loads(row[0])
                default_weights.update(saved_weights)
            conn.close()
        except:
            pass  # Use defaults
        
        return default_weights
    
    def _load_student_weights(self) -> Dict[str, Dict]:
        """Load student-specific weights from database"""
        student_weights = {}
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT student_id, personalized_weights 
                FROM student_learning_profiles
            """)
            for row in cursor.fetchall():
                student_id, weights_json = row
                student_weights[student_id] = json.loads(weights_json)
            conn.close()
        except:
            pass  # No existing data
        
        return student_weights
    
    def _save_student_weights(self):
        """Save student weights to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_learning_profiles (
                    student_id TEXT PRIMARY KEY,
                    personalized_weights TEXT,
                    learning_style TEXT,
                    adaptation_rate REAL,
                    last_updated TIMESTAMP
                )
            """)
            
            # Update student weights
            for student_id, data in self.student_weights.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO student_learning_profiles 
                    (student_id, personalized_weights, last_updated)
                    VALUES (?, ?, ?)
                """, (student_id, json.dumps(data), datetime.now()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving student weights: {e}")
    
    def _update_global_weights(self, student_id: str):
        """Update global weights based on student learning"""
        # This would implement global weight updates
        pass

class PatternDiscoveryEngine:
    """Discovers new emotional patterns and keywords from student interactions"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.min_pattern_frequency = 2
        self.confidence_threshold = 0.5
        
    def discover_patterns
        
        # DEBUG: Enhanced logging for pattern discovery
        def _debug_log_pattern_discovery(self, message: str, data: Dict = None):
            """Enhanced debug logging for pattern discovery"""
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"PATTERN_DISCOVERY: {message}")
            if data:
                logger.debug(f"PATTERN_DATA: {json.dumps(data, indent=2)}")
            print(f"🔍 PATTERN_DEBUG: {message}")
            if data:
                print(f"   Data: {data}")
        (self, lookback_days: int = 30) -> List[Dict]:
        """Discover new patterns from recent interactions"""
        # Get recent high-error predictions
        high_error_interactions = self._get_high_error_interactions(lookback_days)
        
        if len(high_error_interactions) < 10:
            return []
        
        # Extract text and scores
        messages = [interaction['message'] for interaction in high_error_interactions]
        actual_scores = [interaction['actual_frustration'] for interaction in high_error_interactions]
        
        # Find new patterns
        new_patterns = []
        new_patterns.extend(self._discover_phrase_patterns(messages, actual_scores))
        new_patterns.extend(self._discover_word_patterns(messages, actual_scores))
        
        # Validate patterns
        validated_patterns = self._validate_patterns(new_patterns)
        
        # Save discovered patterns
        self._save_discovered_patterns(validated_patterns)
        
        return validated_patterns
    
    def _get_high_error_interactions(self, lookback_days: int) -> List[Dict]:
        """Get interactions with high prediction errors"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            cursor.execute("""
                SELECT lf.interaction_id, i.message, lf.actual_frustration, 
                       lf.predicted_frustration, lf.prediction_error
                FROM learning_feedback lf
                JOIN interactions i ON lf.interaction_id = i.id
                WHERE lf.created_at > ? AND ABS(lf.prediction_error) > 2.0
                ORDER BY ABS(lf.prediction_error) DESC
            """, (cutoff_date,))
            
            return [
                {
                    'interaction_id': row[0],
                    'message': row[1],
                    'actual_frustration': row[2],
                    'predicted_frustration': row[3],
                    'prediction_error': row[4]
                }
                for row in cursor.fetchall()
            ]
        except:
            return []
    
    def _discover_phrase_patterns(self, messages: List[str], scores: List[float]) -> List[Dict]:
        """Discover new emotional phrase patterns"""
        patterns = []
        
        # Extract 2-4 word phrases
        for n in range(2, 5):
            ngrams = self._extract_ngrams(messages, n, scores)
            
            for phrase, phrase_data in ngrams.items():
                if phrase_data['count'] >= self.min_pattern_frequency:
                    avg_frustration = np.mean(phrase_data['scores'])
                    if avg_frustration > 6.0:  # High frustration phrases
                        patterns.append({
                            'type': 'emotional_pattern',
                            'pattern': phrase,
                            'weight': min(avg_frustration - 3.0, 5.0),
                            'frequency': phrase_data['count'],
                            'confidence': self._calculate_pattern_confidence(phrase_data)
                        })
        
        return patterns
    
    def _discover_word_patterns(self, messages: List[str], scores: List[float]) -> List[Dict]:
        """Discover individual words that correlate with frustration"""
        patterns = []
        word_stats = defaultdict(lambda: {'scores': [], 'count': 0})
        
        for message, score in zip(messages, scores):
            words = re.findall(r'\b\w+\b', message.lower())
            for word in words:
                if len(word) > 2:  # Skip very short words
                    word_stats[word]['scores'].append(score)
                    word_stats[word]['count'] += 1
        
        for word, stats in word_stats.items():
            if stats['count'] >= self.min_pattern_frequency:
                avg_frustration = np.mean(stats['scores'])
                if avg_frustration > 5.0:
                    patterns.append({
                        'type': 'keyword',
                        'pattern': word,
                        'weight': min((avg_frustration - 2.0) * 0.5, 3.0),
                        'frequency': stats['count'],
                        'confidence': self._calculate_word_confidence(stats)
                    })
        
        return patterns
    
    def _extract_ngrams(self, messages: List[str], n: int, scores: List[float]) -> Dict[str, Dict]:
        """Extract n-grams from messages with their frustration scores"""
        ngrams = defaultdict(lambda: {'count': 0, 'scores': []})
        
        for message, score in zip(messages, scores):
            # Clean and tokenize message
            words = re.findall(r'\b\w+\b', message.lower())
            
            # Extract n-grams
            for j in range(len(words) - n + 1):
                ngram = ' '.join(words[j:j+n])
                ngrams[ngram]['count'] += 1
                ngrams[ngram]['scores'].append(score)
                
        return dict(ngrams)
    
    def _calculate_pattern_confidence(self, pattern_data: Dict) -> float:
        """Calculate confidence score for a discovered pattern"""
        count = pattern_data.get('count', 0)
        scores = pattern_data.get('scores', [])
        
        if not scores:
            return 0.0
        
        # Confidence based on frequency and score consistency
        frequency_factor = min(count / 10.0, 1.0)  # Cap at 1.0
        consistency_factor = 1.0 - (np.std(scores) / max(np.mean(scores), 1.0))
        
        return (frequency_factor + consistency_factor) / 2.0
    
    def _calculate_word_confidence(self, stats: Dict) -> float:
        """Calculate confidence for a word pattern"""
        count = stats.get('count', 0)
        scores = stats.get('scores', [])
        
        if not scores:
            return 0.0
        
        avg_score = np.mean(scores)
        frequency_factor = min(count / 5.0, 1.0)
        score_factor = min(avg_score / 10.0, 1.0)
        
        return (frequency_factor + score_factor) / 2.0
    
    def _validate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Validate discovered patterns against historical data"""
        validated = []
        
        for pattern in patterns:
            if pattern.get('confidence', 0) >= self.confidence_threshold:
                validated.append(pattern)
        
        return validated
    
    def _save_discovered_patterns(self, patterns: List[Dict]):
        """Save discovered patterns to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS discovered_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_text TEXT,
                    weight REAL,
                    confidence_score REAL,
                    usage_count INTEGER,
                    effectiveness_score REAL,
                    status TEXT DEFAULT 'candidate',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert patterns
            for pattern in patterns:
                pattern_id = f"pattern_{datetime.now().timestamp()}_{hash(pattern['pattern'])}"
                cursor.execute("""
                    INSERT INTO discovered_patterns 
                    (id, pattern_type, pattern_text, weight, confidence_score, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    pattern['type'],
                    pattern['pattern'],
                    pattern.get('weight', 1.0),
                    pattern.get('confidence', 0.5),
                    pattern.get('frequency', 1)
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving patterns: {e}")

class HybridFrustrationAnalyzer:
    """Hybrid analyzer combining rule-based and ML approaches"""
    
    def __init__(self, db_path: str, rule_analyzer):
        self.rule_analyzer = rule_analyzer
        self.feature_extractor = MLFeatureExtractor()
        self.weight_manager = AdaptiveWeightManager(db_path)
        self.pattern_discovery = PatternDiscoveryEngine(db_path)
        self.db_path = db_path
        
    def analyze_with_learning(self, message: str, student_id: str) -> Dict:
        """Analyze message with hybrid approach and learning"""
        
        # Get rule-based analysis
        rule_result = self.rule_analyzer.analyze_message(message)
        
        # Get student history
        history = self._get_student_history(student_id)
        
        # Extract ML features
        ml_features = self.feature_extractor.extract_features(message, history)
        
        # Get ML prediction
        ml_score = self._predict_ml_score(ml_features, student_id)
        
        # Calculate confidence and combine scores
        confidence = self._calculate_confidence(rule_result, ml_score, ml_features)
        final_score = self._combine_scores(rule_result["frustration_score"], ml_score, confidence)
        
        # Prepare result
        result = {
            "frustration_score": final_score,
            "empathy_level": rule_result["empathy_level"],
            "concepts": rule_result["concepts"],
            "confidence": confidence,
            "ml_features": ml_features,
            "rule_contribution": rule_result["frustration_score"],
            "ml_contribution": ml_score,
            "learning_opportunity": confidence < 0.7,
            "debug_info": rule_result.get("debug_info", [])
        }
        
        # Flag for pattern discovery if low confidence
        if confidence < 0.7:
            self._flag_for_pattern_discovery(message, final_score, student_id)
        
        return result
    
    def learn_from_feedback(self, interaction_id: str, feedback: LearningFeedback):
        """Process feedback and update ML components"""
        
        # Get interaction data
        interaction = self._get_interaction(interaction_id)
        if not interaction:
            return
        
        # Extract features
        ml_features_json = interaction.get('ml_features', '{}')
        try:
            ml_features = json.loads(ml_features_json) if ml_features_json else {}
        except:
            ml_features = {}
        
        # Update weights
        self.weight_manager.update_weights(
            feedback.student_id, 
            ml_features, 
            feedback.prediction_error, 
            feedback
        )
        
        # Store learning feedback
        self._store_learning_feedback(feedback)
        
        # Trigger pattern discovery periodically
        if hash(interaction_id) % 50 == 0:  # Every ~50 interactions
            self.pattern_discovery.discover_patterns()
    
    def _predict_ml_score(self, features: Dict[str, float], student_id: str) -> float:
        """Predict frustration using ML features and personalized weights"""
        weights = self.weight_manager.get_weights(student_id)
        
        score = 0.0
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value
        
        # Apply sigmoid normalization
        normalized = 10 / (1 + np.exp(-score / 3))
        return round(normalized, 2)
    
    def _calculate_confidence(self, rule_result: Dict, ml_score: float, features: Dict) -> float:
        """Calculate prediction confidence"""
        rule_score = rule_result["frustration_score"]
        
        # Agreement between rule and ML
        agreement = 1 - abs(rule_score - ml_score) / 10.0
        
        # Feature completeness
        feature_completeness = len(features) / 15.0  # Assuming ~15 key features
        
        # Historical accuracy (placeholder - could be computed from past feedback)
        historical_accuracy = 0.8
        
        confidence = (0.4 * agreement + 0.3 * feature_completeness + 0.3 * historical_accuracy)
        return max(0.1, min(1.0, confidence))
    
    def _combine_scores(self, rule_score: float, ml_score: float, confidence: float) -> float:
        """Intelligently combine rule-based and ML scores"""
        # Higher confidence in ML = more weight to ML
        ml_weight = 0.2 + 0.3 * confidence  # 0.2 to 0.5
        rule_weight = 1 - ml_weight
        
        combined = rule_weight * rule_score + ml_weight * ml_score
        return round(combined, 2)
    
    def _get_student_history(self, student_id: str) -> List[Dict]:
        """Get recent interaction history for a student"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent interactions for this student
            cursor.execute("""
                SELECT message, frustration_score, created_at, response
                FROM interactions 
                WHERE student_id = ? 
                ORDER BY created_at DESC 
                LIMIT 10
            """, (student_id,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "message": row[0],
                    "frustration_score": row[1],
                    "created_at": row[2],
                    "response": row[3]
                })
            
            conn.close()
            return list(reversed(history))  # Return in chronological order
            
        except Exception as e:
            print(f"⚠️  Could not retrieve student history: {e}")
            return []
    
    def _get_interaction(self, interaction_id: str) -> Dict:
        """Get interaction data by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, student_id, message, response, frustration_score, 
                       ml_features, confidence_score, created_at
                FROM interactions 
                WHERE id = ?
            """, (interaction_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    "id": row[0],
                    "student_id": row[1], 
                    "message": row[2],
                    "response": row[3],
                    "frustration_score": row[4],
                    "ml_features": row[5],
                    "confidence_score": row[6],
                    "created_at": row[7]
                }
            return {}
            
        except Exception as e:
            print(f"Error getting interaction: {e}")
            return {}
    
    def _store_learning_feedback(self, feedback: LearningFeedback):
        """Store learning feedback in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO learning_feedback 
                (id, interaction_id, student_id, prediction_error, actual_frustration,
                 predicted_frustration, response_helpful, empathy_appropriate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f"feedback_{datetime.now().timestamp()}",
                feedback.interaction_id,
                feedback.student_id,
                feedback.prediction_error,
                feedback.actual_frustration,
                feedback.predicted_frustration,
                1 if feedback.response_helpful else 0,
                1 if feedback.empathy_appropriate else 0,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing learning feedback: {e}")
    
    def _flag_for_pattern_discovery(self, message: str, score: float, student_id: str):
        """Flag low-confidence predictions for pattern discovery"""
        try:
            # Store in database for later pattern analysis
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create flag table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_discovery_flags (
                    id TEXT PRIMARY KEY,
                    message TEXT,
                    frustration_score REAL,
                    student_id TEXT,
                    flagged_at TIMESTAMP,
                    processed INTEGER DEFAULT 0
                )
            """)
            
            cursor.execute("""
                INSERT INTO pattern_discovery_flags 
                (id, message, frustration_score, student_id, flagged_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"flag_{datetime.now().timestamp()}_{hash(message)}",
                message,
                score,
                student_id,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            print(f"🔍 Flagged for pattern discovery: '{message[:30]}...'")
            
        except Exception as e:
            print(f"Error flagging for pattern discovery: {e}")

# Usage Example
def create_learning_system(db_path: str, existing_analyzer):
    """Create the complete learning system"""
    return HybridFrustrationAnalyzer(db_path, existing_analyzer)