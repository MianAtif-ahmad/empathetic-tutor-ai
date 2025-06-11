import numpy as np
from typing import Dict, List, Optional
import json
import os

class SimpleFrustrationEstimator:
    """
    ML-based frustration estimation with adaptive weights
    """
    
    def __init__(self):
        # Default feature weights (will be personalized per student)
        self.default_weights = {
            'sentiment': -2.0,  # Negative sentiment increases frustration
            'sentiment_neg': 3.0,
            'keywords': 1.5,
            'keyword_intensity': 1.0,
            'punctuation': 0.8,
            'exclamations': 0.6,
            'caps_ratio': 2.0,
            'help_seeking': 0.5,
            'error_mentions': 1.2,
            'emotional_intensity': 0.7,
            'repetition': 1.5,
            'message_length': -0.001,  # Longer messages slightly reduce score
        }
        
        # Load student-specific weights if they exist
        self.student_weights = self._load_student_weights()
        
    def estimate(self, features: Dict[str, float], student_id: Optional[str] = None) -> float:
        """
        Estimate frustration score from features
        """
        # Get weights for this student
        if student_id and student_id in self.student_weights:
            weights = self.student_weights[student_id]
        else:
            weights = self.default_weights
            
        # Calculate weighted sum
        score = 0.0
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value
                
        # Normalize to 0-10 scale
        # Using sigmoid function to keep in reasonable range
        normalized = 10 / (1 + np.exp(-score/3))
        
        return round(normalized, 2)
    
    def update_weights(self, student_id: str, feedback: Dict[str, float]):
        """
        Update student-specific weights based on feedback
        """
        if student_id not in self.student_weights:
            self.student_weights[student_id] = self.default_weights.copy()
            
        # Simple gradient update
        learning_rate = 0.1
        for feature, adjustment in feedback.items():
            if feature in self.student_weights[student_id]:
                self.student_weights[student_id][feature] += learning_rate * adjustment
                
        # Save updated weights
        self._save_student_weights()
        
    def _load_student_weights(self) -> Dict:
        """Load personalized weights from file"""
        weights_file = "student_weights.json"
        if os.path.exists(weights_file):
            with open(weights_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_student_weights(self):
        """Save personalized weights to file"""
        with open("student_weights.json", 'w') as f:
            json.dump(self.student_weights, f, indent=2)
