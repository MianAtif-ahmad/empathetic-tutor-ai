import re
from typing import Dict, List, Optional
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import asyncio

class FeatureExtractor:
    """
    Extracts multiple features from student messages for frustration estimation
    """
    
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            
        self.sia = SentimentIntensityAnalyzer()
        
        # Extended frustration keywords
        self.frustration_keywords = [
            # High frustration
            "stuck", "frustrated", "impossible", "hate", "angry", "furious",
            "giving up", "quit", "stupid", "dumb", "worst", "terrible",
            
            # Medium frustration  
            "confused", "don't understand", "lost", "help", "struggling",
            "difficult", "hard", "complicated", "unclear", "annoying",
            
            # Error-related
            "error", "crash", "bug", "broken", "failed", "exception",
            "not working", "doesn't work", "wrong", "incorrect"
        ]
        
        # Help-seeking patterns
        self.help_patterns = [
            r'\b(help|assist|explain|show|teach|guide)\b',
            r'\b(can you|could you|would you|please)\b',
            r'\b(how do|how to|what is|why is)\b',
            r'\?+$'  # Questions
        ]
        
    async def extract(
        self, 
        message: str, 
        previous_messages: List[str] = None,
        student_id: str = None,
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Extract features from student message
        """
        previous_messages = previous_messages or []
        features = {}
        
        # 1. Sentiment Analysis
        sentiment_scores = self.sia.polarity_scores(message)
        features['sentiment'] = sentiment_scores['compound']
        features['sentiment_neg'] = sentiment_scores['neg']
        features['sentiment_pos'] = sentiment_scores['pos']
        features['sentiment_neu'] = sentiment_scores['neu']
        
        # 2. Keyword Analysis
        message_lower = message.lower()
        keyword_count = 0
        keyword_intensity = 0
        
        for keyword in self.frustration_keywords:
            if keyword in message_lower:
                keyword_count += 1
                # Weight by keyword severity
                if keyword in ["impossible", "hate", "stupid", "giving up"]:
                    keyword_intensity += 2.0
                elif keyword in ["stuck", "frustrated", "angry"]:
                    keyword_intensity += 1.5
                else:
                    keyword_intensity += 1.0
                    
        features['keywords'] = keyword_count
        features['keyword_intensity'] = keyword_intensity
        
        # 3. Punctuation Analysis
        features['punctuation'] = len(re.findall(r'[!?]+', message))
        features['exclamations'] = message.count('!')
        features['questions'] = message.count('?')
        features['caps_ratio'] = sum(1 for c in message if c.isupper()) / max(len(message), 1)
        
        # 4. Message Length Features
        features['message_length'] = len(message)
        features['word_count'] = len(message.split())
        
        # 5. Repetition Detection
        if previous_messages:
            # Check if repeating same message
            features['repetition'] = 1.0 if message in previous_messages else 0.0
            # Check similarity to recent messages
            similar_count = sum(1 for prev in previous_messages[-3:] 
                              if self._similarity(message, prev) > 0.8)
            features['similarity'] = similar_count / 3.0
        else:
            features['repetition'] = 0.0
            features['similarity'] = 0.0
            
        # 6. Help-Seeking Behavior
        help_score = 0
        for pattern in self.help_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                help_score += 1
        features['help_seeking'] = min(help_score / len(self.help_patterns), 1.0)
        
        # 7. Error Mentions
        error_keywords = ['error', 'exception', 'crash', 'bug', 'stack', 'trace']
        features['error_mentions'] = sum(1 for word in error_keywords if word in message_lower)
        
        # 8. Code Indicators
        features['has_code'] = 1.0 if any(indicator in message for indicator in 
                                          ['def ', 'class ', 'import ', '()', '[]', '{}']) else 0.0
        
        # 9. Emotional Intensity
        emotional_words = ['really', 'very', 'so', 'extremely', 'totally', 'completely']
        features['emotional_intensity'] = sum(1 for word in emotional_words if word in message_lower)
        
        return features
    
    def _similarity(self, msg1: str, msg2: str) -> float:
        """Simple similarity check based on common words"""
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union)
