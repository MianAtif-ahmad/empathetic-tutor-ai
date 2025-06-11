# backend/app/services/ml/frustration_estimator.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
import json
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import asyncio

from ...core.config import settings
from ...db.database import get_db
from ...models.student import Student, Interaction
from ..nlp.concept_extractor import ProgrammingConcept
from ...utils.redis_client import redis_client
from ...utils.metrics import track_metric

class FrustrationEstimator:
    """
    Advanced frustration estimation using multiple signals and adaptive learning
    """
    
    def __init__(self):
        self.base_weights = {
            "sentiment": -0.8,
            "keywords": 1.0,
            "punctuation": 0.5,
            "grammar": 0.7,
            "latency": 1.2,
            "repetition": 1.0,
            "help_seeking": -0.3,
            "code_errors": 0.9,
            "concept_difficulty": 0.6,
            "time_since_last": 0.4,
            "session_duration": 0.3,
            "error_frequency": 0.8
        }
        
        # Load or initialize ML model
        self.model = self._load_or_create_model()
        self.scaler = StandardScaler()
        
        # Cache for student-specific weights
        self.student_weights_cache = {}
        
        # Concept difficulty mapping
        self.concept_difficulties = self._load_concept_difficulties()
        
    def _load_or_create_model(self) -> RandomForestRegressor:
        """Load existing model or create new one"""
        try:
            with open(f"{settings.MODEL_DIR}/frustration_model.pkl", "rb") as f:
                model = pickle.load(f)
                logger.info("✅ Loaded existing frustration model")
                return model
        except FileNotFoundError:
            logger.info("🆕 Creating new frustration model")
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
    
    def _load_concept_difficulties(self) -> Dict[str, float]:
        """Load concept difficulty scores"""
        try:
            with open(f"{settings.DATA_DIR}/concept_difficulties.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Default difficulties
            return {
                "variables": 0.1,
                "conditionals": 0.3,
                "loops": 0.4,
                "functions": 0.5,
                "recursion": 0.8,
                "classes": 0.6,
                "inheritance": 0.7,
                "decorators": 0.8,
                "generators": 0.9,
                "async": 0.9,
                "metaclasses": 1.0
            }
    
    async def estimate(
        self,
        features: Dict[str, float],
        student_id: str,
        concepts: List[ProgrammingConcept],
        use_ml: bool = True
    ) -> float:
        """
        Estimate frustration level with adaptive weights and ML enhancement
        """
        start_time = datetime.utcnow()
        
        try:
            # Get student-specific weights
            weights = await self._get_student_weights(student_id)
            
            # Add concept-based features
            concept_features = self._extract_concept_features(concepts)
            features.update(concept_features)
            
            # Add temporal features
            temporal_features = await self._extract_temporal_features(student_id)
            features.update(temporal_features)
            
            # Calculate base frustration score
            base_score = self._calculate_weighted_score(features, weights)
            
            # Apply ML model if available and requested
            if use_ml and self.model is not None:
                ml_score = await self._apply_ml_model(features, student_id)
                # Blend base and ML scores
                final_score = 0.7 * base_score + 0.3 * ml_score
            else:
                final_score = base_score
            
            # Normalize to 0-10 range
            normalized_score = max(0, min(10, final_score))
            
            # Log metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await track_metric("frustration_estimation_time", processing_time)
            await track_metric("frustration_score", normalized_score, {"student_id": student_id})
            
            logger.info(
                f"😤 Frustration estimated for {student_id}: {normalized_score:.2f} "
                f"(base: {base_score:.2f}, processing: {processing_time:.3f}s)"
            )
            
            # Cache result
            await self._cache_estimation(student_id, features, normalized_score)
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"❌ Error estimating frustration: {str(e)}")
            # Fallback to simple calculation
            return self._calculate_weighted_score(features, self.base_weights)
    
    async def _get_student_weights(self, student_id: str) -> Dict[str, float]:
        """Get personalized weights for student"""
        # Check cache first
        if student_id in self.student_weights_cache:
            return self.student_weights_cache[student_id]
        
        # Check Redis
        cached = await redis_client.get(f"weights:{student_id}")
        if cached:
            weights = json.loads(cached)
            self.student_weights_cache[student_id] = weights
            return weights
        
        # Load from database
        from ...db.database import SessionLocal
        db = SessionLocal()
        try:
            student = db.query(Student).filter(Student.id == student_id).first()
            if student and student.personalized_weights:
                weights = student.personalized_weights
            else:
                weights = self.base_weights.copy()
            
            # Cache for future use
            await redis_client.setex(
                f"weights:{student_id}",
                3600,  # 1 hour TTL
                json.dumps(weights)
            )
            self.student_weights_cache[student_id] = weights
            
            return weights
            
        finally:
            db.close()
    
    def _extract_concept_features(self, concepts: List[ProgrammingConcept]) -> Dict[str, float]:
        """Extract features from programming concepts"""
        if not concepts:
            return {
                "concept_difficulty": 0.0,
                "concept_count": 0.0,
                "max_difficulty": 0.0,
                "difficulty_variance": 0.0
            }
        
        difficulties = [
            self.concept_difficulties.get(c.name, 0.5)
            for c in concepts
        ]
        
        return {
            "concept_difficulty": np.mean(difficulties),
            "concept_count": len(concepts) / 10.0,  # Normalize
            "max_difficulty": max(difficulties),
            "difficulty_variance": np.var(difficulties) if len(difficulties) > 1 else 0.0
        }
    
    async def _extract_temporal_features(self, student_id: str) -> Dict[str, float]:
        """Extract time-based features"""
        from ...db.database import SessionLocal
        db = SessionLocal()
        
        try:
            # Get recent interactions
            recent = db.query(Interaction).filter(
                Interaction.student_id == student_id
            ).order_by(Interaction.created_at.desc()).limit(10).all()
            
            if not recent:
                return {
                    "time_since_last": 0.0,
                    "session_duration": 0.0,
                    "interaction_frequency": 0.0,
                    "frustration_momentum": 0.0
                }
            
            now = datetime.utcnow()
            
            # Time since last interaction
            time_since_last = (now - recent[0].created_at).total_seconds() / 3600.0  # Hours
            
            # Session duration (time span of recent interactions)
            if len(recent) > 1:
                session_duration = (recent[0].created_at - recent[-1].created_at).total_seconds() / 3600.0
            else:
                session_duration = 0.0
            
            # Interaction frequency
            interaction_frequency = len(recent) / max(1.0, session_duration)
            
            # Frustration momentum (trend)
            if len(recent) >= 3:
                recent_scores = [i.frustration_score for i in recent[:3]]
                older_scores = [i.frustration_score for i in recent[3:6] if i.frustration_score]
                if older_scores:
                    momentum = np.mean(recent_scores) - np.mean(older_scores)
                else:
                    momentum = 0.0
            else:
                momentum = 0.0
            
            return {
                "time_since_last": min(time_since_last, 24.0) / 24.0,  # Cap at 24 hours
                "session_duration": min(session_duration, 4.0) / 4.0,  # Cap at 4 hours
                "interaction_frequency": min(interaction_frequency, 10.0) / 10.0,
                "frustration_momentum": momentum / 10.0  # Normalize
            }
            
        finally:
            db.close()
    
    def _calculate_weighted_score(
        self,
        features: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted frustration score"""
        score = 0.0
        total_weight = 0.0
        
        for feature, value in features.items():
            if feature in weights:
                score += weights[feature] * value
                total_weight += abs(weights[feature])
        
        # Normalize by total weight
        if total_weight > 0:
            normalized = score / total_weight * 10
        else:
            normalized = 5.0  # Default middle score
        
        return normalized
    
    async def _apply_ml_model(
        self,
        features: Dict[str, float],
        student_id: str
    ) -> float:
        """Apply ML model for refined estimation"""
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Get prediction
            if hasattr(self.model, "predict"):
                # Use async prediction if available
                loop = asyncio.get_event_loop()
                prediction = await loop.run_in_executor(
                    None,
                    self.model.predict,
                    feature_vector.reshape(1, -1)
                )
                return prediction[0]
            else:
                # Model not trained yet
                return 5.0
                
        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            return 5.0
    
    def _prepare_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Prepare feature vector for ML model"""
        # Ensure consistent feature ordering
        feature_names = sorted(self.base_weights.keys())
        vector = []
        
        for name in feature_names:
            if name in features:
                vector.append(features[name])
            else:
                vector.append(0.0)
        
        return np.array(vector)
    
    async def _cache_estimation(
        self,
        student_id: str,
        features: Dict[str, float],
        score: float
    ):
        """Cache estimation for learning and analysis"""
        cache_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "student_id": student_id,
            "features": features,
            "score": score
        }
        
        # Store in Redis with TTL
        await redis_client.lpush(
            "frustration_estimations",
            json.dumps(cache_entry)
        )
        await redis_client.ltrim("frustration_estimations", 0, 9999)  # Keep last 10k
    
    async def update_weights(
        self,
        student_id: str,
        feedback: Dict[str, float]
    ):
        """Update student-specific weights based on feedback"""
        current_weights = await self._get_student_weights(student_id)
        
        # Simple gradient update based on feedback
        learning_rate = settings.ALPHA_STUDENT
        
        for feature, current_weight in current_weights.items():
            if feature in feedback:
                # Update weight based on feedback
                gradient = feedback[feature]
                new_weight = current_weight + learning_rate * gradient
                
                # Clip weights to reasonable range
                current_weights[feature] = max(-2.0, min(2.0, new_weight))
        
        # Save updated weights
        from ...db.database import SessionLocal
        db = SessionLocal()
        try:
            student = db.query(Student).filter(Student.id == student_id).first()
            if student:
                student.personalized_weights = current_weights
                db.commit()
                
                # Update caches
                await redis_client.setex(
                    f"weights:{student_id}",
                    3600,
                    json.dumps(current_weights)
                )
                self.student_weights_cache[student_id] = current_weights
                
                logger.info(f"✅ Updated weights for student {student_id}")
                
        finally:
            db.close()
    
    async def retrain_model(self, training_data: Optional[List[Dict]] = None):
        """Retrain the ML model with new data"""
        logger.info("🔄 Starting model retraining...")
        
        if training_data is None:
            # Load from cache/database
            training_data = await self._load_training_data()
        
        if len(training_data) < 100:
            logger.warning("Insufficient training data for retraining")
            return
        
        # Prepare training set
        X = []
        y = []
        
        for entry in training_data:
            features = entry["features"]
            score = entry["score"]
            
            feature_vector = self._prepare_feature_vector(features)
            X.append(feature_vector)
            y.append(score)
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit scaler and model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Save model
        with open(f"{settings.MODEL_DIR}/frustration_model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        
        logger.info(f"✅ Model retrained with {len(training_data)} samples")
    
    async def _load_training_data(self) -> List[Dict]:
        """Load training data from storage"""
        # Get from Redis cache
        data = []
        cached_entries = await redis_client.lrange("frustration_estimations", 0, -1)
        
        for entry in cached_entries:
            try:
                data.append(json.loads(entry))
            except:
                continue
        
        return data