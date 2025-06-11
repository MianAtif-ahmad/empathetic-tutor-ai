# backend/app/services/feedback/learning_engine.py

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
from loguru import logger
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

from ...core.config import settings
from ...db.database import get_db
from ...models.student import Student, Interaction, Feedback
from ..ml.concept_extractor import ProgrammingConcept
from ..intervention.decision_engine import InterventionStrategy
from ...utils.redis_client import redis_client
from ...utils.metrics import track_metric

class LearningEngine:
    """
    Multi-level learning system:
    1. Student-level personalization
    2. Global pattern learning
    3. Empathy-frustration calibration
    """
    
    def __init__(self):
        # Student-level models
        self.student_models = {}
        self.student_model_lock = asyncio.Lock()
        
        # Global model
        self.global_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=settings.ALPHA_GLOBAL,
            random_state=42
        )
        self.global_scaler = StandardScaler()
        self.global_model_trained = False
        
        # Empathy calibration model
        self.empathy_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=settings.ALPHA_EMPATHY,
            random_state=42
        )
        self.empathy_scaler = StandardScaler()
        self.empathy_model_trained = False
        
        # Feature importance tracking
        self.feature_importance = self._initialize_feature_importance()
        
        # Learning queues
        self.learning_queue = asyncio.Queue(maxsize=1000)
        self.batch_size = 32
        
        # Start background learning tasks
        asyncio.create_task(self._learning_worker())
        asyncio.create_task(self._periodic_model_update())
    
    def _initialize_feature_importance(self) -> Dict[str, float]:
        """Initialize feature importance tracking"""
        features = [
            "sentiment", "keywords", "punctuation", "grammar",
            "latency", "repetition", "help_seeking", "code_errors",
            "concept_difficulty", "time_since_last", "session_duration",
            "error_frequency"
        ]
        return {f: 1.0 for f in features}
    
    async def update_from_interaction(
        self,
        interaction_id: str,
        student_id: str,
        features: Dict[str, float],
        frustration_score: float,
        intervention: InterventionStrategy,
        concepts: List[ProgrammingConcept]
    ):
        """Update learning from a new interaction"""
        try:
            # Create learning entry
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "interaction_id": interaction_id,
                "student_id": student_id,
                "features": features,
                "frustration_score": frustration_score,
                "intervention_level": intervention.level,
                "concepts": [c.dict() for c in concepts],
                "feedback": None  # Will be updated when feedback arrives
            }
            
            # Add to learning queue
            await self.learning_queue.put(entry)
            
            # Log to persistent storage
            await self._log_learning_data(entry)
            
            # Update student-specific model immediately
            await self._update_student_model(student_id, features, frustration_score)
            
            logger.debug(f"📚 Queued learning update for interaction {interaction_id}")
            
        except Exception as e:
            logger.error(f"❌ Error updating from interaction: {str(e)}")
    
    async def update_from_feedback(
        self,
        feedback_id: str,
        interaction: Interaction
    ):
        """Update learning from student feedback"""
        try:
            # Get the feedback
            from ...db.database import SessionLocal
            db = SessionLocal()
            feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
            
            if not feedback:
                return
            
            # Calculate effectiveness metrics
            effectiveness = self._calculate_effectiveness(feedback)
            
            # Update the interaction's learning entry
            await self._update_learning_entry(
                interaction_id=interaction.id,
                feedback=feedback,
                effectiveness=effectiveness
            )
            
            # Update empathy calibration
            await self._update_empathy_calibration(
                frustration_score=interaction.frustration_score,
                intervention_level=interaction.intervention_level,
                effectiveness=effectiveness
            )
            
            # Update feature importance based on feedback
            await self._update_feature_importance(
                features=interaction.features,
                effectiveness=effectiveness
            )
            
            logger.info(f"✅ Updated learning from feedback {feedback_id}")
            
        except Exception as e:
            logger.error(f"❌ Error updating from feedback: {str(e)}")
        finally:
            db.close()
    
    def _calculate_effectiveness(self, feedback: Feedback) -> float:
        """Calculate intervention effectiveness from feedback"""
        # Weighted combination of feedback signals
        weights = {
            "helpful": 0.4,
            "frustration_reduced": 0.4,
            "clarity_rating": 0.2
        }
        
        score = 0.0
        score += weights["helpful"] * (1.0 if feedback.helpful else 0.0)
        score += weights["frustration_reduced"] * (1.0 if feedback.frustration_reduced else 0.0)
        score += weights["clarity_rating"] * (feedback.clarity_rating / 5.0 if feedback.clarity_rating else 0.5)
        
        return score
    
    async def _update_student_model(
        self,
        student_id: str,
        features: Dict[str, float],
        frustration_score: float
    ):
        """Update student-specific model"""
        async with self.student_model_lock:
            if student_id not in self.student_models:
                self.student_models[student_id] = {
                    "model": SGDRegressor(
                        learning_rate='adaptive',
                        eta0=settings.ALPHA_STUDENT,
                        random_state=42
                    ),
                    "scaler": StandardScaler(),
                    "trained": False,
                    "samples": []
                }
            
            student_model = self.student_models[student_id]
            
            # Prepare feature vector
            X = self._features_to_vector(features)
            y = frustration_score
            
            # Store sample
            student_model["samples"].append((X, y))
            
            # Train when enough samples
            if len(student_model["samples"]) >= 10:
                X_train = np.array([s[0] for s in student_model["samples"]])
                y_train = np.array([s[1] for s in student_model["samples"]])
                
                # Fit scaler
                X_scaled = student_model["scaler"].fit_transform(X_train)
                
                # Partial fit model
                student_model["model"].partial_fit(X_scaled, y_train)
                student_model["trained"] = True
                
                # Keep only recent samples
                if len(student_model["samples"]) > 100:
                    student_model["samples"] = student_model["samples"][-100:]
                
                logger.debug(f"📊 Updated model for student {student_id}")
    
    async def _update_empathy_calibration(
        self,
        frustration_score: float,
        intervention_level: int,
        effectiveness: float
    ):
        """Update empathy-frustration calibration model"""
        try:
            # Feature: [frustration_score, intervention_level]
            X = np.array([[frustration_score, intervention_level]])
            y = np.array([effectiveness])
            
            if self.empathy_model_trained:
                X_scaled = self.empathy_scaler.transform(X)
                self.empathy_model.partial_fit(X_scaled, y)
            else:
                # Need more samples for initial training
                await self._cache_empathy_sample(frustration_score, intervention_level, effectiveness)
                
                # Check if we have enough samples
                samples = await self._get_cached_empathy_samples()
                if len(samples) >= 50:
                    X_train = np.array([[s[0], s[1]] for s in samples])
                    y_train = np.array([s[2] for s in samples])
                    
                    X_scaled = self.empathy_scaler.fit_transform(X_train)
                    self.empathy_model.fit(X_scaled, y_train)
                    self.empathy_model_trained = True
                    
                    logger.info("✅ Empathy calibration model trained")
            
        except Exception as e:
            logger.error(f"Error updating empathy calibration: {str(e)}")
    
    async def _update_feature_importance(
        self,
        features: Dict[str, float],
        effectiveness: float
    ):
        """Update feature importance based on effectiveness"""
        # Simple approach: increase importance of features that correlate with effectiveness
        for feature, value in features.items():
            if feature in self.feature_importance:
                # Update using exponential moving average
                alpha = 0.1
                correlation = value * effectiveness  # Simplified correlation
                self.feature_importance[feature] = (
                    (1 - alpha) * self.feature_importance[feature] +
                    alpha * correlation
                )
        
        # Normalize importance scores
        total = sum(self.feature_importance.values())
        if total > 0:
            self.feature_importance = {
                k: v / total for k, v in self.feature_importance.items()
            }
    
    async def _learning_worker(self):
        """Background worker for batch learning"""
        while True:
            try:
                # Collect batch
                batch = []
                for _ in range(self.batch_size):
                    try:
                        entry = await asyncio.wait_for(
                            self.learning_queue.get(),
                            timeout=10.0
                        )
                        batch.append(entry)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    await self._process_learning_batch(batch)
                
                await asyncio.sleep(1)  # Small delay between batches
                
            except Exception as e:
                logger.error(f"Learning worker error: {str(e)}")
                await asyncio.sleep(5)
    
    async def _process_learning_batch(self, batch: List[Dict]):
        """Process a batch of learning entries"""
        try:
            # Update global model
            X_batch = []
            y_batch = []
            
            for entry in batch:
                X = self._features_to_vector(entry["features"])
                y = entry["frustration_score"]
                X_batch.append(X)
                y_batch.append(y)
            
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            
            if self.global_model_trained:
                X_scaled = self.global_scaler.transform(X_batch)
                self.global_model.partial_fit(X_scaled, y_batch)
            else:
                # Initial training
                if len(X_batch) >= 100:
                    X_scaled = self.global_scaler.fit_transform(X_batch)
                    self.global_model.fit(X_scaled, y_batch)
                    self.global_model_trained = True
                    logger.info("✅ Global model trained")
            
            # Update metrics
            await track_metric("learning_batch_processed", len(batch))
            
        except Exception as e:
            logger.error(f"Error processing learning batch: {str(e)}")
    
    async def _periodic_model_update(self):
        """Periodically update and save models"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                
                # Save models
                await self._save_models()
                
                # Analyze and report learning progress
                await self._analyze_learning_progress()
                
                # Clean up old data
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Periodic update error: {str(e)}")
    
    async def get_optimal_intervention(
        self,
        frustration_score: float,
        student_id: Optional[str] = None
    ) -> int:
        """Get optimal intervention level based on learned patterns"""
        try:
            if self.empathy_model_trained:
                # Test different intervention levels
                best_level = 0
                best_effectiveness = 0.0
                
                for level in range(4):
                    X = np.array([[frustration_score, level]])
                    X_scaled = self.empathy_scaler.transform(X)
                    effectiveness = self.empathy_model.predict(X_scaled)[0]
                    
                    if effectiveness > best_effectiveness:
                        best_effectiveness = effectiveness
                        best_level = level
                
                return best_level
            else:
                # Fallback to rule-based
                if frustration_score < 3:
                    return 0
                elif frustration_score < 6:
                    return 1
                elif frustration_score < 8:
                    return 2
                else:
                    return 3
                    
        except Exception as e:
            logger.error(f"Error getting optimal intervention: {str(e)}")
            return 2  # Default to moderate intervention
    
    async def get_personalized_weights(self, student_id: str) -> Dict[str, float]:
        """Get personalized feature weights for a student"""
        try:
            if student_id in self.student_models and self.student_models[student_id]["trained"]:
                model = self.student_models[student_id]["model"]
                
                # Get feature weights from linear model
                if hasattr(model, 'coef_'):
                    feature_names = sorted(self.feature_importance.keys())
                    weights = {}
                    
                    for i, name in enumerate(feature_names):
                        if i < len(model.coef_):
                            # Combine global importance with student-specific weight
                            weights[name] = (
                                0.7 * self.feature_importance[name] +
                                0.3 * abs(model.coef_[i])
                            )
                    
                    # Normalize
                    total = sum(weights.values())
                    if total > 0:
                        weights = {k: v / total for k, v in weights.items()}
                    
                    return weights
            
            # Return global weights
            return self.feature_importance
            
        except Exception as e:
            logger.error(f"Error getting personalized weights: {str(e)}")
            return self.feature_importance
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to consistent vector"""
        feature_names = sorted(self.feature_importance.keys())
        vector = []
        
        for name in feature_names:
            vector.append(features.get(name, 0.0))
        
        return np.array(vector)
    
    async def _log_learning_data(self, entry: Dict):
        """Log learning data for analysis"""
        # Store in Redis
        await redis_client.lpush(
            "learning_data",
            json.dumps(entry)
        )
        await redis_client.ltrim("learning_data", 0, 99999)  # Keep last 100k
        
        # Also log to file for persistent storage
        log_file = f"{settings.LOG_DIR}/learning_data.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    async def _update_learning_entry(
        self,
        interaction_id: str,
        feedback: Feedback,
        effectiveness: float
    ):
        """Update a learning entry with feedback data"""
        # Update in Redis
        # This is simplified - in production, use a proper database
        pass
    
    async def _cache_empathy_sample(
        self,
        frustration: float,
        intervention: int,
        effectiveness: float
    ):
        """Cache empathy calibration sample"""
        sample = [frustration, intervention, effectiveness]
        await redis_client.lpush(
            "empathy_samples",
            json.dumps(sample)
        )
        await redis_client.ltrim("empathy_samples", 0, 9999)
    
    async def _get_cached_empathy_samples(self) -> List[List[float]]:
        """Get cached empathy samples"""
        samples = []
        cached = await redis_client.lrange("empathy_samples", 0, -1)
        
        for entry in cached:
            try:
                samples.append(json.loads(entry))
            except:
                continue
        
        return samples
    
    async def _save_models(self):
        """Save all models to disk"""
        import pickle
        
        try:
            # Save global model
            if self.global_model_trained:
                with open(f"{settings.MODEL_DIR}/global_model.pkl", "wb") as f:
                    pickle.dump({
                        "model": self.global_model,
                        "scaler": self.global_scaler,
                        "feature_importance": self.feature_importance
                    }, f)
            
            # Save empathy model
            if self.empathy_model_trained:
                with open(f"{settings.MODEL_DIR}/empathy_model.pkl", "wb") as f:
                    pickle.dump({
                        "model": self.empathy_model,
                        "scaler": self.empathy_scaler
                    }, f)
            
            # Save student models
            for student_id, model_data in self.student_models.items():
                if model_data["trained"]:
                    with open(f"{settings.MODEL_DIR}/student_{student_id}.pkl", "wb") as f:
                        pickle.dump(model_data, f)
            
            logger.info("✅ Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    async def _analyze_learning_progress(self):
        """Analyze and log learning progress"""
        try:
            # Get recent learning data
            recent_data = await redis_client.lrange("learning_data", 0, 999)
            
            if not recent_data:
                return
            
            # Convert to DataFrame for analysis
            entries = [json.loads(d) for d in recent_data]
            df = pd.DataFrame(entries)
            
            # Calculate metrics
            avg_frustration = df["frustration_score"].mean()
            intervention_distribution = df["intervention_level"].value_counts().to_dict()
            
            # Log metrics
            await track_metric("avg_frustration_score", avg_frustration)
            for level, count in intervention_distribution.items():
                await track_metric(f"intervention_level_{level}_count", count)
            
            logger.info(
                f"📊 Learning Progress: Avg Frustration={avg_frustration:.2f}, "
                f"Interventions={intervention_distribution}"
            )
            
        except Exception as e:
            logger.error(f"Error analyzing learning progress: {str(e)}")
    
    async def _cleanup_old_data(self):
        """Clean up old learning data"""
        try:
            # Remove old entries from Redis
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            
            # This is simplified - in production, use proper data retention policies
            
            logger.debug("🧹 Cleaned up old learning data")
            
        except Exception as e:
            logger.error(f"Error cleaning up data: {str(e)}")
    
    async def get_learning_insights(self, student_id: Optional[str] = None) -> Dict:
        """Get learning insights for reporting"""
        insights = {
            "global_patterns": {},
            "student_patterns": {},
            "feature_importance": self.feature_importance,
            "empathy_calibration": {}
        }
        
        try:
            # Global patterns
            if self.global_model_trained:
                insights["global_patterns"] = {
                    "model_samples": len(await redis_client.lrange("learning_data", 0, -1)),
                    "feature_weights": self.feature_importance
                }
            
            # Student patterns
            if student_id and student_id in self.student_models:
                model_data = self.student_models[student_id]
                insights["student_patterns"] = {
                    "samples": len(model_data["samples"]),
                    "trained": model_data["trained"]
                }
            
            # Empathy calibration insights
            if self.empathy_model_trained:
                # Generate effectiveness heatmap
                effectiveness_map = {}
                for frustration in range(0, 11):
                    for intervention in range(4):
                        X = np.array([[frustration, intervention]])
                        X_scaled = self.empathy_scaler.transform(X)
                        effectiveness = self.empathy_model.predict(X_scaled)[0]
                        effectiveness_map[f"{frustration}_{intervention}"] = effectiveness
                
                insights["empathy_calibration"] = effectiveness_map
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {str(e)}")
            return insights