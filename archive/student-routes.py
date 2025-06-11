# backend/app/api/routes/student_routes.py

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from datetime import datetime
import uuid
from loguru import logger

from ...core.schemas import (
    StudentMessageRequest,
    StudentMessageResponse,
    FeedbackRequest,
    StudentProgress
)
from ...db.database import get_db
from ...models.student import Student, Interaction, Feedback
from ...services.nlp.feature_extractor import FeatureExtractor
from ...services.ml.frustration_estimator import FrustrationEstimator
from ...services.intervention.decision_engine import DecisionEngine
from ...services.intervention.response_generator import ResponseGenerator
from ...services.feedback.learning_engine import LearningEngine
from ...services.ml.concept_extractor import ConceptExtractor
from ...utils.auth import get_current_student

router = APIRouter()

# Initialize services
feature_extractor = FeatureExtractor()
frustration_estimator = FrustrationEstimator()
decision_engine = DecisionEngine()
response_generator = ResponseGenerator()
learning_engine = LearningEngine()
concept_extractor = ConceptExtractor()

@router.post("/message", response_model=StudentMessageResponse)
async def process_student_message(
    request: StudentMessageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student)
):
    """Process student message and generate adaptive response"""
    
    interaction_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"📥 Processing message from student {current_student.id}: {request.message[:50]}...")
        
        # Step 1: Extract features from message
        previous_messages = db.query(Interaction).filter(
            Interaction.student_id == current_student.id
        ).order_by(Interaction.created_at.desc()).limit(5).all()
        
        features = await feature_extractor.extract(
            message=request.message,
            previous_messages=[m.message for m in previous_messages],
            student_id=current_student.id,
            context=request.context
        )
        logger.debug(f"🔍 Extracted features: {features}")
        
        # Step 2: Extract programming concepts
        concepts = await concept_extractor.extract(
            message=request.message,
            code_snippet=request.code_snippet
        )
        logger.debug(f"🧩 Identified concepts: {concepts}")
        
        # Step 3: Estimate frustration level
        frustration_score = await frustration_estimator.estimate(
            features=features,
            student_id=current_student.id,
            concepts=concepts
        )
        logger.info(f"😤 Frustration score: {frustration_score:.2f}")
        
        # Step 4: Decide intervention strategy
        intervention = await decision_engine.decide(
            frustration_score=frustration_score,
            student_profile=current_student,
            concepts=concepts,
            history=previous_messages
        )
        logger.info(f"🎯 Intervention decision: {intervention}")
        
        # Step 5: Generate response
        response = await response_generator.generate(
            intervention=intervention,
            message=request.message,
            concepts=concepts,
            student_profile=current_student,
            frustration_level=frustration_score
        )
        logger.debug(f"💬 Generated response: {response.content[:100]}...")
        
        # Step 6: Store interaction
        interaction = Interaction(
            id=interaction_id,
            student_id=current_student.id,
            message=request.message,
            response=response.content,
            frustration_score=frustration_score,
            intervention_level=intervention.level,
            features=features,
            concepts=[c.dict() for c in concepts],
            metadata={
                "tone_level": response.tone_level,
                "hints_given": response.hints,
                "resources": response.resources,
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            }
        )
        db.add(interaction)
        db.commit()
        
        # Step 7: Background learning tasks
        background_tasks.add_task(
            learning_engine.update_from_interaction,
            interaction_id=interaction_id,
            student_id=current_student.id,
            features=features,
            frustration_score=frustration_score,
            intervention=intervention,
            concepts=concepts
        )
        
        # Log success
        logger.info(f"✅ Successfully processed message for student {current_student.id}")
        
        return StudentMessageResponse(
            interaction_id=interaction_id,
            response=response.content,
            frustration_score=frustration_score,
            intervention_level=intervention.level,
            tone_level=response.tone_level,
            hints=response.hints,
            resources=response.resources,
            concepts=[c.name for c in concepts],
            processing_time=(datetime.utcnow() - start_time).total_seconds()
        )
        
    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student)
):
    """Submit feedback for an interaction"""
    
    try:
        # Verify interaction belongs to student
        interaction = db.query(Interaction).filter(
            Interaction.id == request.interaction_id,
            Interaction.student_id == current_student.id
        ).first()
        
        if not interaction:
            raise HTTPException(status_code=404, detail="Interaction not found")
        
        # Store feedback
        feedback = Feedback(
            interaction_id=request.interaction_id,
            student_id=current_student.id,
            helpful=request.helpful,
            frustration_reduced=request.frustration_reduced,
            clarity_rating=request.clarity_rating,
            additional_comments=request.comments
        )
        db.add(feedback)
        db.commit()
        
        # Trigger learning update
        background_tasks.add_task(
            learning_engine.update_from_feedback,
            feedback_id=feedback.id,
            interaction=interaction
        )
        
        logger.info(f"📝 Feedback received for interaction {request.interaction_id}")
        
        return {"status": "success", "message": "Thank you for your feedback!"}
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/progress", response_model=StudentProgress)
async def get_student_progress(
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student)
):
    """Get student's learning progress and statistics"""
    
    try:
        # Get recent interactions
        recent_interactions = db.query(Interaction).filter(
            Interaction.student_id == current_student.id
        ).order_by(Interaction.created_at.desc()).limit(50).all()
        
        # Calculate progress metrics
        if recent_interactions:
            avg_frustration = sum(i.frustration_score for i in recent_interactions) / len(recent_interactions)
            frustration_trend = calculate_trend([i.frustration_score for i in recent_interactions])
            
            # Get concept mastery
            all_concepts = []
            for interaction in recent_interactions:
                if interaction.concepts:
                    all_concepts.extend(interaction.concepts)
            
            concept_mastery = calculate_concept_mastery(all_concepts)
            
            # Get learning velocity
            learning_velocity = calculate_learning_velocity(recent_interactions)
            
        else:
            avg_frustration = 0.0
            frustration_trend = "stable"
            concept_mastery = {}
            learning_velocity = 0.0
        
        return StudentProgress(
            student_id=current_student.id,
            total_interactions=len(recent_interactions),
            average_frustration=avg_frustration,
            frustration_trend=frustration_trend,
            concept_mastery=concept_mastery,
            learning_velocity=learning_velocity,
            strengths=identify_strengths(concept_mastery),
            areas_for_improvement=identify_weaknesses(concept_mastery),
            recent_milestones=get_recent_milestones(current_student.id, db)
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting progress: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_interaction_history(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_student: Student = Depends(get_current_student)
):
    """Get student's interaction history"""
    
    try:
        interactions = db.query(Interaction).filter(
            Interaction.student_id == current_student.id
        ).order_by(Interaction.created_at.desc()).offset(offset).limit(limit).all()
        
        return {
            "interactions": [
                {
                    "id": i.id,
                    "message": i.message,
                    "response": i.response,
                    "frustration_score": i.frustration_score,
                    "concepts": i.concepts,
                    "created_at": i.created_at.isoformat()
                }
                for i in interactions
            ],
            "total": db.query(Interaction).filter(
                Interaction.student_id == current_student.id
            ).count()
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def calculate_trend(scores: List[float]) -> str:
    """Calculate trend from a list of scores"""
    if len(scores) < 2:
        return "stable"
    
    # Simple linear regression
    n = len(scores)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(scores) / n
    
    numerator = sum((x[i] - x_mean) * (scores[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return "stable"
    
    slope = numerator / denominator
    
    if slope < -0.1:
        return "improving"
    elif slope > 0.1:
        return "worsening"
    else:
        return "stable"

def calculate_concept_mastery(concepts: List[Dict]) -> Dict[str, float]:
    """Calculate mastery level for each concept"""
    concept_scores = {}
    concept_counts = {}
    
    for concept_list in concepts:
        for concept in concept_list:
            name = concept.get("name", "unknown")
            score = concept.get("mastery_score", 0.5)
            
            if name not in concept_scores:
                concept_scores[name] = 0
                concept_counts[name] = 0
            
            concept_scores[name] += score
            concept_counts[name] += 1
    
    # Calculate averages
    mastery = {}
    for name, total_score in concept_scores.items():
        mastery[name] = total_score / concept_counts[name]
    
    return mastery

def calculate_learning_velocity(interactions: List[Interaction]) -> float:
    """Calculate how quickly the student is learning"""
    if len(interactions) < 2:
        return 0.0
    
    # Compare frustration scores over time
    early_frustration = sum(i.frustration_score for i in interactions[-10:]) / min(10, len(interactions))
    recent_frustration = sum(i.frustration_score for i in interactions[:10]) / min(10, len(interactions))
    
    # Positive velocity means improving (less frustration)
    velocity = (early_frustration - recent_frustration) / early_frustration if early_frustration > 0 else 0
    
    return max(-1.0, min(1.0, velocity))  # Clamp between -1 and 1

def identify_strengths(mastery: Dict[str, float]) -> List[str]:
    """Identify concepts where student excels"""
    return [concept for concept, score in mastery.items() if score > 0.7]

def identify_weaknesses(mastery: Dict[str, float]) -> List[str]:
    """Identify concepts that need work"""
    return [concept for concept, score in mastery.items() if score < 0.4]

def get_recent_milestones(student_id: str, db: Session) -> List[Dict]:
    """Get recent learning milestones"""
    # This would be more sophisticated in production
    return [
        {"date": datetime.utcnow().isoformat(), "achievement": "Completed 10 interactions"},
        {"date": datetime.utcnow().isoformat(), "achievement": "Mastered loops concept"}
    ]