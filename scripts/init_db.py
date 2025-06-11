# scripts/init_db.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from backend.app.db.database import Base, engine
from backend.app.models.student import Student, Interaction, Feedback
from backend.app.models.concept import Concept, ConceptRelationship, StudentConceptMastery
from backend.app.models.learning import LearningData, ModelCheckpoint
from loguru import logger

def init_database():
    """Initialize database with tables"""
    logger.info("🗄️ Initializing database...")
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Add initial data if needed
        from backend.app.db.database import SessionLocal
        db = SessionLocal()
        
        # Check if concepts exist
        concept_count = db.query(Concept).count()
        if concept_count == 0:
            logger.info("📚 Adding initial concepts...")
            initial_concepts = [
                Concept(name="variables", category="basics", difficulty=0.1, 
                       description="Variable declaration and assignment"),
                Concept(name="conditionals", category="control_flow", difficulty=0.3,
                       description="If-else statements and boolean logic"),
                Concept(name="loops", category="control_flow", difficulty=0.4,
                       description="For and while loops"),
                Concept(name="functions", category="basics", difficulty=0.5,
                       description="Function definition and calling"),
                Concept(name="lists", category="data_structures", difficulty=0.3,
                       description="List operations and methods"),
                Concept(name="dictionaries", category="data_structures", difficulty=0.4,
                       description="Dictionary operations and methods"),
                Concept(name="classes", category="oop", difficulty=0.6,
                       description="Class definition and object creation"),
                Concept(name="recursion", category="algorithms", difficulty=0.8,
                       description="Recursive function calls"),
                Concept(name="exceptions", category="error_handling", difficulty=0.5,