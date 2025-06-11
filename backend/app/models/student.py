
#backend/app/models/student.py
from sqlalchemy import Column, String, Float, DateTime, JSON, Boolean, ForeignKey, Integer
from sqlalchemy.orm import relationship
from datetime import datetime
from ..db.database import Base

class Student(Base):
    __tablename__ = "students"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    skill_level = Column(String, default="beginner")
    personalized_weights = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    interactions = relationship("Interaction", back_populates="student")

class Interaction(Base):
    __tablename__ = "interactions"
    
    id = Column(String, primary_key=True)
    student_id = Column(String, ForeignKey("students.id"))
    message = Column(String)
    response = Column(String)
    frustration_score = Column(Float)
    intervention_level = Column(Integer)
    features = Column(JSON)
    concepts = Column(JSON)
    additional_data = Column(JSON)  # Changed from 'metadata' to 'additional_data'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    student = relationship("Student", back_populates="interactions")
    feedback = relationship("Feedback", back_populates="interaction", uselist=False)

class Feedback(Base):
    __tablename__ = "feedback"
    
    id = Column(String, primary_key=True)
    interaction_id = Column(String, ForeignKey("interactions.id"))
    student_id = Column(String, ForeignKey("students.id"))
    helpful = Column(Boolean)
    frustration_reduced = Column(Boolean)
    clarity_rating = Column(Integer)
    additional_comments = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    interaction = relationship("Interaction", back_populates="feedback")
