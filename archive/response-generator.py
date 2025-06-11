# backend/app/services/intervention/response_generator.py

from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import asyncio
import json
from loguru import logger

from ...core.config import settings
from ...models.student import Student, Interaction
from ..ml.concept_extractor import ProgrammingConcept
from ..nlp.llm_client import LLMClient
from ...utils.prompt_templates import PromptTemplates
from ...utils.metrics import track_metric
from .knowledge_graph import KnowledgeGraphManager

@dataclass
class InterventionStrategy:
    """Represents an intervention strategy"""
    level: int  # 0: no action, 1: gentle nudge, 2: full support, 3: direct
    tone: str  # supportive, neutral, direct, encouraging
    scaffolding_level: str  # minimal, moderate, extensive
    hint_progression: List[str]
    focus_concepts: List[str]
    
@dataclass
class GeneratedResponse:
    """Represents a generated tutor response"""
    content: str
    tone_level: int
    hints: List[str]
    resources: List[Dict[str, str]]
    next_steps: List[str]
    estimated_helpfulness: float

class ResponseGenerator:
    """
    Generates adaptive, pedagogically-sound responses based on student state
    """
    
    def __init__(self):
        self.llm_client = LLMClient()
        self.prompt_templates = PromptTemplates()
        self.kg_manager = KnowledgeGraphManager()
        
        # Response strategies based on frustration and concept difficulty
        self.response_strategies = self._initialize_strategies()
        
        # Hint progression templates
        self.hint_progressions = self._load_hint_progressions()
        
        # Resource database
        self.resources = self._load_resources()
    
    def _initialize_strategies(self) -> Dict[Tuple[int, float], InterventionStrategy]:
        """Initialize response strategies based on frustration level and concept difficulty"""
        return {
            # (frustration_level, concept_difficulty) -> strategy
            (0, 0.0): InterventionStrategy(
                level=0,
                tone="neutral",
                scaffolding_level="minimal",
                hint_progression=[],
                focus_concepts=[]
            ),
            (1, 0.3): InterventionStrategy(
                level=1,
                tone="supportive",
                scaffolding_level="minimal",
                hint_progression=["direction", "example"],
                focus_concepts=["current"]
            ),
            (2, 0.5): InterventionStrategy(
                level=2,
                tone="encouraging",
                scaffolding_level="moderate",
                hint_progression=["recognition", "direction", "example", "partial_solution"],
                focus_concepts=["current", "prerequisites"]
            ),
            (3, 0.7): InterventionStrategy(
                level=3,
                tone="direct",
                scaffolding_level="extensive",
                hint_progression=["immediate_fix", "explanation", "alternative"],
                focus_concepts=["simplified", "fundamentals"]
            )
        }
    
    def _load_hint_progressions(self) -> Dict[str, List[str]]:
        """Load hint progression templates"""
        return {
            "loops": [
                "Think about what needs to happen repeatedly",
                "Consider using a for loop with range() or iterating over a collection",
                "The pattern is: for item in collection: do_something(item)",
                "Here's a starter: for i in range(n):"
            ],
            "recursion": [
                "Every recursive function needs a base case - when should it stop?",
                "Think about how the problem can be broken into smaller versions",
                "Make sure each recursive call moves toward the base case",
                "Pattern: if base_case: return result; else: return recursive_call(smaller_problem)"
            ],
            "debugging": [
                "What exactly is the error message telling you?",
                "Try adding print statements to see the values at each step",
                "Break down the problem - which line is causing the issue?",
                "Use a debugger or step through your code mentally"
            ],
            "syntax": [
                "Check for missing colons, parentheses, or quotes",
                "Python is indentation-sensitive - ensure consistent spacing",
                "Look at the line number in the error message",
                "Common issues: missing :, unmatched (), incorrect indentation"
            ]
        }
    
    def _load_resources(self) -> Dict[str, List[Dict]]:
        """Load educational resources"""
        return {
            "loops": [
                {"title": "Python Loops Tutorial", "url": "https://docs.python.org/3/tutorial/controlflow.html#for-statements", "type": "documentation"},
                {"title": "Visualizing Loops", "url": "https://pythontutor.com/", "type": "visualization"}
            ],
            "recursion": [
                {"title": "Recursion Explained", "url": "https://realpython.com/python-recursion/", "type": "tutorial"},
                {"title": "Recursion Visualizer", "url": "https://recursion.now.sh/", "type": "visualization"}
            ],
            "functions": [
                {"title": "Functions in Python", "url": "https://docs.python.org/3/tutorial/controlflow.html#defining-functions", "type": "documentation"},
                {"title": "Function Best Practices", "url": "https://realpython.com/defining-your-own-python-function/", "type": "tutorial"}
            ]
        }
    
    async def generate(
        self,
        intervention: InterventionStrategy,
        message: str,
        concepts: List[ProgrammingConcept],
        student_profile: Student,
        frustration_level: float,
        context_window: int = 5
    ) -> GeneratedResponse:
        """
        Generate an adaptive response based on all inputs
        """
        start_time = datetime.utcnow()
        
        try:
            # Get student's learning history and context
            context = await self._build_context(student_profile, context_window)
            
            # Get knowledge graph insights
            kg_insights = await self.kg_manager.get_concept_insights(
                student_profile.id,
                [c.name for c in concepts]
            )
            
            # Select appropriate prompt template
            prompt = await self._build_prompt(
                intervention=intervention,
                message=message,
                concepts=concepts,
                student_profile=student_profile,
                frustration_level=frustration_level,
                context=context,
                kg_insights=kg_insights
            )
            
            # Generate response using LLM
            llm_response = await self.llm_client.generate(
                prompt=prompt,
                temperature=self._get_temperature(frustration_level),
                max_tokens=800
            )
            
            # Extract hints based on intervention level
            hints = await self._generate_hints(
                concepts=concepts,
                intervention=intervention,
                student_level=student_profile.skill_level
            )
            
            # Get relevant resources
            resources = self._get_resources(concepts, student_profile.skill_level)
            
            # Generate next steps
            next_steps = await self._generate_next_steps(
                concepts=concepts,
                student_profile=student_profile,
                kg_insights=kg_insights
            )
            
            # Estimate helpfulness
            helpfulness = self._estimate_helpfulness(
                intervention=intervention,
                concepts=concepts,
                frustration_level=frustration_level
            )
            
            # Track metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            await track_metric("response_generation_time", processing_time)
            await track_metric("response_length", len(llm_response))
            
            logger.info(
                f"💬 Generated response for student {student_profile.id} "
                f"(frustration: {frustration_level:.2f}, intervention: {intervention.level})"
            )
            
            return GeneratedResponse(
                content=llm_response,
                tone_level=intervention.level,
                hints=hints,
                resources=resources,
                next_steps=next_steps,
                estimated_helpfulness=helpfulness
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {str(e)}")
            # Fallback response
            return GeneratedResponse(
                content=self._get_fallback_response(frustration_level),
                tone_level=2,
                hints=["Try breaking down the problem into smaller steps"],
                resources=[],
                next_steps=["Review the basics", "Ask a more specific question"],
                estimated_helpfulness=0.5
            )
    
    async def _build_context(self, student: Student, window: int) -> Dict:
        """Build context from student's recent interactions"""
        from ...db.database import SessionLocal
        db = SessionLocal()
        
        try:
            recent_interactions = db.query(Interaction).filter(
                Interaction.student_id == student.id
            ).order_by(Interaction.created_at.desc()).limit(window).all()
            
            context = {
                "recent_topics": [],
                "struggling_concepts": [],
                "mastered_concepts": [],
                "error_patterns": [],
                "learning_pace": "normal"
            }
            
            # Analyze recent interactions
            for interaction in recent_interactions:
                if interaction.concepts:
                    for concept in interaction.concepts:
                        context["recent_topics"].append(concept.get("name"))
                        
                        # Track struggling vs mastered
                        if interaction.frustration_score > 6:
                            context["struggling_concepts"].append(concept.get("name"))
                        elif interaction.frustration_score < 3:
                            context["mastered_concepts"].append(concept.get("name"))
            
            # Deduplicate
            context["recent_topics"] = list(set(context["recent_topics"]))
            context["struggling_concepts"] = list(set(context["struggling_concepts"]))
            context["mastered_concepts"] = list(set(context["mastered_concepts"]))
            
            # Determine learning pace
            if len(context["struggling_concepts"]) > len(context["mastered_concepts"]):
                context["learning_pace"] = "slow"
            elif len(context["mastered_concepts"]) > len(context["struggling_concepts"]) * 2:
                context["learning_pace"] = "fast"
            
            return context
            
        finally:
            db.close()
    
    async def _build_prompt(
        self,
        intervention: InterventionStrategy,
        message: str,
        concepts: List[ProgrammingConcept],
        student_profile: Student,
        frustration_level: float,
        context: Dict,
        kg_insights: Dict
    ) -> str:
        """Build the prompt for LLM"""
        
        # Base system prompt
        system_prompt = self.prompt_templates.get_system_prompt(
            tone=intervention.tone,
            scaffolding_level=intervention.scaffolding_level
        )
        
        # Student context
        student_context = f"""
Student Profile:
- Skill Level: {student_profile.skill_level}
- Learning Pace: {context['learning_pace']}
- Current Frustration: {frustration_level:.1f}/10
- Struggling Concepts: {', '.join(context['struggling_concepts'][:3])}
- Recent Topics: {', '.join(context['recent_topics'][:5])}

Current Question/Issue:
{message}

Identified Concepts: {', '.join([c.name for c in concepts[:3]])}
Concept Difficulties: {', '.join([f"{c.name}({c.difficulty_level:.1f})" for c in concepts[:3]])}
"""
        
        # Add knowledge graph insights
        if kg_insights.get("prerequisites_missing"):
            student_context += f"\nMissing Prerequisites: {', '.join(kg_insights['prerequisites_missing'])}"
        
        # Response guidelines based on intervention
        guidelines = self._get_response_guidelines(intervention, frustration_level)
        
        # Combine into full prompt
        prompt = f"{system_prompt}\n\n{student_context}\n\n{guidelines}"
        
        return prompt
    
    def _get_response_guidelines(
        self,
        intervention: InterventionStrategy,
        frustration_level: float
    ) -> str:
        """Get response guidelines based on intervention strategy"""
        
        if intervention.level == 0:
            return "The student seems to be doing well. Provide minimal guidance if needed."
        
        elif intervention.level == 1:
            return """
Provide a gentle nudge:
1. Acknowledge their effort
2. Give a subtle hint about the direction to explore
3. Ask a guiding question
4. Keep response concise and encouraging
"""
        
        elif intervention.level == 2:
            return """
Provide supportive guidance:
1. Validate their struggle - this concept is challenging
2. Break down the problem into steps
3. Provide a concrete example or analogy
4. Give specific hints without revealing the solution
5. Suggest what to focus on
6. End with encouragement
"""
        
        else:  # level 3
            return """
Provide direct assistance (high frustration detected):
1. Get straight to the point - no excessive reassurance
2. Identify the specific issue clearly
3. Provide a clear example that directly addresses their problem
4. Give actionable steps to fix the immediate issue
5. Keep emotional language minimal
6. Focus on solving the problem efficiently
"""
    
    async def _generate_hints(
        self,
        concepts: List[ProgrammingConcept],
        intervention: InterventionStrategy,
        student_level: str
    ) -> List[str]:
        """Generate progressive hints"""
        hints = []
        
        for concept in concepts[:2]:  # Focus on top 2 concepts
            if concept.name in self.hint_progressions:
                progression = self.hint_progressions[concept.name]
                
                # Select hints based on intervention level
                if intervention.level == 1:
                    hints.extend(progression[:1])
                elif intervention.level == 2:
                    hints.extend(progression[:2])
                else:
                    hints.extend(progression[:3])
        
        # Adjust hints based on student level
        if student_level == "beginner":
            hints = [self._simplify_hint(hint) for hint in hints]
        
        return hints[:4]  # Maximum 4 hints
    
    def _simplify_hint(self, hint: str) -> str:
        """Simplify hint for beginners"""
        # Replace technical terms with simpler ones
        replacements = {
            "iterate": "go through each item",
            "recursive": "function that calls itself",
            "base case": "stopping condition",
            "collection": "list or group of items"
        }
        
        for term, simple in replacements.items():
            hint = hint.replace(term, simple)
        
        return hint
    
    def _get_resources(
        self,
        concepts: List[ProgrammingConcept],
        skill_level: str
    ) -> List[Dict[str, str]]:
        """Get relevant learning resources"""
        resources = []
        
        for concept in concepts[:2]:  # Top 2 concepts
            if concept.name in self.resources:
                concept_resources = self.resources[concept.name]
                
                # Filter by skill level
                if skill_level == "beginner":
                    # Prefer tutorials and visualizations
                    filtered = [r for r in concept_resources if r["type"] in ["tutorial", "visualization"]]
                else:
                    filtered = concept_resources
                
                resources.extend(filtered[:2])  # Max 2 resources per concept
        
        return resources[:3]  # Maximum 3 resources total
    
    async def _generate_next_steps(
        self,
        concepts: List[ProgrammingConcept],
        student_profile: Student,
        kg_insights: Dict
    ) -> List[str]:
        """Generate recommended next steps"""
        next_steps = []
        
        # Address missing prerequisites first
        if kg_insights.get("prerequisites_missing"):
            for prereq in kg_insights["prerequisites_missing"][:2]:
                next_steps.append(f"Review the basics of {prereq}")
        
        # Practice current concepts
        for concept in concepts[:2]:
            if concept.confidence < 0.7:
                next_steps.append(f"Practice more {concept.name} problems")
        
        # Suggest related concepts to explore
        if student_profile.skill_level != "beginner":
            for concept in concepts[:1]:
                if concept.related_concepts:
                    next_steps.append(f"Explore {concept.related_concepts[0]}")
        
        return next_steps[:3]  # Maximum 3 next steps
    
    def _estimate_helpfulness(
        self,
        intervention: InterventionStrategy,
        concepts: List[ProgrammingConcept],
        frustration_level: float
    ) -> float:
        """Estimate how helpful the response will be"""
        base_score = 0.5
        
        # Intervention appropriateness
        if 3 <= frustration_level <= 7:
            base_score += 0.2  # Good range for intervention
        
        # Concept coverage
        if concepts:
            avg_difficulty = sum(c.difficulty_level for c in concepts) / len(concepts)
            if 0.3 <= avg_difficulty <= 0.7:
                base_score += 0.1  # Moderate difficulty is ideal
        
        # Scaffolding match
        if intervention.scaffolding_level == "moderate":
            base_score += 0.1
        
        # Hint availability
        if intervention.hint_progression:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _get_temperature(self, frustration_level: float) -> float:
        """Get LLM temperature based on frustration"""
        # Lower temperature for high frustration (more focused responses)
        if frustration_level > 7:
            return 0.3
        elif frustration_level > 5:
            return 0.5
        else:
            return 0.7
    
    def _get_fallback_response(self, frustration_level: float) -> str:
        """Get fallback response when generation fails"""
        if frustration_level > 7:
            return (
                "I see you're having trouble. Let's break this down step by step. "
                "Can you tell me specifically which part is causing the issue?"
            )
        else:
            return (
                "I'd be happy to help! Could you provide more details about "
                "what you're trying to accomplish and where you're stuck?"
            )