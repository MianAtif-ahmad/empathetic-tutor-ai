# tests/test_integration.py

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.ml.frustration_estimator import FrustrationEstimator
from backend.app.services.ml.concept_extractor import ConceptExtractor, ProgrammingConcept
from backend.app.services.intervention.response_generator import ResponseGenerator, InterventionStrategy
from backend.app.services.feedback.learning_engine import LearningEngine
from backend.app.services.nlp.feature_extractor import FeatureExtractor
from backend.app.core.config import settings

# Test client
client = TestClient(app)

class TestFrustrationEstimator:
    """Test suite for FrustrationEstimator"""
    
    @pytest.fixture
    def estimator(self):
        return FrustrationEstimator()
    
    @pytest.fixture
    def sample_features(self):
        return {
            "sentiment": -0.5,
            "keywords": 2.0,
            "punctuation": 3.0,
            "grammar": 0.7,
            "latency": 0.5,
            "repetition": 1.0,
            "help_seeking": 1.0
        }
    
    @pytest.fixture
    def sample_concepts(self):
        return [
            ProgrammingConcept(
                name="recursion",
                category="algorithm",
                confidence=0.8,
                context="struggling with recursion",
                related_concepts=["functions", "base_case"],
                difficulty_level=0.8
            )
        ]
    
    @pytest.mark.asyncio
    async def test_estimate_basic(self, estimator, sample_features, sample_concepts):
        """Test basic frustration estimation"""
        score = await estimator.estimate(
            features=sample_features,
            student_id="test_student",
            concepts=sample_concepts,
            use_ml=False
        )
        
        assert isinstance(score, float)
        assert 0 <= score <= 10
    
    @pytest.mark.asyncio
    async def test_estimate_with_concepts(self, estimator, sample_features, sample_concepts):
        """Test estimation with concept difficulty"""
        score = await estimator.estimate(
            features=sample_features,
            student_id="test_student",
            concepts=sample_concepts,
            use_ml=False
        )
        
        # High difficulty concept should increase frustration
        assert score > 5.0
    
    @pytest.mark.asyncio
    async def test_personalized_weights(self, estimator):
        """Test student-specific weight retrieval"""
        student_id = "test_student"
        weights = await estimator._get_student_weights(student_id)
        
        assert isinstance(weights, dict)
        assert "sentiment" in weights
        assert all(isinstance(v, (int, float)) for v in weights.values())
    
    def test_feature_vector_preparation(self, estimator, sample_features):
        """Test feature vector preparation"""
        vector = estimator._prepare_feature_vector(sample_features)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(estimator.base_weights)

class TestConceptExtractor:
    """Test suite for ConceptExtractor"""
    
    @pytest.fixture
    def extractor(self):
        return ConceptExtractor()
    
    @pytest.mark.asyncio
    async def test_extract_from_message(self, extractor):
        """Test concept extraction from natural language"""
        message = "I'm having trouble with recursion, my function keeps calling itself infinitely"
        concepts = await extractor.extract(message)
        
        assert len(concepts) > 0
        assert any(c.name == "recursion" for c in concepts)
        assert all(isinstance(c, ProgrammingConcept) for c in concepts)
    
    @pytest.mark.asyncio
    async def test_extract_from_code(self, extractor):
        """Test concept extraction from code"""
        code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
"""
        concepts = await extractor.extract("", code_snippet=code)
        
        assert any(c.name == "functions" for c in concepts)
        assert any(c.name == "recursion" for c in concepts)
        assert any(c.name == "conditionals" for c in concepts)
    
    @pytest.mark.asyncio
    async def test_extract_from_error(self, extractor):
        """Test concept extraction from error messages"""
        message = "RecursionError: maximum recursion depth exceeded"
        concepts = await extractor.extract(message)
        
        assert any(c.name == "recursion" for c in concepts)
        assert any(c.category == "error_related" for c in concepts)
    
    @pytest.mark.asyncio
    async def test_syntax_error_detection(self, extractor):
        """Test syntax error detection"""
        code = "def broken_function("
        concepts = await extractor.extract("", code_snippet=code)
        
        assert any(c.name == "syntax_error" for c in concepts)

class TestResponseGenerator:
    """Test suite for ResponseGenerator"""
    
    @pytest.fixture
    def generator(self):
        return ResponseGenerator()
    
    @pytest.fixture
    def sample_intervention(self):
        return InterventionStrategy(
            level=2,
            tone="encouraging",
            scaffolding_level="moderate",
            hint_progression=["hint1", "hint2"],
            focus_concepts=["recursion"]
        )
    
    @pytest.fixture
    def sample_student(self):
        student = Mock()
        student.id = "test_student"
        student.skill_level = "intermediate"
        return student
    
    @pytest.mark.asyncio
    async def test_generate_response(self, generator, sample_intervention, sample_student):
        """Test response generation"""
        with patch.object(generator.llm_client, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = "Here's a helpful response about recursion..."
            
            response = await generator.generate(
                intervention=sample_intervention,
                message="I don't understand recursion",
                concepts=[],
                student_profile=sample_student,
                frustration_level=6.0
            )
            
            assert response.content != ""
            assert response.tone_level == 2
            assert isinstance(response.hints, list)
            assert isinstance(response.resources, list)
    
    def test_hint_simplification(self, generator):
        """Test hint simplification for beginners"""
        complex_hint = "Consider using a recursive approach to iterate through the collection"
        simple_hint = generator._simplify_hint(complex_hint)
        
        assert "recursive" not in simple_hint
        assert "iterate" not in simple_hint
    
    def test_resource_filtering(self, generator):
        """Test resource filtering by skill level"""
        concepts = [
            ProgrammingConcept(
                name="loops",
                category="syntax",
                confidence=0.8,
                context="",
                related_concepts=[],
                difficulty_level=0.4
            )
        ]
        
        resources = generator._get_resources(concepts, "beginner")
        
        # Should prefer tutorials and visualizations for beginners
        assert all(r["type"] in ["tutorial", "visualization"] for r in resources)

class TestLearningEngine:
    """Test suite for LearningEngine"""
    
    @pytest.fixture
    def engine(self):
        return LearningEngine()
    
    @pytest.fixture
    def sample_features(self):
        return {
            "sentiment": -0.5,
            "keywords": 2.0,
            "punctuation": 1.0,
            "grammar": 0.8,
            "latency": 0.3
        }
    
    @pytest.mark.asyncio
    async def test_update_from_interaction(self, engine, sample_features):
        """Test learning from interaction"""
        await engine.update_from_interaction(
            interaction_id="test_interaction",
            student_id="test_student",
            features=sample_features,
            frustration_score=6.5,
            intervention=InterventionStrategy(
                level=2,
                tone="supportive",
                scaffolding_level="moderate",
                hint_progression=[],
                focus_concepts=[]
            ),
            concepts=[]
        )
        
        # Check that entry was queued
        assert engine.learning_queue.qsize() > 0
    
    def test_effectiveness_calculation(self, engine):
        """Test effectiveness calculation from feedback"""
        feedback = Mock()
        feedback.helpful = True
        feedback.frustration_reduced = True
        feedback.clarity_rating = 4
        
        effectiveness = engine._calculate_effectiveness(feedback)
        
        assert 0 <= effectiveness <= 1
        assert effectiveness > 0.5  # Should be positive with good feedback
    
    @pytest.mark.asyncio
    async def test_optimal_intervention(self, engine):
        """Test optimal intervention selection"""
        # Without trained model, should use rule-based
        level = await engine.get_optimal_intervention(7.0)
        assert level == 2
        
        level = await engine.get_optimal_intervention(2.0)
        assert level == 0
        
        level = await engine.get_optimal_intervention(9.0)
        assert level == 3
    
    def test_feature_vector_conversion(self, engine, sample_features):
        """Test feature dictionary to vector conversion"""
        vector = engine._features_to_vector(sample_features)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(engine.feature_importance)

class TestFeatureExtractor:
    """Test suite for FeatureExtractor"""
    
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()
    
    @pytest.mark.asyncio
    async def test_extract_features(self, extractor):
        """Test feature extraction from message"""
        message = "I'm so frustrated!!! This stupid recursion won't work. HELP!"
        features = await extractor.extract(message, [], "test_student")
        
        assert "sentiment" in features
        assert features["sentiment"] < 0  # Negative sentiment
        assert features["punctuation"] > 2  # Multiple exclamation marks
        assert features["keywords"] > 0  # Contains frustration keywords
        assert features["help_seeking"] == 1.0  # Contains "HELP"
    
    @pytest.mark.asyncio
    async def test_repetition_detection(self, extractor):
        """Test repetition detection"""
        message = "Why doesn't this work?"
        previous = ["Why doesn't this work?", "Other message"]
        
        features = await extractor.extract(message, previous, "test_student")
        
        assert features["repetition"] == 1.0

class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture
    def auth_headers(self):
        # Mock authentication headers
        return {"Authorization": "Bearer test_token"}
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_process_message_endpoint(self, auth_headers):
        """Test message processing endpoint"""
        with patch('backend.app.api.routes.student_routes.get_current_student') as mock_auth:
            mock_student = Mock()
            mock_student.id = "test_student"
            mock_student.skill_level = "intermediate"
            mock_auth.return_value = mock_student
            
            request_data = {
                "message": "I don't understand loops",
                "context": {},
                "code_snippet": "for i in range(10):"
            }
            
            # Mock the various services
            with patch('backend.app.api.routes.student_routes.feature_extractor.extract', new_callable=AsyncMock) as mock_extract:
                mock_extract.return_value = {"sentiment": -0.3, "keywords": 1.0}
                
                with patch('backend.app.api.routes.student_routes.concept_extractor.extract', new_callable=AsyncMock) as mock_concepts:
                    mock_concepts.return_value = []
                    
                    with patch('backend.app.api.routes.student_routes.frustration_estimator.estimate', new_callable=AsyncMock) as mock_estimate:
                        mock_estimate.return_value = 5.5
                        
                        with patch('backend.app.api.routes.student_routes.decision_engine.decide', new_callable=AsyncMock) as mock_decide:
                            mock_decide.return_value = InterventionStrategy(
                                level=1,
                                tone="supportive",
                                scaffolding_level="minimal",
                                hint_progression=[],
                                focus_concepts=[]
                            )
                            
                            with patch('backend.app.api.routes.student_routes.response_generator.generate', new_callable=AsyncMock) as mock_generate:
                                from backend.app.services.intervention.response_generator import GeneratedResponse
                                mock_generate.return_value = GeneratedResponse(
                                    content="Let me help you with loops...",
                                    tone_level=1,
                                    hints=["Think about repetition"],
                                    resources=[],
                                    next_steps=["Practice more"],
                                    estimated_helpfulness=0.7
                                )
                                
                                response = client.post(
                                    "/api/v1/student/message",
                                    json=request_data,
                                    headers=auth_headers
                                )
                                
                                assert response.status_code == 200
                                data = response.json()
                                assert "response" in data
                                assert "frustration_score" in data
                                assert "intervention_level" in data

class TestIntegrationFlow:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_interaction_flow(self):
        """Test complete interaction flow from message to response"""
        # Initialize components
        feature_extractor = FeatureExtractor()
        concept_extractor = ConceptExtractor()
        frustration_estimator = FrustrationEstimator()
        response_generator = ResponseGenerator()
        learning_engine = LearningEngine()
        
        # Student message
        message = "I keep getting RecursionError and I don't know why!!"
        student_id = "test_student"
        
        # Extract features
        features = await feature_extractor.extract(message, [], student_id)
        assert features["sentiment"] < 0
        assert features["punctuation"] > 0
        
        # Extract concepts
        concepts = await concept_extractor.extract(message)
        assert any(c.name == "recursion" for c in concepts)
        
        # Estimate frustration
        frustration = await frustration_estimator.estimate(
            features=features,
            student_id=student_id,
            concepts=concepts,
            use_ml=False
        )
        assert frustration > 5.0  # Should be frustrated
        
        # Generate response
        intervention = InterventionStrategy(
            level=2,
            tone="supportive",
            scaffolding_level="moderate",
            hint_progression=["Check base case"],
            focus_concepts=["recursion"]
        )
        
        with patch.object(response_generator.llm_client, 'generate', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "I understand recursion errors can be frustrating..."
            
            student = Mock()
            student.id = student_id
            student.skill_level = "intermediate"
            
            response = await response_generator.generate(
                intervention=intervention,
                message=message,
                concepts=concepts,
                student_profile=student,
                frustration_level=frustration
            )
            
            assert response.content != ""
            assert response.tone_level == 2
            assert len(response.hints) > 0
        
        # Update learning
        await learning_engine.update_from_interaction(
            interaction_id="test_interaction",
            student_id=student_id,
            features=features,
            frustration_score=frustration,
            intervention=intervention,
            concepts=concepts
        )
        
        assert learning_engine.learning_queue.qsize() > 0

# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])