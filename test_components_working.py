import asyncio
import sys
sys.path.append('.')
from backend.app.services.nlp.feature_extractor import FeatureExtractor
from backend.app.services.ml.frustration_estimator_simple import SimpleFrustrationEstimator

async def test_components():
    print("🔧 COMPONENT TESTS")
    print("=" * 60)
    
    # Test Feature Extractor
    print("\n1️⃣ Testing Feature Extractor...")
    extractor = FeatureExtractor()
    
    # Test with a message that definitely has frustration keywords
    test_message = "I am so frustrated and stuck with this error!"
    features = await extractor.extract(test_message)
    
    print(f"   Test message: '{test_message}'")
    print(f"   Features extracted: {len(features)}")
    print(f"   Keywords found: {features['keywords']}")
    print(f"   Sentiment: {features['sentiment']:.2f}")
    assert features['keywords'] >= 2  # "frustrated" and "stuck"
    print("✅ Feature extractor working correctly")
    
    # Test ML Estimator
    print("\n2️⃣ Testing ML Estimator...")
    estimator = SimpleFrustrationEstimator()
    
    # Test with actual extracted features
    score1 = estimator.estimate(features)
    print(f"   Frustrated message score: {score1:.2f}")
    assert score1 > 5  # Should be high
    
    # Test calm message
    calm_features = await extractor.extract("Can you please help me understand this?")
    calm_score = estimator.estimate(calm_features)
    print(f"   Calm message score: {calm_score:.2f}")
    assert calm_score < score1  # Should be lower than frustrated
    
    print("✅ ML scoring working correctly")
    
    # Test different frustration levels
    print("\n3️⃣ Testing Frustration Level Detection...")
    
    test_messages = [
        "How do functions work?",  # Calm
        "I'm confused about this",  # Mild
        "This is really frustrating!",  # High
        "I HATE THIS SO MUCH!!!",  # Very high
    ]
    
    scores = []
    for msg in test_messages:
        feat = await extractor.extract(msg)
        score = estimator.estimate(feat)
        scores.append(score)
        print(f"   '{msg[:30]}...' → {score:.2f}")
    
    # Verify scores are in ascending order (more frustration = higher score)
    assert scores[0] < scores[2]  # Calm < Frustrating
    assert scores[2] < scores[3]  # Frustrating < HATE
    
    print("✅ Frustration levels detected correctly")
    
    print("\n✅ ALL COMPONENT TESTS PASSED!")
    
    # Show what keywords are actually in the system
    print("\n📝 System Information:")
    print(f"   Frustration keywords: {len(extractor.frustration_keywords)}")
    print(f"   Sample keywords: {extractor.frustration_keywords[:5]}")

if __name__ == "__main__":
    asyncio.run(test_components())
