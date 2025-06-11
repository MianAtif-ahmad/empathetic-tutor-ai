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
    features = await extractor.extract("This is SO FRUSTRATING!!!")
    
    assert 'sentiment' in features
    assert features['exclamations'] == 3
    assert features['caps_ratio'] > 0
    assert features['keywords'] > 0
    print("✅ Feature extractor working")
    print(f"   Extracted {len(features)} features")
    
    # Test ML Estimator
    print("\n2️⃣ Testing ML Estimator...")
    estimator = SimpleFrustrationEstimator()
    
    # Low frustration
    low_features = {'sentiment': 0.5, 'keywords': 0}
    low_score = estimator.estimate(low_features)
    assert low_score < 5
    print(f"✅ Low frustration: {low_score}")
    
    # High frustration
    high_features = {
        'sentiment': -0.8, 
        'keywords': 3,
        'exclamations': 5,
        'caps_ratio': 0.5
    }
    high_score = estimator.estimate(high_features)
    assert high_score > 7
    print(f"✅ High frustration: {high_score}")
    
    print("\n✅ ALL COMPONENT TESTS PASSED!")

if __name__ == "__main__":
    asyncio.run(test_components())
