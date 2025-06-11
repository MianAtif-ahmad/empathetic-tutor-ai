import sqlite3
import os

print("🔍 CURRENT SYSTEM CAPABILITIES")
print("=" * 60)

print("\n✅ IMPLEMENTED:")
print("  1. Basic API (FastAPI)")
print("  2. SQLite Database")
print("  3. Simple keyword-based frustration detection")
print("  4. Basic concept detection (keyword matching)")
print("  5. Data persistence")
print("  6. API documentation (/docs)")

print("\n❌ NOT YET IMPLEMENTED:")
print("  1. ML-based frustration estimation")
print("  2. Self-learning system")
print("  3. Student-specific weight adaptation")
print("  4. Knowledge graph")
print("  5. Advanced concept extraction (AST parsing)")
print("  6. Reinforcement learning")
print("  7. OpenAI/LLM integration")
print("  8. Real-time feature extraction")
print("  9. Empathy calibration")
print("  10. Multi-level learning (student/global/empathy)")

print("\n📊 Current Algorithm:")
print("  - Frustration = count of keywords * 2.0")
print("  - Keywords: 'stuck', 'confused', 'help', etc.")
print("  - Concepts: Simple keyword matching")
print("  - Response: Fixed templates based on score ranges")

# Check what files exist
print("\n📁 Missing Implementation Files:")
ml_files = [
    "backend/app/services/ml/frustration_estimator.py",
    "backend/app/services/feedback/learning_engine.py",
    "backend/app/services/ml/concept_extractor.py",
    "backend/app/services/intervention/response_generator.py",
    "backend/app/services/intervention/knowledge_graph.py"
]

for file in ml_files:
    if not os.path.exists(file):
        print(f"  ❌ {file}")

print("\n" + "=" * 60)
