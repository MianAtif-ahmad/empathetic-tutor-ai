#!/usr/bin/env python3
# setup_ml_system.py - Setup script for your ML learning system

import os
import sys
import subprocess
import sqlite3
import json
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'scikit-learn>=1.0.0',
        'numpy>=1.21.0', 
        'scipy>=1.7.0'
    ]
    
    print("📦 Checking required packages...")
    
    missing_packages = []
    for package in required_packages:
        try:
            package_name = package.split('>=')[0]
            __import__(package_name.replace('-', '_'))
            print(f"  ✅ {package_name}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package_name}")
    
    if missing_packages:
        print(f"\n🔧 Installing missing packages...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"  ✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {package}: {e}")
                return False
    
    return True

def setup_database_tables(db_path: str):
    """Create ML learning tables in your existing database"""
    print(f"🗃️ Setting up ML tables in {db_path}...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Learning feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_feedback (
                id TEXT PRIMARY KEY,
                interaction_id TEXT,
                student_id TEXT,
                prediction_error REAL,
                actual_frustration REAL,
                predicted_frustration REAL,
                feature_importance TEXT,
                response_helpful INTEGER,
                empathy_appropriate INTEGER,
                response_time_seconds REAL,
                follow_up_needed INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        """)
        print("  ✅ learning_feedback table")
        
        # Discovered patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discovered_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT,
                pattern_text TEXT,
                concept_category TEXT,
                weight REAL,
                confidence_score REAL,
                usage_count INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.5,
                status TEXT DEFAULT 'candidate',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                approved_at TIMESTAMP,
                last_used TIMESTAMP
            )
        """)
        print("  ✅ discovered_patterns table")
        
        # Student learning profiles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student_learning_profiles (
                student_id TEXT PRIMARY KEY,
                personalized_weights TEXT,
                learning_style TEXT,
                adaptation_rate REAL DEFAULT 0.01,
                total_interactions INTEGER DEFAULT 0,
                avg_prediction_error REAL DEFAULT 0.0,
                last_weight_update TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  ✅ student_learning_profiles table")
        
        # Global weights history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_weights (
                id TEXT PRIMARY KEY,
                weights TEXT,
                version INTEGER,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("  ✅ global_weights table")
        
        # Pattern performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_performance (
                id TEXT PRIMARY KEY,
                pattern_id TEXT,
                usage_date DATE,
                prediction_improvement REAL,
                student_satisfaction REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pattern_id) REFERENCES discovered_patterns (id)
            )
        """)
        print("  ✅ pattern_performance table")
        
        # Add ML columns to existing interactions table
        try:
            cursor.execute("ALTER TABLE interactions ADD COLUMN ml_features TEXT")
            print("  ✅ Added ml_features column to interactions")
        except sqlite3.OperationalError:
            print("  ℹ️ ml_features column already exists")
        
        try:
            cursor.execute("ALTER TABLE interactions ADD COLUMN confidence_score REAL")
            print("  ✅ Added confidence_score column to interactions")
        except sqlite3.OperationalError:
            print("  ℹ️ confidence_score column already exists")
        
        conn.commit()
        conn.close()
        print("✅ Database setup complete")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_backup_directory(config_dir: str):
    """Create backup directory for config files"""
    backup_dir = Path(config_dir) / "backups"
    backup_dir.mkdir(exist_ok=True)
    print(f"✅ Created backup directory: {backup_dir}")
    return str(backup_dir)

def verify_config_files(config_dir: str):
    """Verify your existing config files"""
    config_path = Path(config_dir)
    required_files = [
        "emotional_patterns.json",
        "keyword_weights.json",
        "concept_keywords.json", 
        "detection_settings.json"
    ]
    
    print(f"📁 Verifying config files in {config_dir}...")
    
    all_exist = True
    for file in required_files:
        file_path = config_path / file
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    json.load(f)  # Validate JSON
                print(f"  ✅ {file}")
            except json.JSONDecodeError as e:
                print(f"  ❌ {file} - Invalid JSON: {e}")
                all_exist = False
        else:
            print(f"  ❌ {file} - Not found")
            all_exist = False
    
    return all_exist

def create_test_script():
    """Create a test script to verify ML system"""
    test_content = '''#!/usr/bin/env python3
# test_ml_system.py - Test your ML learning system

import asyncio
import json
from ml_learning_system import HybridFrustrationAnalyzer
from frustration_analyzer import frustration_analyzer

async def test_ml_system():
    """Test the ML learning system"""
    print("🧪 Testing ML Learning System")
    print("=" * 50)
    
    # Initialize ML system
    try:
        ml_analyzer = HybridFrustrationAnalyzer("empathetic_tutor.db", frustration_analyzer)
        print("✅ ML analyzer initialized")
    except Exception as e:
        print(f"❌ ML analyzer failed: {e}")
        return
    
    # Test cases
    test_messages = [
        {
            "message": "this code is driving me absolutely crazy and nothing works!",
            "student_id": "test_student_1",
            "expected_high_frustration": True
        },
        {
            "message": "can you help me understand how loops work?",
            "student_id": "test_student_2", 
            "expected_high_frustration": False
        },
        {
            "message": "I feel so stupid, this syntax error makes no sense",
            "student_id": "test_student_3",
            "expected_high_frustration": True
        }
    ]
    
    for i, test in enumerate(test_messages, 1):
        print(f"\\n--- Test {i}: {test['message'][:30]}... ---")
        
        try:
            # Test ML analysis
            result = ml_analyzer.analyze_with_learning(test["message"], test["student_id"])
            
            print(f"Frustration Score: {result['frustration_score']:.2f}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Rule Contribution: {result['rule_contribution']:.2f}")
            print(f"ML Contribution: {result['ml_contribution']:.2f}")
            print(f"Learning Opportunity: {result['learning_opportunity']}")
            
            # Validate expectations
            is_high_frustration = result['frustration_score'] > 6.0
            if is_high_frustration == test['expected_high_frustration']:
                print("✅ Expectation met")
            else:
                print("⚠️ Unexpected result")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\\n🎯 ML system test completed")

if __name__ == "__main__":
    asyncio.run(test_ml_system())
'''
    
    with open("test_ml_system.py", "w") as f:
        f.write(test_content)
    
    print("✅ Created test_ml_system.py")

def main():
    """Main setup function"""
    print("🚀 Setting up ML Learning System for Empathetic Tutor")
    print("=" * 60)
    
    # Configuration
    PROJECT_ROOT = "/Users/atif/empathetic-tutor-ai"
    CONFIG_DIR = f"{PROJECT_ROOT}/config"
    DB_PATH = f"{PROJECT_ROOT}/empathetic_tutor.db"
    
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Config directory: {CONFIG_DIR}")
    print(f"🗃️ Database: {DB_PATH}")
    
    # Step 1: Check requirements
    if not check_requirements():
        print("❌ Setup failed: Missing required packages")
        return False
    
    # Step 2: Verify config files
    if not verify_config_files(CONFIG_DIR):
        print("❌ Setup failed: Config files missing or invalid")
        return False
    
    # Step 3: Setup database
    if not setup_database_tables(DB_PATH):
        print("❌ Setup failed: Database setup failed")
        return False
    
    # Step 4: Create backup directory
    create_backup_directory(CONFIG_DIR)
    
    # Step 5: Create test script
    create_test_script()
    
    print("\n🎉 ML Learning System Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Copy the ML system files to your project:")
    print("   - ml_learning_system.py")
    print("   - ml_api_integration.py") 
    print("   - auto_config_updater.py")
    print("\n2. Update your simple_api.py with the enhanced version")
    print("\n3. Test the system:")
    print("   python test_ml_system.py")
    print("\n4. Start your enhanced API:")
    print("   python simple_api.py")
    print("\n5. Test ML endpoints:")
    print("   curl -X POST http://localhost:8000/analyze/ml \\")
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"message": "this is so frustrating!", "student_id": "test"}\'')
    print("\n📚 New API endpoints available:")
    print("   - POST /analyze/ml - Enhanced ML analysis")
    print("   - POST /feedback/detailed - Rich feedback collection")
    print("   - GET /learning/patterns - View discovered patterns")
    print("   - POST /learning/discover - Trigger pattern discovery")
    print("   - GET /ml/status - ML system status")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)