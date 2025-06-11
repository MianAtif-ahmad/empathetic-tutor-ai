# auto_config_updater.py - Automatic Configuration Learning & Updates
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
import shutil
from collections import defaultdict
import numpy as np

class AutoConfigUpdater:
    """
    Automatically updates JSON configuration files based on ML learning insights
    """
    
    def __init__(self, config_dir: str, db_path: str):
        self.config_dir = Path(config_dir)
        self.db_path = db_path
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Configuration update thresholds
        self.min_usage_count = 10  # Minimum times pattern must be used
        self.min_effectiveness = 0.7  # Minimum effectiveness score
        self.confidence_threshold = 0.8  # Minimum confidence for auto-update
        
    def auto_update_configurations(self, review_period_days: int = 7) -> Dict:
        """
        Automatically update configurations based on learning data
        """
        print(f"🔄 Starting auto-configuration update (reviewing last {review_period_days} days)")
        
        # Backup current configurations
        backup_id = self._create_backup()
        
        try:
            # Get approved patterns ready for activation
            patterns_to_activate = self._get_patterns_for_activation()
            
            # Update emotional patterns
            emotional_updates = self._update_emotional_patterns(patterns_to_activate)
            
            # Update keyword weights
            keyword_updates = self._update_keyword_weights(patterns_to_activate)
            
            # Update concept keywords
            concept_updates = self._update_concept_keywords(patterns_to_activate)
            
            # Update detection settings based on performance
            settings_updates = self._update_detection_settings(review_period_days)
            
            # Remove obsolete patterns
            obsolete_removals = self._remove_obsolete_patterns()
            
            # Mark patterns as active in database
            self._mark_patterns_as_active(patterns_to_activate)
            
            update_summary = {
                "backup_id": backup_id,
                "emotional_patterns": emotional_updates,
                "keyword_weights": keyword_updates,
                "concept_keywords": concept_updates,
                "settings_updates": settings_updates,
                "obsolete_removals": obsolete_removals,
                "total_changes": sum([
                    len(emotional_updates),
                    len(keyword_updates), 
                    len(concept_updates),
                    len(settings_updates),
                    len(obsolete_removals)
                ])
            }
            
            print(f"✅ Auto-update completed: {update_summary['total_changes']} changes made")
            return update_summary
            
        except Exception as e:
            print(f"❌ Auto-update failed: {e}")
            self._restore_backup(backup_id)
            return {"error": str(e), "backup_restored": True}
    
    def _create_backup(self) -> str:
        """Create backup of current configuration files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_id = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_id
        backup_path.mkdir(exist_ok=True)
        
        config_files = [
            "emotional_patterns.json",
            "keyword_weights.json", 
            "concept_keywords.json",
            "detection_settings.json"
        ]
        
        for file in config_files:
            source = self.config_dir / file
            if source.exists():
                shutil.copy2(source, backup_path / file)
        
        print(f"📁 Configuration backup created: {backup_id}")
        return backup_id
    
    def _restore_backup(self, backup_id: str):
        """Restore configuration from backup"""
        backup_path = self.backup_dir / backup_id
        if not backup_path.exists():
            print(f"❌ Backup {backup_id} not found")
            return
        
        for backup_file in backup_path.glob("*.json"):
            target = self.config_dir / backup_file.name
            shutil.copy2(backup_file, target)
        
        print(f"🔄 Configuration restored from backup: {backup_id}")
    
    def _get_patterns_for_activation(self) -> List[Dict]:
        """Get approved patterns that meet activation criteria"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT dp.*, 
                       COALESCE(AVG(pp.prediction_improvement), 0) as avg_improvement,
                       COALESCE(AVG(pp.student_satisfaction), 0) as avg_satisfaction
                FROM discovered_patterns dp
                LEFT JOIN pattern_performance pp ON dp.id = pp.pattern_id
                WHERE dp.status = 'approved' 
                  AND dp.usage_count >= ?
                  AND dp.confidence_score >= ?
                GROUP BY dp.id
                HAVING avg_improvement >= ? OR dp.confidence_score >= ?
            """, (
                self.min_usage_count,
                self.confidence_threshold,
                self.min_effectiveness,
                0.9  # Very high confidence patterns can bypass effectiveness requirement
            ))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    "id": row[0],
                    "type": row[1],
                    "pattern": row[2],
                    "concept_category": row[3],
                    "weight": row[4],
                    "confidence": row[5],
                    "usage_count": row[6],
                    "effectiveness": row[7],
                    "avg_improvement": row[9] if len(row) > 9 else 0,
                    "avg_satisfaction": row[10] if len(row) > 10 else 0
                })
            
            conn.close()
            return patterns
            
        except Exception as e:
            print(f"Error getting patterns for activation: {e}")
            return []
    
    def _update_emotional_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Update emotional_patterns.json with new patterns"""
        emotional_patterns = [p for p in patterns if p["type"] == "emotional_pattern"]
        if not emotional_patterns:
            return []
        
        # Load current patterns
        config_file = self.config_dir / "emotional_patterns.json"
        try:
            with open(config_file, 'r') as f:
                current_patterns = json.load(f)
        except:
            current_patterns = {}
        
        updates = []
        for pattern in emotional_patterns:
            pattern_text = pattern["pattern"]
            new_weight = self._calculate_optimized_weight(pattern)
            
            if pattern_text not in current_patterns or current_patterns[pattern_text] != new_weight:
                current_patterns[pattern_text] = new_weight
                updates.append({
                    "pattern": pattern_text,
                    "weight": new_weight,
                    "confidence": pattern["confidence"],
                    "action": "added" if pattern_text not in current_patterns else "updated"
                })
        
        # Save updated patterns
        if updates:
            with open(config_file, 'w') as f:
                json.dump(current_patterns, f, indent=2)
            print(f"✅ Updated {len(updates)} emotional patterns")
        
        return updates
    
    def _update_keyword_weights(self, patterns: List[Dict]) -> List[Dict]:
        """Update keyword_weights.json with new keyword patterns"""
        keyword_patterns = [p for p in patterns if p["type"] == "keyword"]
        if not keyword_patterns:
            return []
        
        # Load current keyword weights
        config_file = self.config_dir / "keyword_weights.json"
        try:
            with open(config_file, 'r') as f:
                current_weights = json.load(f)
        except:
            current_weights = {}
        
        updates = []
        
        # Group new keywords by weight range to determine category
        for pattern in keyword_patterns:
            keyword = pattern["pattern"]
            weight = pattern["weight"]
            confidence = pattern["confidence"]
            
            # Determine appropriate category based on weight
            if weight >= 2.5:
                category = "high_impact"
            elif weight >= 1.5:
                category = "medium_impact"
            elif weight >= 0.8:
                category = "mild_impact"
            else:
                category = "error_words"
            
            # Ensure category exists
            if category not in current_weights:
                current_weights[category] = {
                    "weight": 2.0 if category == "high_impact" else 1.5 if category == "medium_impact" else 1.0,
                    "keywords": []
                }
            
            # Add keyword if not already present
            if keyword not in current_weights[category]["keywords"]:
                current_weights[category]["keywords"].append(keyword)
                updates.append({
                    "keyword": keyword,
                    "category": category,
                    "weight": weight,
                    "confidence": confidence,
                    "action": "added"
                })
            
            # Update category weight if this pattern suggests a different weight
            category_weight = current_weights[category]["weight"]
            suggested_weight = self._calculate_category_weight(category, weight, confidence)
            if abs(category_weight - suggested_weight) > 0.2:
                current_weights[category]["weight"] = suggested_weight
                updates.append({
                    "category": category,
                    "old_weight": category_weight,
                    "new_weight": suggested_weight,
                    "action": "weight_updated"
                })
        
        # Save updated keyword weights
        if updates:
            with open(config_file, 'w') as f:
                json.dump(current_weights, f, indent=2)
            print(f"✅ Updated {len(updates)} keyword weight entries")
        
        return updates
    
    def _update_concept_keywords(self, patterns: List[Dict]) -> List[Dict]:
        """Update concept_keywords.json with new programming concept keywords"""
        concept_patterns = [p for p in patterns if p["type"] == "concept_keyword"]
        if not concept_patterns:
            return []
        
        # Load current concept keywords
        config_file = self.config_dir / "concept_keywords.json"
        try:
            with open(config_file, 'r') as f:
                current_concepts = json.load(f)
        except:
            current_concepts = {}
        
        updates = []
        
        for pattern in concept_patterns:
            concept = pattern["concept_category"]
            keyword = pattern["pattern"]
            confidence = pattern["confidence"]
            
            # Ensure concept category exists
            if concept not in current_concepts:
                current_concepts[concept] = []
            
            # Add keyword if not already present
            if keyword not in current_concepts[concept]:
                current_concepts[concept].append(keyword)
                updates.append({
                    "concept": concept,
                    "keyword": keyword,
                    "confidence": confidence,
                    "action": "added"
                })
        
        # Save updated concept keywords
        if updates:
            with open(config_file, 'w') as f:
                json.dump(current_concepts, f, indent=2)
            print(f"✅ Updated {len(updates)} concept keywords")
        
        return updates
    
    def _update_detection_settings(self, review_period_days: int) -> List[Dict]:
        """Update detection_settings.json based on performance analytics"""
        updates = []
        
        # Analyze prediction accuracy over review period
        accuracy_stats = self._analyze_prediction_accuracy(review_period_days)
        
        # Load current settings
        config_file = self.config_dir / "detection_settings.json"
        try:
            with open(config_file, 'r') as f:
                current_settings = json.load(f)
        except:
            return updates
        
        # Update empathy thresholds if accuracy suggests adjustments
        if accuracy_stats.get("avg_error", 0) > 1.5:  # High prediction error
            empathy_thresholds = current_settings.get("empathy_thresholds", {})
            
            # Adjust thresholds to be more sensitive
            if "medium" in empathy_thresholds:
                old_medium = empathy_thresholds["medium"]
                new_medium = max(3.0, old_medium - 0.5)
                if old_medium != new_medium:
                    empathy_thresholds["medium"] = new_medium
                    updates.append({
                        "setting": "empathy_thresholds.medium",
                        "old_value": old_medium,
                        "new_value": new_medium,
                        "reason": "high_prediction_error"
                    })
            
            if "high" in empathy_thresholds:
                old_high = empathy_thresholds["high"]
                new_high = max(6.0, old_high - 0.5)
                if old_high != new_high:
                    empathy_thresholds["high"] = new_high
                    updates.append({
                        "setting": "empathy_thresholds.high",
                        "old_value": old_high,
                        "new_value": new_high,
                        "reason": "high_prediction_error"
                    })
        
        # Save updated settings
        if updates:
            with open(config_file, 'w') as f:
                json.dump(current_settings, f, indent=2)
            print(f"✅ Updated {len(updates)} detection settings")
        
        return updates
    
    def _remove_obsolete_patterns(self) -> List[Dict]:
        """Remove patterns that are no longer effective"""
        removals = []
        
        # Get patterns with low effectiveness
        obsolete_patterns = self._identify_obsolete_patterns()
        
        if not obsolete_patterns:
            return removals
        
        # Remove from emotional patterns
        emotional_file = self.config_dir / "emotional_patterns.json"
        if emotional_file.exists():
            with open(emotional_file, 'r') as f:
                emotional_patterns = json.load(f)
            
            for pattern in obsolete_patterns:
                if pattern["type"] == "emotional_pattern" and pattern["pattern"] in emotional_patterns:
                    del emotional_patterns[pattern["pattern"]]
                    removals.append({
                        "type": "emotional_pattern",
                        "pattern": pattern["pattern"],
                        "reason": "low_effectiveness"
                    })
            
            if any(r["type"] == "emotional_pattern" for r in removals):
                with open(emotional_file, 'w') as f:
                    json.dump(emotional_patterns, f, indent=2)
        
        # Remove from keyword weights
        keyword_file = self.config_dir / "keyword_weights.json"
        if keyword_file.exists():
            with open(keyword_file, 'r') as f:
                keyword_weights = json.load(f)
            
            for pattern in obsolete_patterns:
                if pattern["type"] == "keyword":
                    keyword = pattern["pattern"]
                    for category, data in keyword_weights.items():
                        if keyword in data.get("keywords", []):
                            data["keywords"].remove(keyword)
                            removals.append({
                                "type": "keyword",
                                "pattern": keyword,
                                "category": category,
                                "reason": "low_effectiveness"
                            })
                            break
            
            if any(r["type"] == "keyword" for r in removals):
                with open(keyword_file, 'w') as f:
                    json.dump(keyword_weights, f, indent=2)
        
        # Mark as deprecated in database
        if obsolete_patterns:
            self._mark_patterns_as_deprecated([p["id"] for p in obsolete_patterns])
        
        if removals:
            print(f"🗑️ Removed {len(removals)} obsolete patterns")
        
        return removals
    
    def _calculate_optimized_weight(self, pattern: Dict) -> float:
        """Calculate optimized weight for a pattern based on performance"""
        base_weight = pattern["weight"]
        confidence = pattern["confidence"]
        effectiveness = pattern.get("avg_improvement", 0.5)
        
        # Adjust weight based on performance
        adjustment_factor = (confidence * 0.7 + effectiveness * 0.3)
        optimized_weight = base_weight * adjustment_factor
        
        # Keep within reasonable bounds
        return round(min(max(optimized_weight, 0.5), 5.0), 1)
    
    def _calculate_category_weight(self, category: str, pattern_weight: float, confidence: float) -> float:
        """Calculate appropriate category weight"""
        base_weights = {
            "high_impact": 3.0,
            "medium_impact": 2.0,
            "mild_impact": 1.0,
            "error_words": 1.5
        }
        
        base = base_weights.get(category, 1.0)
        
        # Adjust based on pattern performance
        if confidence > 0.8:
            return round(min(base * 1.1, 4.0), 1)
        elif confidence < 0.6:
            return round(max(base * 0.9, 0.5), 1)
        
        return base
    
    def _analyze_prediction_accuracy(self, days: int) -> Dict:
        """Analyze prediction accuracy over time period"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT AVG(ABS(prediction_error)) as avg_error,
                       COUNT(*) as total_predictions,
                       AVG(response_helpful) as avg_helpful_rate
                FROM learning_feedback
                WHERE created_at > ?
            """, (cutoff_date,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[1] > 0:
                return {
                    "avg_error": row[0] or 0,
                    "total_predictions": row[1],
                    "avg_helpful_rate": row[2] or 0
                }
            
        except Exception as e:
            print(f"Error analyzing prediction accuracy: {e}")
        
        return {"avg_error": 0, "total_predictions": 0, "avg_helpful_rate": 0}
    
    def _identify_obsolete_patterns(self) -> List[Dict]:
        """Identify patterns that should be removed due to poor performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find patterns with low effectiveness and high usage
            cursor.execute("""
                SELECT dp.*, 
                       COALESCE(AVG(pp.prediction_improvement), 0) as avg_improvement,
                       COALESCE(AVG(pp.student_satisfaction), 0) as avg_satisfaction
                FROM discovered_patterns dp
                LEFT JOIN pattern_performance pp ON dp.id = pp.pattern_id
                WHERE dp.status = 'active' 
                  AND dp.usage_count >= 20  -- Only consider well-tested patterns
                GROUP BY dp.id
                HAVING avg_improvement < 0.2 OR avg_satisfaction < 0.3
            """)
            
            obsolete = []
            for row in cursor.fetchall():
                obsolete.append({
                    "id": row[0],
                    "type": row[1],
                    "pattern": row[2],
                    "avg_improvement": row[9] if len(row) > 9 else 0,
                    "avg_satisfaction": row[10] if len(row) > 10 else 0
                })
            
            conn.close()
            return obsolete
            
        except Exception as e:
            print(f"Error identifying obsolete patterns: {e}")
            return []
    
    def _mark_patterns_as_active(self, patterns: List[Dict]):
        """Mark approved patterns as active in database"""
        if not patterns:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            pattern_ids = [p["id"] for p in patterns]
            placeholders = ",".join(["?" for _ in pattern_ids])
            
            cursor.execute(f"""
                UPDATE discovered_patterns 
                SET status = 'active', last_used = ?
                WHERE id IN ({placeholders})
            """, [datetime.now()] + pattern_ids)
            
            conn.commit()
            conn.close()
            
            print(f"✅ Marked {len(patterns)} patterns as active")
            
        except Exception as e:
            print(f"Error marking patterns as active: {e}")
    
    def _mark_patterns_as_deprecated(self, pattern_ids: List[str]):
        """Mark patterns as deprecated in database"""
        if not pattern_ids:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ",".join(["?" for _ in pattern_ids])
            
            cursor.execute(f"""
                UPDATE discovered_patterns 
                SET status = 'deprecated'
                WHERE id IN ({placeholders})
            """, pattern_ids)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error marking patterns as deprecated: {e}")
    
    def get_update_history(self, days: int = 30) -> List[Dict]:
        """Get history of configuration updates"""
        update_history = []
        
        # Look through backup directory for recent updates
        for backup_dir in self.backup_dir.glob("backup_*"):
            try:
                backup_date = datetime.strptime(backup_dir.name.split("_")[1], "%Y%m%d")
                if (datetime.now() - backup_date).days <= days:
                    update_history.append({
                        "backup_id": backup_dir.name,
                        "date": backup_date.isoformat(),
                        "type": "configuration_update"
                    })
            except:
                continue
        
        return sorted(update_history, key=lambda x: x["date"], reverse=True)
    
    def schedule_auto_update(self, interval_hours: int = 24):
        """Schedule automatic configuration updates"""
        # This would integrate with a task scheduler
        # For now, just return the configuration
        return {
            "interval_hours": interval_hours,
            "next_update": (datetime.now() + timedelta(hours=interval_hours)).isoformat(),
            "auto_update_enabled": True
        }

# Integration function for the main system
def setup_auto_config_updates(config_dir: str, db_path: str) -> AutoConfigUpdater:
    """Setup automatic configuration updates"""
    updater = AutoConfigUpdater(config_dir, db_path)
    print("✅ Auto-configuration updater initialized")
    return updater

# Usage example
if __name__ == "__main__":
    updater = AutoConfigUpdater("config", "empathetic_tutor.db")
    result = updater.auto_update_configurations(review_period_days=7)
    print("Update result:", result)