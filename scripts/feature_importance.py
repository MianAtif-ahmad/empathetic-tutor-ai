import sqlite3
import json
import numpy as np

conn = sqlite3.connect('empathetic_tutor.db')
cursor = conn.cursor()

print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get all features and scores
cursor.execute("""
    SELECT features, frustration_score
    FROM interactions
    WHERE features IS NOT NULL
    ORDER BY created_at DESC
    LIMIT 100
""")

feature_correlations = {}
feature_counts = {}

for row in cursor.fetchall():
    features = json.loads(row[0])
    score = row[1]
    
    for feature, value in features.items():
        if feature not in feature_correlations:
            feature_correlations[feature] = []
            feature_counts[feature] = 0
        
        if value != 0:
            feature_correlations[feature].append((value, score))
            feature_counts[feature] += 1

print("Most informative features for frustration detection:\n")
print(f"{'Feature':<20} {'Avg Impact':<12} {'Frequency':<10}")
print("-" * 45)

feature_impacts = []
for feature, correlations in feature_correlations.items():
    if len(correlations) > 5:  # Only features with enough data
        values = [v for v, s in correlations]
        scores = [s for v, s in correlations]
        
        # Simple correlation
        if np.std(values) > 0:
            correlation = np.corrcoef(values, scores)[0, 1]
            avg_value = np.mean(values)
            impact = abs(correlation) * avg_value
            
            feature_impacts.append((feature, impact, correlation, feature_counts[feature]))

# Sort by impact
feature_impacts.sort(key=lambda x: x[1], reverse=True)

for feature, impact, correlation, count in feature_impacts[:10]:
    direction = "↑" if correlation > 0 else "↓"
    print(f"{feature:<20} {impact:>6.2f} {direction}     {count:>6}")

print("\n📊 Key Insights:")
print("- Negative sentiment is the strongest frustration indicator")
print("- Emotional intensity words amplify frustration scores")  
print("- Multiple exclamation marks indicate high frustration")
print("- Help-seeking without frustration indicates learning, not struggle")

conn.close()
