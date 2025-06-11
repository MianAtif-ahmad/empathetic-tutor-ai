# test_frustration.py - Test frustration detection with phrase support
def test_frustration_detection():
    message = "what the fuck is why isnt it print{} not working"
    message_lower = message.lower()
    
    print(f"Testing message: '{message}'")
    print(f"Lowercase: '{message_lower}'")
    
    frustration_score = 0.0
    
    # High impact words (weight 3.0)
    high_impact = ["fuck", "shit", "damn", "hell", "impossible", "hate", "stupid", "giving up", "quit", "worst", "terrible"]
    found_high = []
    for keyword in high_impact:
        if keyword in message_lower:
            frustration_score += 3.0
            found_high.append(keyword)
            print(f"   Found HIGH impact: '{keyword}' (+3.0)")
    
    # Medium impact words (weight 2.0)
    medium_impact = ["stuck", "frustrated", "angry", "annoying", "difficult", "hard", "broken"]
    found_medium = []
    for keyword in medium_impact:
        if keyword in message_lower:
            frustration_score += 2.0
            found_medium.append(keyword)
            print(f"   Found MEDIUM impact: '{keyword}' (+2.0)")
    
    # Low impact words (weight 1.0)
    low_impact = ["confused", "don't understand", "lost", "help", "struggling", "unclear"]
    found_low = []
    for keyword in low_impact:
        if keyword in message_lower:
            frustration_score += 1.0
            found_low.append(keyword)
            print(f"   Found LOW impact: '{keyword}' (+1.0)")
    
    # Error phrases (weight 1.5) - CHECK PHRASES FIRST
    error_phrases = ["not working", "doesn't work", "won't work", "isn't working", "doesn't working"]
    found_error_phrases = []
    for phrase in error_phrases:
        if phrase in message_lower:
            frustration_score += 1.5
            found_error_phrases.append(phrase)
            print(f"   Found ERROR phrase: '{phrase}' (+1.5)")
    
    # Error words (weight 1.5)
    error_words = ["error", "exception", "crash", "bug", "broken", "failed", "traceback"]
    found_error_words = []
    for word in error_words:
        if word in message_lower:
            frustration_score += 1.5
            found_error_words.append(word)
            print(f"   Found ERROR word: '{word}' (+1.5)")
    
    # Add punctuation analysis
    exclamation_count = message_lower.count('!')
    question_count = message_lower.count('?')
    caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)
    
    punct_score = 0.0
    punct_score += min(exclamation_count * 0.5, 2.0)
    punct_score += min(question_count * 0.3, 1.0)
    punct_score += caps_ratio * 3.0
    
    print(f"\nPunctuation analysis:")
    print(f"   Exclamations: {exclamation_count} (+{min(exclamation_count * 0.5, 2.0)})")
    print(f"   Questions: {question_count} (+{min(question_count * 0.3, 1.0)})")
    print(f"   Caps ratio: {caps_ratio:.2f} (+{caps_ratio * 3.0:.2f})")
    print(f"   Punctuation total: +{punct_score}")
    
    frustration_score += punct_score
    frustration_score = min(frustration_score, 10.0)
    
    print(f"\nDetection Summary:")
    print(f"   High impact words: {found_high}")
    print(f"   Medium impact words: {found_medium}")
    print(f"   Low impact words: {found_low}")
    print(f"   Error phrases: {found_error_phrases}")
    print(f"   Error words: {found_error_words}")
    print(f"   Final frustration score: {frustration_score}")
    
    # Determine empathy level
    if frustration_score > 7:
        empathy_level = "high"
        print("✅ HIGH frustration detected")
    elif frustration_score > 4:
        empathy_level = "medium"
        print("⚠️ MEDIUM frustration detected")
    else:
        empathy_level = "low"
        print("❌ LOW frustration detected")
    
    print(f"   Empathy level: {empathy_level}")
    
    # Test if "not working" is actually in the message
    print(f"\nPhrase detection test:")
    print(f"   Message contains 'not working': {'not working' in message_lower}")
    print(f"   Message contains 'isnt': {'isnt' in message_lower}")
    print(f"   Message contains 'working': {'working' in message_lower}")
    
    # Alternative detection for "isnt...working" pattern
    if "isnt" in message_lower and "working" in message_lower:
        print(f"   Found pattern 'isnt...working' - could add +1.5 for this!")
        frustration_score += 1.5
        print(f"   Updated score with pattern: {frustration_score}")

if __name__ == "__main__":
    test_frustration_detection()