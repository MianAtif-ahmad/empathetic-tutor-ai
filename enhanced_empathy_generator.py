# enhanced_empathy_generator.py - Final Version
from typing import Dict, List, Optional
import json
import re

class EnhancedEmpathyGenerator:
    """
    Generates highly empathetic, personalized prompts based on emotional state,
    learning style, and interaction history
    """
    
    def __init__(self):
        # Emotional state patterns
        self.emotional_patterns = {
            "overwhelmed": {
                "keywords": ["too much", "overwhelming", "can't handle", "too hard", "everything at once"],
                "empathy_approach": "break_down_and_simplify",
                "tone": "very_gentle",
                "priority": "high"
            },
            "discouraged": {
                "keywords": ["giving up", "hopeless", "never understand", "too stupid", "can't do this"],
                "empathy_approach": "rebuild_confidence",
                "tone": "encouraging_and_supportive",
                "priority": "high"
            },
            "impatient": {
                "keywords": ["quickly", "fast", "hurry", "just tell me", "don't have time"],
                "empathy_approach": "acknowledge_urgency",
                "tone": "efficient_but_caring",
                "priority": "medium"
            },
            "curious": {
                "keywords": ["why", "how does", "what happens", "interesting", "cool", "wonder"],
                "empathy_approach": "feed_curiosity",
                "tone": "enthusiastic",
                "priority": "low"
            },
            "analytical": {
                "keywords": ["logic", "understand the reason", "step by step", "methodology", "systematic"],
                "empathy_approach": "provide_deep_explanation",
                "tone": "detailed_and_systematic",
                "priority": "low"
            },
            "frustrated_technical": {
                "keywords": ["error", "bug", "doesn't work", "broken", "keeps failing"],
                "empathy_approach": "technical_problem_solving",
                "tone": "focused_and_helpful",
                "priority": "medium"
            }
        }
        
        # Learning style indicators
        self.learning_styles = {
            "visual": ["see", "show", "picture", "diagram", "visual", "chart", "graph"],
            "auditory": ["explain", "tell", "hear", "sounds like", "verbal", "talk through"],
            "kinesthetic": ["try", "practice", "hands-on", "do it myself", "build", "create"],
            "reading": ["read", "documentation", "text", "written", "article", "tutorial"]
        }
        
        # Confidence level indicators
        self.confidence_indicators = {
            "low": ["don't know", "confused", "lost", "no idea", "clueless"],
            "medium": ["think", "maybe", "not sure", "might be", "possibly"],
            "high": ["know", "understand", "figured out", "got it", "makes sense"]
        }
    
    def generate_empathetic_prompt(
        self,
        message: str,
        frustration_score: float,
        concepts: List[str],
        student_id: str,
        interaction_history: List[Dict] = None
    ) -> str:
        """
        Generate highly empathetic, personalized prompt
        """
        
        # Analyze student's emotional and learning state
        emotional_state = self._detect_emotional_state(message)
        learning_style = self._detect_learning_style(message, interaction_history)
        confidence_level = self._detect_confidence_level(message)
        frustration_pattern = self._analyze_frustration_pattern(interaction_history)
        
        # Build comprehensive empathy context
        empathy_context = self._build_empathy_context(
            message, frustration_score, emotional_state, 
            learning_style, confidence_level, frustration_pattern
        )
        
        # Get specific response guidelines
        guidelines = self._get_response_guidelines(
            emotional_state, learning_style, confidence_level, frustration_score
        )
        
        # Create the full prompt
        full_prompt = f"""{empathy_context}

=== STUDENT ANALYSIS ===
Student's message: "{message}"
Frustration level: {frustration_score}/10
Programming concepts: {', '.join(concepts) if concepts else 'general programming'}
Detected emotional state: {emotional_state['state']} (approach: {emotional_state['approach']})
Preferred learning style: {learning_style}
Confidence level: {confidence_level}
Frustration pattern: {frustration_pattern}

=== RESPONSE GUIDELINES ===
{guidelines}

=== YOUR TASK ===
Generate a response that demonstrates deep empathy and understanding. Your response should:
1. Address their emotional state first if needed
2. Adapt to their learning style
3. Provide concrete, actionable help
4. Build confidence while solving their problem
5. Feel genuinely caring and personalized

Response:"""
        
        return full_prompt
    
    def _detect_emotional_state(self, message: str) -> Dict:
        """Detect student's primary emotional state"""
        message_lower = message.lower()
        detected_states = []
        
        # Check for all emotional patterns
        for state, pattern in self.emotional_patterns.items():
            score = 0
            for keyword in pattern["keywords"]:
                if keyword in message_lower:
                    score += 1
            
            if score > 0:
                detected_states.append({
                    "state": state,
                    "score": score,
                    "approach": pattern["empathy_approach"],
                    "tone": pattern["tone"],
                    "priority": pattern["priority"]
                })
        
        if detected_states:
            # Sort by priority and score
            priority_order = {"high": 3, "medium": 2, "low": 1}
            detected_states.sort(key=lambda x: (priority_order[x["priority"]], x["score"]), reverse=True)
            return detected_states[0]
        
        # Default state
        return {
            "state": "neutral",
            "approach": "standard_support",
            "tone": "friendly",
            "priority": "low"
        }
    
    def _detect_learning_style(self, message: str, history: List[Dict] = None) -> str:
        """Detect preferred learning style from current message and history"""
        message_lower = message.lower()
        style_scores = {style: 0 for style in self.learning_styles.keys()}
        
        # Analyze current message
        for style, keywords in self.learning_styles.items():
            for keyword in keywords:
                if keyword in message_lower:
                    style_scores[style] += 1
        
        # Analyze interaction history for patterns
        if history:
            for interaction in history[-5:]:  # Last 5 interactions
                msg = interaction.get('message', '').lower()
                for style, keywords in self.learning_styles.items():
                    for keyword in keywords:
                        if keyword in msg:
                            style_scores[style] += 0.5  # Lower weight for history
        
        # Return style with highest score, or "mixed" if no clear preference
        max_score = max(style_scores.values())
        if max_score > 0:
            return max(style_scores, key=style_scores.get)
        
        return "mixed"
    
    def _detect_confidence_level(self, message: str) -> str:
        """Detect student's confidence level"""
        message_lower = message.lower()
        
        for level, indicators in self.confidence_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                return level
        
        return "medium"  # Default
    
    def _analyze_frustration_pattern(self, history: List[Dict] = None) -> str:
        """Analyze student's frustration patterns over time"""
        if not history or len(history) < 2:
            return "new_student"
        
        recent_scores = [h.get('frustration_score', 0) for h in history[-5:]]
        
        if len(recent_scores) < 2:
            return "insufficient_data"
        
        # Analyze trend and pattern
        avg_score = sum(recent_scores) / len(recent_scores)
        trend = recent_scores[-1] - recent_scores[0]
        
        if avg_score > 7:
            return "persistently_frustrated"
        elif trend > 2 and recent_scores[-1] > 6:
            return "escalating_frustration"
        elif avg_score < 3:
            return "generally_confident"
        elif trend < -2:
            return "improving"
        elif max(recent_scores) - min(recent_scores) > 4:
            return "highly_variable"
        else:
            return "stable"
    
    def _build_empathy_context(
        self, message: str, frustration_score: float, 
        emotional_state: Dict, learning_style: str, 
        confidence_level: str, frustration_pattern: str
    ) -> str:
        """Build personalized empathy context"""
        
        # Base empathy role
        context = "You are Sarah, an experienced and deeply empathetic programming tutor who has helped thousands of students overcome coding challenges. You genuinely care about each student's success and emotional well-being."
        
        # Add emotional state awareness
        if emotional_state["state"] == "overwhelmed":
            context += "\n\n🚨 CRITICAL: This student is feeling overwhelmed. They need you to break everything into tiny, digestible pieces. Be extremely patient and gentle. Consider suggesting they step back and breathe."
        
        elif emotional_state["state"] == "discouraged":
            context += "\n\n💔 IMPORTANT: This student is losing confidence. Your primary job is to rebuild their self-esteem. Focus on what they're doing RIGHT before addressing what needs fixing. Use lots of encouragement."
        
        elif emotional_state["state"] == "impatient":
            context += "\n\n⏰ NOTE: This student seems pressed for time or impatient. Acknowledge their urgency, provide immediate actionable help, then explain why it works."
        
        elif emotional_state["state"] == "curious":
            context += "\n\n✨ OPPORTUNITY: This student is showing genuine curiosity! This is wonderful - feed their interest with engaging explanations and maybe some cool facts or advanced concepts."
        
        elif emotional_state["state"] == "frustrated_technical":
            context += "\n\n🔧 FOCUS: This student has a specific technical problem causing frustration. Address the immediate issue first, then explain the underlying concepts."
        
        # Add learning style adaptation
        if learning_style == "visual":
            context += "\n\n👁️ LEARNING STYLE: Visual learner detected. Suggest diagrams, flowcharts, or visual representations. Use spatial language like 'above', 'below', 'flows from'."
        
        elif learning_style == "auditory":
            context += "\n\n🎧 LEARNING STYLE: Auditory learner detected. Use rich analogies, metaphors, and step-by-step verbal explanations. Think 'explain it like I'm talking to a friend'."
        
        elif learning_style == "kinesthetic":
            context += "\n\n🖐️ LEARNING STYLE: Hands-on learner detected. Suggest interactive exercises, building something together, or 'let's try this' approaches. Use action-oriented language."
        
        elif learning_style == "reading":
            context += "\n\n📚 LEARNING STYLE: Reading/research learner detected. Suggest documentation, articles, or written resources. Provide structured, detailed explanations."
        
        # Add confidence level awareness
        if confidence_level == "low":
            context += "\n\n🤗 CONFIDENCE: Low confidence detected. Be extra encouraging. Start with simple wins. Use phrases like 'You're on the right track' and 'This is totally normal'."
        
        elif confidence_level == "high":
            context += "\n\n🚀 CONFIDENCE: High confidence detected. You can be more direct and even challenge them with slightly advanced concepts. They can handle it!"
        
        # Add frustration pattern awareness
        if frustration_pattern == "persistently_frustrated":
            context += "\n\n⚠️ PATTERN ALERT: This student has been struggling consistently across multiple sessions. Consider suggesting a break, different learning resources, or a completely different approach."
        
        elif frustration_pattern == "escalating_frustration":
            context += "\n\n🚨 ESCALATION ALERT: This student's frustration is building up over time. This needs immediate gentle intervention. Address emotions before technical content."
        
        elif frustration_pattern == "improving":
            context += "\n\n📈 POSITIVE TREND: This student is making progress! Acknowledge their improvement and keep them motivated. They're on the right path."
        
        elif frustration_pattern == "highly_variable":
            context += "\n\n🎢 VARIABLE PATTERN: This student has ups and downs. Be extra attentive to their current emotional state and adapt accordingly."
        
        return context
    
    def _get_response_guidelines(
        self, emotional_state: Dict, learning_style: str, 
        confidence_level: str, frustration_score: float
    ) -> str:
        """Get specific, actionable response guidelines"""
        
        guidelines = []
        
        # Emotional state specific guidelines
        if emotional_state["approach"] == "break_down_and_simplify":
            guidelines.extend([
                "- Start with: 'Let's slow down and take this one tiny step at a time'",
                "- Break ANY explanation into maximum 3 simple steps",
                "- Ask: 'Should we just focus on one small piece first?'",
                "- Use calming language: 'There's no rush, we'll figure this out together'"
            ])
        
        elif emotional_state["approach"] == "rebuild_confidence":
            guidelines.extend([
                "- Start with validation: 'This is genuinely challenging, and your struggle is completely normal'",
                "- Find something they did right: 'I notice you [specific positive thing]'",
                "- Use confidence-building phrases: 'You're closer than you think', 'You have the right instincts'",
                "- Share that others struggle with this too: 'Even experienced programmers find this tricky'"
            ])
        
        elif emotional_state["approach"] == "acknowledge_urgency":
            guidelines.extend([
                "- Acknowledge time pressure: 'I can see you need this figured out quickly'",
                "- Give immediate actionable step: 'Here's what to try right now...'",
                "- Explain briefly: 'The reason this works is...'",
                "- Offer deeper explanation: 'Want me to explain more, or does this solve your immediate need?'"
            ])
        
        elif emotional_state["approach"] == "feed_curiosity":
            guidelines.extend([
                "- Match their enthusiasm: 'Great question! This is actually really interesting...'",
                "- Give the full picture: Don't just solve the problem, explain why it's cool",
                "- Add bonus info: 'Fun fact: this concept is also used in...'",
                "- Encourage exploration: 'Want to see what happens if we...?'"
            ])
        
        elif emotional_state["approach"] == "technical_problem_solving":
            guidelines.extend([
                "- Address the error immediately: 'Let's fix this specific issue first'",
                "- Provide the solution clearly: 'Try changing line X to...'",
                "- Explain why it happened: 'This error occurs because...'",
                "- Prevent future issues: 'To avoid this next time...'"
            ])
        
        # Learning style specific guidelines
        if learning_style == "visual":
            guidelines.extend([
                "- Suggest: 'Would it help to draw this out?' or 'Let me paint a picture...'",
                "- Use visual metaphors: 'Think of it like a flowchart where...'",
                "- Recommend: 'Try sketching the flow of your program'"
            ])
        
        elif learning_style == "kinesthetic":
            guidelines.extend([
                "- Suggest hands-on action: 'Let's build this step by step together'",
                "- Use action language: 'Try this', 'experiment with', 'play around with'",
                "- Encourage experimentation: 'What happens if you change this and run it?'"
            ])
        
        elif learning_style == "auditory":
            guidelines.extend([
                "- Use rich analogies: 'Think of it like...' or 'It's similar to...'",
                "- Explain step-by-step verbally: 'First we do this, then we do that, and finally...'",
                "- Use conversational tone: 'So basically what's happening here is...'"
            ])
        
        # Frustration level specific guidelines
        if frustration_score > 8:
            guidelines.extend([
                "- Address emotions FIRST: 'I can really hear how frustrating this is'",
                "- Suggest a micro-break: 'Want to take a 30-second breather with me?'",
                "- Normalize the struggle: 'You know what? This particular concept trips up most people'",
                "- Focus on tiny wins: 'Let's get just one small thing working, then build from there'"
            ])
        
        elif frustration_score < 3:
            guidelines.extend([
                "- You can be more technical and detailed",
                "- Challenge them slightly: 'Since you've got this, want to try something a bit more advanced?'",
                "- Provide comprehensive explanations",
                "- Suggest exploration: 'This opens up some interesting possibilities...'"
            ])
        
        # Confidence level guidelines
        if confidence_level == "low":
            guidelines.extend([
                "- Use lots of reassurance: 'You're asking exactly the right questions'",
                "- Celebrate small progress: 'See? You totally got that part!'",
                "- Normalize confusion: 'Everyone finds this confusing at first'"
            ])
        
        return "\n".join(guidelines) if guidelines else "- Provide helpful, empathetic guidance tailored to this student"

# Test function
def test_enhanced_empathy():
    """Test the enhanced empathy generator"""
    generator = EnhancedEmpathyGenerator()
    
    test_case = {
        "message": "I've been trying to understand functions for hours and I still don't get it! This is impossible! I'm so stupid.",
        "frustration_score": 8.5,
        "concepts": ["functions"],
        "student_id": "student_123",
        "interaction_history": [
            {"message": "I'm confused about variables", "frustration_score": 4.0},
            {"message": "Still don't get variables", "frustration_score": 6.0},
            {"message": "Now I'm stuck on functions too", "frustration_score": 7.5}
        ]
    }
    
    prompt = generator.generate_empathetic_prompt(**test_case)
    
    print("🧪 ENHANCED EMPATHY PROMPT TEST:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

if __name__ == "__main__":
    test_enhanced_empathy()