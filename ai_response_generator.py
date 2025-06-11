#ai_response_generator.py - Clean Version with Fixed Indentation
import os
import asyncio
import sqlite3
from typing import Dict, List, Optional
import openai
import anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
try:
    from config_manager import config
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️  config_manager not available, using defaults")
    CONFIG_AVAILABLE = False

try:
    from enhanced_empathy_generator import EnhancedEmpathyGenerator
    ENHANCED_EMPATHY_AVAILABLE = True
except ImportError:
    print("⚠️  enhanced_empathy_generator not available, using basic prompts")
    ENHANCED_EMPATHY_AVAILABLE = False

class AIResponseGenerator:
    """
    Generates empathetic tutoring responses using AI APIs with enhanced empathy
    """
    
    def __init__(self, db_path: str = "empathetic_tutor.db"):
        self.db_path = db_path
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Setup OpenAI
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key.startswith('sk-'):
            try:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("✅ OpenAI client initialized")
            except Exception as e:
                print(f"❌ OpenAI initialization failed: {e}")
        else:
            print("⚠️  OpenAI API key not found or invalid")
        
        # Setup Anthropic
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and anthropic_key.startswith('sk-ant-'):
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("✅ Anthropic client initialized")
            except Exception as e:
                print(f"❌ Anthropic initialization failed: {e}")
        else:
            print("⚠️  Anthropic API key not found or invalid")
        
        # Initialize enhanced empathy generator
        if ENHANCED_EMPATHY_AVAILABLE and CONFIG_AVAILABLE:
            if config.is_enhanced_empathy_enabled():
                self.empathy_generator = EnhancedEmpathyGenerator()
                print("✅ Enhanced empathy system enabled")
            else:
                self.empathy_generator = None
                print("⚠️  Enhanced empathy disabled in config")
        else:
            self.empathy_generator = None
            print("⚠️  Using basic empathy system")
        
        # Print current configuration
        if CONFIG_AVAILABLE:
            print("🔧 CURRENT CONFIGURATION:")
            if hasattr(config, 'print_current_config'):
                config.print_current_config()
        else:
            print("🔧 Using default configuration (no config.yaml found)")
    
    async def generate_response(
        self, 
        message: str, 
        frustration_score: float,
        concepts: List[str],
        student_id: str = "default_student"
    ) -> str:
        """
        Generate response using provider from config with enhanced empathy
        """
        
        # Get provider from config or default to openai
        if CONFIG_AVAILABLE:
            provider = config.get_ai_provider()
        else:
            provider = "openai"
        
        # Debug logging
        if CONFIG_AVAILABLE and config.config.get('system', {}).get('debug', False):
            print(f"🤖 Using AI provider: {provider}")
            print(f"📊 Frustration score: {frustration_score}")
            print(f"🎯 Concepts: {concepts}")
        
        # Handle different providers
        if provider == "disabled":
            return self._fallback_response(frustration_score)
        elif provider == "local":
            return self._fallback_response(frustration_score)
        
        # Get student interaction history for enhanced empathy
        interaction_history = self._get_student_history(student_id)
        
        # Build context using enhanced empathy if available
        if self.empathy_generator:
            context = self.empathy_generator.generate_empathetic_prompt(
                message=message,
                frustration_score=frustration_score,
                concepts=concepts,
                student_id=student_id,
                interaction_history=interaction_history
            )
        else:
            context = self._build_basic_context(message, frustration_score, concepts)
        
        try:
            if provider == "openai" and self.openai_client:
                # Pass both context AND original message to preserve empathy
                response = await self._generate_openai_response_with_empathy(context, message, frustration_score)
                self._log_response("openai", message, response)
                return response
                
            elif provider == "anthropic" and self.anthropic_client:
                response = await self._generate_anthropic_response_with_empathy(context, message, frustration_score)
                self._log_response("anthropic", message, response)
                return response
                
            else:
                print(f"⚠️  Provider {provider} not available, using fallback")
                return await self._handle_fallback(message, frustration_score, concepts, student_id)
        
        except Exception as e:
            print(f"❌ AI API Error with {provider}: {e}")
            if CONFIG_AVAILABLE and config.is_fallback_enabled():
                return await self._handle_fallback(message, frustration_score, concepts, student_id)
            else:
                return self._fallback_response(frustration_score)
    
    async def _generate_openai_response_with_empathy(self, empathy_context: str, original_message: str, frustration_score: float) -> str:
        """Generate response using OpenAI with empathy context but ensuring correct message"""
        try:
            # Get empathy level
            empathy_level = "high" if frustration_score > 7 else "medium" if frustration_score > 4 else "standard"
            
            # Determine if this is a complex coding question (multi-line code, multiple concepts)
            is_complex_code = len(original_message) > 100 or '\n' in original_message or original_message.count('=') > 1
            
            # Adjust response length based on complexity
            if is_complex_code:
                max_tokens = 150  # Longer for code explanations
                word_limit = "under 100 words"
            else:
                max_tokens = 80   # Shorter for simple questions
                word_limit = "under 60 words"
            
            # Create system prompt that incorporates empathy guidance
            system_prompt = f"""You are a Python programming tutor with enhanced empathy. 

EMPATHY GUIDANCE:
{empathy_context[:200]}...

PYTHON TUTORING RULES:
- {word_limit} total
- Focus on the actual Python coding problem
- If student shows code with errors, identify ALL syntax errors
- For code blocks, provide complete corrected examples
- Be {empathy_level}ly supportive but give practical help
- Always provide working Python examples

For syntax errors, always:
1. Identify the specific error
2. Explain why it's wrong 
3. Show the corrected code
4. Make sure code examples are complete

Example format for code fixes:
"I see the issues! You need: [list fixes]

Here's the corrected code:
```python
[complete working code]
```"

Give practical Python help with complete examples."""

            # Use the original message directly
            user_message = f"Student's Python question: {original_message}"
            
            # Debug logging
            print(f"🔍 DEBUG - Complex code detected: {is_complex_code}")
            print(f"🔍 DEBUG - Using max_tokens: {max_tokens}")
            print(f"🔍 DEBUG - Message length: {len(original_message)} chars")

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,  # Dynamic based on complexity
                temperature=0.3
            )
            
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                print(f"🔍 DEBUG - OpenAI returned: '{result[:100]}...'")
                return result
            else:
                raise Exception("No choices in OpenAI response")
                
        except Exception as e:
            print(f"OpenAI Empathy API Error: {e}")
            raise

    async def _generate_anthropic_response_with_empathy(self, empathy_context: str, original_message: str, frustration_score: float) -> str:
        """Generate response using Anthropic with empathy context"""
        try:
            empathy_level = "high" if frustration_score > 7 else "medium" if frustration_score > 4 else "standard"
            
            prompt = f"""You are a Python tutor with enhanced empathy.

EMPATHY GUIDANCE: {empathy_context[:200]}...

Student's Python question: {original_message}

Provide {empathy_level} empathy and practical Python help under 60 words. If they mention print{{}}, explain Python uses print() not print{{}}."""

            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model='claude-3-sonnet-20240229',
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            print(f"Anthropic Empathy API Error: {e}")
            raise

    async def _handle_fallback(self, message: str, frustration_score: float, concepts: List[str], student_id: str) -> str:
        """Handle fallback to secondary provider"""
        if CONFIG_AVAILABLE:
            fallback_provider = config.get_fallback_provider()
        else:
            fallback_provider = "local"
        
        if fallback_provider == "local" or fallback_provider == "disabled":
            return self._fallback_response(frustration_score)
        
        print(f"🔄 Trying fallback provider: {fallback_provider}")
        
        # Get context for fallback
        interaction_history = self._get_student_history(student_id)
        if self.empathy_generator:
            context = self.empathy_generator.generate_empathetic_prompt(
                message=message,
                frustration_score=frustration_score,
                concepts=concepts,
                student_id=student_id,
                interaction_history=interaction_history
            )
        else:
            context = self._build_basic_context(message, frustration_score, concepts)
        
        try:
            if fallback_provider == "openai" and self.openai_client:
                return await self._generate_openai_response_with_empathy(context, message, frustration_score)
            elif fallback_provider == "anthropic" and self.anthropic_client:
                return await self._generate_anthropic_response_with_empathy(context, message, frustration_score)
        except Exception as e:
            print(f"❌ Fallback provider {fallback_provider} also failed: {e}")
        
        return self._fallback_response(frustration_score)
    
    def _get_student_history(self, student_id: str) -> List[Dict]:
        """Get recent interaction history for empathy analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent interactions for this student
            cursor.execute("""
                SELECT message, frustration_score, created_at, response
                FROM interactions 
                WHERE student_id = ? 
                ORDER BY created_at DESC 
                LIMIT 10
            """, (student_id,))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    "message": row[0],
                    "frustration_score": row[1],
                    "created_at": row[2],
                    "response": row[3]
                })
            
            conn.close()
            return list(reversed(history))  # Return in chronological order
            
        except Exception as e:
            print(f"⚠️  Could not retrieve student history: {e}")
            return []
    
    def _build_basic_context(self, message: str, frustration_score: float, concepts: List[str]) -> str:
        """Build basic context prompt with proper formatting guidelines"""
        
        # Get thresholds from config or use defaults
        if CONFIG_AVAILABLE:
            low_threshold = config.config.get('frustration', {}).get('low_threshold', 3.0)
            high_threshold = config.config.get('frustration', {}).get('high_threshold', 7.0)
        else:
            low_threshold = 3.0
            high_threshold = 7.0
        
        frustration_level = "low" if frustration_score < low_threshold else \
                           "medium" if frustration_score < high_threshold else "high"
        concepts_text = ", ".join(concepts) if concepts else "general programming"
        
        context = f"""You are an empathetic Python programming tutor.

STUDENT MESSAGE: "{message}"
FRUSTRATION LEVEL: {frustration_level} ({frustration_score}/10)
PYTHON CONCEPTS: {concepts_text}

RESPONSE REQUIREMENTS:
- Keep response under 60 words
- Focus ONLY on Python - no other languages
- Write 2-3 short paragraphs maximum
- Use simple Python examples with backticks
- Be encouraging and practical

EMPATHY GUIDELINES:
- High frustration: Be very supportive, acknowledge feelings first
- Medium frustration: Be encouraging and helpful
- Low frustration: Be informative but friendly

PYTHON FOCUS:
- Always show Python syntax: `print()`, `for`, `if`, etc.
- For print syntax issues: explain Python uses print() not print{{}}
- Keep examples simple and Python-focused

Your Python tutoring response:"""

        return context
    
    def _fallback_response(self, frustration_score: float) -> str:
        """Fallback to template responses with proper formatting"""
        
        # Get thresholds from config or use defaults
        if CONFIG_AVAILABLE:
            thresholds = config.config.get('frustration', {})
            low_threshold = thresholds.get('low_threshold', 3.0)
            high_threshold = thresholds.get('high_threshold', 7.0)
        else:
            low_threshold = 3.0
            high_threshold = 7.0
        
        if frustration_score < low_threshold:
            return """Great question! I can see you're thinking through this carefully.

Let me help you understand this better with a clear explanation. What specific part would you like me to focus on?"""
            
        elif frustration_score < high_threshold:
            return """I understand this can be challenging - you're definitely not alone in finding this tricky!

Let's break it down step by step and work through it together. What's the main thing that's confusing you right now?"""
            
        else:
            return """I can really hear the frustration in your message, and I want you to know that's completely normal. This stuff is genuinely hard!

Let's slow down and tackle this one small piece at a time. Take a deep breath with me.

What specific part is causing the most trouble right now? We'll figure this out together."""
    
    def _log_response(self, provider: str, message: str, response: str):
        """Log AI responses if enabled"""
        should_log = True
        if CONFIG_AVAILABLE:
            should_log = config.config.get('system', {}).get('log_ai_responses', True)
        
        if should_log:
            print(f"\n📝 AI Response Log ({provider}):")
            print(f"   Input: {message[:50]}...")
            print(f"   Output: {response[:50]}...")

# Test function
async def test_ai_generator():
    """Test the AI response generator with enhanced empathy"""
    print("🧪 Testing Enhanced AI Response Generator")
    print("=" * 60)
    
    generator = AIResponseGenerator()
    
    test_cases = [
        {
            "message": "what the fuck is why isnt it print{} not working",
            "frustration_score": 6.0,
            "concepts": ["functions", "strings"],
            "description": "Medium frustration - syntax error"
        },
        {
            "message": "Can you help me understand how loops work? I think I'm close to getting it.",
            "frustration_score": 3.0,
            "concepts": ["loops"],
            "description": "Low frustration - curious student"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n=== Test {i+1}: {test['description']} ===")
        print(f"Input: {test['message']}")
        print(f"Frustration: {test['frustration_score']}")
        print(f"Concepts: {test['concepts']}")
        
        try:
            response = await generator.generate_response(
                message=test['message'],
                frustration_score=test['frustration_score'],
                concepts=test['concepts'],
                student_id=f"test_student_{i+1}"
            )
            
            print(f"Response: {response}")
            print("✅ Success")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🤖 AI Response Generator - Enhanced Version")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_ai_generator())