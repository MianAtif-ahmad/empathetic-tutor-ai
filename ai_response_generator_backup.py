async def _generate_openai_response_with_empathy(self, empathy_context: str, original_message: str, frustration_score: float) -> str:
        """Generate response using OpenAI with empathy context but ensuring correct message"""
        try:
            # Get empathy level
            empathy_level = "high" if frustration_score > 7 else "medium" if frustration_score > 4 else "standard"
            
            # Create system prompt that incorporates empathy guidance
            system_prompt = f"""You are a Python programming tutor with enhanced empathy. 

EMPATHY GUIDANCE:
{empathy_context[:200]}...

PYTHON TUTORING RULES:
- Under 60 words total
- Focus on the actual Python coding problem
- If student mentions print{{}} or print(), explain Python syntax 
- For "print{{}}" issues: explain Python uses print() not print{{}}
- Be {empathy_level}ly supportive but give practical help
- Always provide working Python examples

Example for print syntax issues:
"I understand your frustration! Python uses parentheses () not curly braces {{}}.

Try: `print("Hello!")`

That's the correct syntax!"

Give practical Python help, not generic responses."""

            # Use the original message directly
            user_message = f"Student's Python question: {original_message}"
            
            # Debug logging
            print(f"🔍 DEBUG - Original message: '{original_message}'")
            print(f"🔍 DEBUG - Empathy level: {empathy_level}")
            print(f"🔍 DEBUG - Using empathy context: {len(empathy_context)} chars")

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=80,
                temperature=0.3
            )
            
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                print(f"🔍 DEBUG - OpenAI returned: '{result[:50]}...'")
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
            raise#ai_response_generator.py - Final Version with Better Formatting
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
                return await self._generate_openai_response(context)
            elif fallback_provider == "anthropic" and self.anthropic_client:
                return await self._generate_anthropic_response(context)
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
        
        context = f"""You are a Python programming tutor helping students learn Python specifically.

STUDENT MESSAGE: "{message}"
FRUSTRATION LEVEL: {frustration_level} ({frustration_score}/10)
PYTHON CONCEPTS: {concepts_text}

RESPONSE REQUIREMENTS:
1. Keep response under 100 words
2. Focus ONLY on Python - no other languages
3. Write 2-3 short paragraphs maximum
4. Use simple Python examples with backticks
5. Be encouraging and practical

EMPATHY GUIDELINES:
- High frustration: Acknowledge feelings, be very supportive
- Medium frustration: Be encouraging and clear
- Low frustration: Be friendly and informative

PYTHON FOCUS:
- Always show Python syntax: `print()`, `for`, `if`, etc.
- Mention Python-specific features when relevant
- Keep examples simple and Python-focused

EXAMPLE RESPONSE:
I can see this is frustrating! Let me help with this Python issue.

In Python, functions use parentheses () not curly braces {{}}.

Try: `print("Hello!")` 

You're learning - this is normal! What happens when you try that?

Your Python tutoring response:"""

        return context
    
    async def _generate_openai_response(self, context: str) -> str:
        """Generate response using OpenAI GPT with better formatting"""
        try:
            # Get OpenAI config or use defaults
            if CONFIG_AVAILABLE:
                openai_config = config.get_openai_config()
            else:
                openai_config = {
                    'model': 'gpt-3.5-turbo',
                    'max_tokens': 80,
                    'temperature': 0.3
                }
            
            # Create a more direct system prompt
            system_prompt = """You are a Python programming tutor. ALWAYS assume questions are about Python coding problems.

RULES:
- Under 60 words total
- If someone mentions print{} or print() or any code syntax, give Python help
- If someone asks about "print{}" specifically, explain Python uses print() not print{}
- Be encouraging but focus on the actual coding problem
- Always provide a working Python example

COMMON ISSUES:
- print{} → Should be print()
- Missing quotes in print statements
- Syntax errors with parentheses vs braces

Example response for "print{} not working":
"I see the issue! Python uses parentheses () not curly braces {}.

Try: `print("Hello!")`

That's the correct Python syntax!"

ALWAYS give practical Python help, not generic responses."""

            # Extract just the student message safely - try multiple patterns
            student_message = "Help with Python code"  # Default fallback
            
            try:
                # Try different context formats
                if 'Student message: "' in context:
                    start = context.find('Student message: "') + len('Student message: "')
                    end = context.find('"', start)
                    student_message = context[start:end] if end != -1 else context[start:start+100]
                elif 'STUDENT MESSAGE: "' in context:
                    start = context.find('STUDENT MESSAGE: "') + len('STUDENT MESSAGE: "')
                    end = context.find('"', start)
                    student_message = context[start:end] if end != -1 else context[start:start+100]
                elif 'message=' in context:
                    start = context.find('message=') + len('message=')
                    end = context.find('\n', start)
                    student_message = context[start:end] if end != -1 else context[start:start+100]
                else:
                    # If no pattern matches, take the first 100 characters
                    student_message = context[:100]
                    
                # Clean up the extracted message
                student_message = student_message.strip().strip('"').strip("'")
                if len(student_message) < 5:  # Too short, use fallback
                    student_message = "Help with Python code"
                    
            except Exception as e:
                print(f"⚠️ Message extraction failed: {e}")
                student_message = "Help with Python code"

            # Debug logging
            print(f"🔍 DEBUG - Student message extracted: '{student_message}'")
            print(f"🔍 DEBUG - Using system prompt: {system_prompt[:100]}...")
            
            user_message = f"Python coding question: {student_message}. Help me fix this Python code issue."
            print(f"🔍 DEBUG - Sending to OpenAI: '{user_message}'")

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=openai_config.get('model', 'gpt-3.5-turbo'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=80,  # Very short responses
                temperature=0.3  # More focused, less creative
            )
            
            # Safely get the response content
            if response.choices and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    result = response.choices[0].message.content.strip()
                    print(f"🔍 DEBUG - OpenAI returned: '{result}'")
                    return result
                else:
                    raise Exception("No message content in response")
            else:
                raise Exception("No choices in OpenAI response")
                
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            raise
    
    async def _generate_anthropic_response(self, context: str) -> str:
        """Generate response using Anthropic Claude with better formatting"""
        try:
            # Get Anthropic config or use defaults
            if CONFIG_AVAILABLE:
                anthropic_config = config.get_anthropic_config()
            else:
                anthropic_config = {
                    'model': 'claude-3-sonnet-20240229',
                    'max_tokens': 200
                }
            
            enhanced_context = f"""You are a helpful programming tutor. Format your responses with clear paragraphs and proper line breaks.

{context}"""
            
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model=anthropic_config.get('model', 'claude-3-sonnet-20240229'),
                max_tokens=anthropic_config.get('max_tokens', 200),
                messages=[
                    {"role": "user", "content": enhanced_context}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API Error: {e}")
            raise
    
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
            "message": "what the hell is print{} not working? i am so frustrated",
            "frustration_score": 8.5,
            "concepts": ["functions", "strings"],
            "description": "High frustration - syntax error"
        },
        {
            "message": "Can you help me understand how loops work? I think I'm close to getting it.",
            "frustration_score": 3.0,
            "concepts": ["loops"],
            "description": "Low frustration - curious student"
        },
        {
            "message": "This error keeps showing up and I don't know why. It's really annoying.",
            "frustration_score": 6.0,
            "concepts": ["errors"],
            "description": "Medium frustration - technical problem"
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

def check_system_status():
    """Check if all components are properly configured"""
    print("🔍 System Status Check:")
    print("=" * 40)
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    
    print(f"OpenAI Key: {'✅ Present' if openai_key and openai_key.startswith('sk-') else '❌ Missing/Invalid'}")
    print(f"Anthropic Key: {'✅ Present' if anthropic_key and anthropic_key.startswith('sk-ant-') else '❌ Missing/Invalid'}")
    print(f"Config Manager: {'✅ Available' if CONFIG_AVAILABLE else '❌ Missing'}")
    print(f"Enhanced Empathy: {'✅ Available' if ENHANCED_EMPATHY_AVAILABLE else '❌ Missing'}")
    
    # Check database
    db_exists = os.path.exists("empathetic_tutor.db")
    print(f"Database: {'✅ Present' if db_exists else '⚠️  Not found (will be created)'}")
    
    if CONFIG_AVAILABLE:
        print(f"Current AI Provider: {config.get_ai_provider()}")
        print(f"Enhanced Empathy Enabled: {config.is_enhanced_empathy_enabled()}")

if __name__ == "__main__":
    print("🤖 AI Response Generator - Enhanced Version")
    print("=" * 60)
    
    # Check system status first
    check_system_status()
    
    # Run tests
    asyncio.run(test_ai_generator())