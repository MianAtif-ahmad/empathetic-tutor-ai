#!/usr/bin/env python3
"""
Comprehensive Test Script for AI Programming Tutor
Tests 50 different cases across various frustration levels, vocabulary, and code issues
"""

import requests
import json
import time
import random
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_STUDENT_ID = "test_student_comprehensive"

class TutorTester:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def test_case(self, case_num, message, expected_frustration_range, expected_concepts, description):
        """Test a single case and evaluate the response"""
        print(f"\n{'='*60}")
        print(f"TEST CASE {case_num}: {description}")
        print(f"{'='*60}")
        print(f"Input: {message}")
        print(f"Expected Frustration: {expected_frustration_range}")
        print(f"Expected Concepts: {expected_concepts}")
        
        try:
            # Send request to API
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                json={
                    "message": message,
                    "student_id": TEST_STUDENT_ID
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                self.failed += 1
                return
            
            data = response.json()
            
            # Extract results
            frustration_score = data.get('frustration_score', 0)
            concepts = data.get('concepts', [])
            ai_response = data.get('response', '')
            empathy_level = data.get('empathy_level', 'unknown')
            
            print(f"\nRESULTS:")
            print(f"  Frustration Score: {frustration_score:.2f}/10")
            print(f"  Empathy Level: {empathy_level}")
            print(f"  Concepts Detected: {concepts}")
            print(f"  Response: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
            
            # Evaluate results
            frustration_ok = expected_frustration_range[0] <= frustration_score <= expected_frustration_range[1]
            concepts_ok = any(concept in concepts for concept in expected_concepts) if expected_concepts else True
            response_relevant = len(ai_response) > 20  # Basic relevance check
            
            # Check for appropriate empathy response
            empathy_ok = True
            if frustration_score > 7 and 'understand' not in ai_response.lower():
                empathy_ok = False
            if frustration_score < 2 and len(ai_response) < 30:
                empathy_ok = False
                
            print(f"\nEVALUATION:")
            print(f"  Frustration Range: {'✅' if frustration_ok else '❌'} ({frustration_score:.1f} in {expected_frustration_range})")
            print(f"  Concept Detection: {'✅' if concepts_ok else '❌'} (found: {concepts})")
            print(f"  Response Quality: {'✅' if response_relevant else '❌'} ({len(ai_response)} chars)")
            print(f"  Empathy Appropriate: {'✅' if empathy_ok else '❌'}")
            
            overall_pass = frustration_ok and concepts_ok and response_relevant and empathy_ok
            
            if overall_pass:
                print(f"  OVERALL: ✅ PASS")
                self.passed += 1
            else:
                print(f"  OVERALL: ❌ FAIL")
                self.failed += 1
            
            # Store results
            self.results.append({
                'case_num': case_num,
                'description': description,
                'message': message,
                'frustration_score': frustration_score,
                'expected_frustration': expected_frustration_range,
                'concepts': concepts,
                'expected_concepts': expected_concepts,
                'response': ai_response,
                'empathy_level': empathy_level,
                'passed': overall_pass
            })
            
            # Brief pause between requests
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Network Error: {e}")
            self.failed += 1
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            self.failed += 1

def run_comprehensive_tests():
    """Run all 50 test cases"""
    tester = TutorTester()
    
    print("🧪 COMPREHENSIVE AI TUTOR TEST SUITE")
    print("=" * 60)
    print(f"Testing API at: {API_BASE_URL}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test cases organized by category
    test_cases = [
        # CATEGORY 1: HIGH FRUSTRATION CASES (7-10)
        (1, "What the fuck is wrong with this stupid print function?!", (6, 10), ["functions", "strings"], "High frustration - profanity"),
        (2, "I HATE PYTHON! Nothing works! This is impossible!", (8, 10), [], "High frustration - caps and strong language"),
        (3, "I'm giving up on this shit. Loops are fucking impossible to understand.", (7, 10), ["loops"], "High frustration - giving up"),
        (4, "This goddamn error keeps showing up and I can't fix it!", (6, 9), ["errors"], "High frustration - persistent error"),
        (5, "Why is programming so damn hard?! I've been stuck for hours!", (7, 10), [], "High frustration - time pressure"),
        
        # CATEGORY 2: MEDIUM FRUSTRATION CASES (4-7)
        (6, "I'm really stuck on this loop and getting frustrated.", (4, 7), ["loops"], "Medium frustration - explicit"),
        (7, "This is annoying. The if statement isn't working right.", (3, 6), ["conditionals"], "Medium frustration - mild profanity"),
        (8, "I'm confused about functions. This is harder than I thought.", (2, 5), ["functions"], "Medium frustration - confusion"),
        (9, "Help! My code is broken and I don't know why.", (3, 6), ["errors"], "Medium frustration - help seeking"),
        (10, "This error message doesn't make sense. So frustrating!", (4, 7), ["errors"], "Medium frustration - error confusion"),
        
        # CATEGORY 3: LOW FRUSTRATION CASES (0-3)
        (11, "Can you help me understand how for loops work?", (0, 3), ["loops"], "Low frustration - polite inquiry"),
        (12, "I'm curious about list comprehensions in Python.", (0, 2), ["lists"], "Low frustration - curiosity"),
        (13, "Could you explain the difference between lists and tuples?", (0, 2), ["lists"], "Low frustration - comparison question"),
        (14, "I'd like to learn more about functions and parameters.", (0, 3), ["functions"], "Low frustration - learning interest"),
        (15, "What's the best way to handle exceptions in Python?", (0, 2), ["errors"], "Low frustration - best practices"),
        
        # CATEGORY 4: SYNTAX ERROR CASES
        (16, "print{hello} isn't working. What am I doing wrong?", (1, 4), ["functions", "strings"], "Syntax error - braces vs parentheses"),
        (17, "for i in range(10) print(i) # Missing colon", (2, 5), ["loops"], "Syntax error - missing colon"),
        (18, "if x = 5: print('five') # Wrong assignment", (2, 5), ["conditionals"], "Syntax error - assignment vs comparison"),
        (19, "def my_function() return 5 # Missing colon", (2, 5), ["functions"], "Syntax error - function definition"),
        (20, "print('Hello world') # Missing closing quote", (1, 4), ["strings"], "Syntax error - string quotes"),
        
        # CATEGORY 5: COMPLEX CODE DEBUGGING
        (21, """My loop isn't working:
for i in range(5)
    print(i)
    if i = 3
        print('three')""", (3, 6), ["loops", "conditionals"], "Complex - multiple syntax errors"),
        
        (22, """This function has problems:
def calculate(x, y)
    result = x + y
    return result""", (2, 5), ["functions"], "Complex - function syntax issues"),
        
        (23, """List comprehension not working:
numbers = [1, 2, 3, 4, 5]
squared = [x^2 for x in numbers]""", (2, 5), ["lists"], "Complex - wrong operator"),
        
        (24, """Exception handling broken:
try:
    result = 10 / 0
catch Exception as e:
    print(e)""", (2, 5), ["errors"], "Complex - wrong exception syntax"),
        
        (25, """Class definition issues:
class MyClass
    def __init__(self, value)
        self.value = value""", (2, 5), ["classes"], "Complex - class syntax errors"),
        
        # CATEGORY 6: LOGIC ERRORS
        (26, "My loop runs forever. How do I fix infinite loops?", (3, 6), ["loops"], "Logic error - infinite loop"),
        (27, "The function returns None instead of the calculation.", (2, 5), ["functions"], "Logic error - return issue"),
        (28, "My if statement never executes the else block.", (2, 5), ["conditionals"], "Logic error - condition issue"),
        (29, "The list index is out of range. What does that mean?", (3, 6), ["lists", "errors"], "Logic error - index error"),
        (30, "My variable is undefined but I declared it.", (2, 5), ["variables"], "Logic error - scope issue"),
        
        # CATEGORY 7: CONCEPT UNDERSTANDING
        (31, "What's the difference between a list and a dictionary?", (0, 2), ["lists", "dictionaries"], "Concept - data structures"),
        (32, "How do lambda functions work in Python?", (0, 2), ["functions"], "Concept - advanced functions"),
        (33, "Explain object-oriented programming concepts.", (0, 2), ["classes"], "Concept - OOP"),
        (34, "What are decorators and how do I use them?", (0, 2), ["functions"], "Concept - decorators"),
        (35, "How does Python handle memory management?", (0, 2), [], "Concept - advanced topic"),
        
        # CATEGORY 8: MIXED VOCABULARY STYLES
        (36, "Yo, this code ain't working, help me out!", (2, 5), [], "Casual - informal language"),
        (37, "Excuse me, I require assistance with this algorithm.", (0, 2), [], "Formal - polite language"),
        (38, "Dude, my script is totally broken!", (3, 6), [], "Casual - slang"),
        (39, "I would appreciate guidance on this programming challenge.", (0, 2), [], "Formal - academic tone"),
        (40, "Ugh, this bug is driving me crazy!", (4, 7), ["errors"], "Casual - mild frustration"),
        
        # CATEGORY 9: EMOTIONAL STATES
        (41, "I'm feeling overwhelmed by all these Python concepts.", (4, 7), [], "Emotional - overwhelmed"),
        (42, "I'm excited to learn more about programming!", (0, 2), [], "Emotional - positive"),
        (43, "I feel like I'm not smart enough for programming.", (5, 8), [], "Emotional - self-doubt"),
        (44, "This makes me feel really stupid and confused.", (5, 8), [], "Emotional - negative self-talk"),
        (45, "I'm proud that I solved the last problem!", (0, 2), [], "Emotional - achievement"),
        
        # CATEGORY 10: EDGE CASES
        (46, "?", (0, 2), [], "Edge case - minimal input"),
        (47, "a" * 500, (1, 4), [], "Edge case - very long input"),
        (48, "Help help help help help help help", (2, 5), [], "Edge case - repetitive"),
        (49, "12345 + 67890 = ?", (0, 2), [], "Edge case - math not code"),
        (50, "Thanks for all your help! You're the best tutor!", (0, 2), [], "Edge case - gratitude")
    ]
    
    # Run all test cases
    for case_data in test_cases:
        tester.test_case(*case_data)
    
    # Generate summary report
    print("\n" + "="*80)
    print("📊 FINAL TEST REPORT")
    print("="*80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {tester.passed} ✅")
    print(f"Failed: {tester.failed} ❌")
    print(f"Success Rate: {(tester.passed / len(test_cases)) * 100:.1f}%")
    
    # Category breakdown
    categories = {
        "High Frustration (1-5)": [r for r in tester.results if r['case_num'] <= 5],
        "Medium Frustration (6-10)": [r for r in tester.results if 6 <= r['case_num'] <= 10],
        "Low Frustration (11-15)": [r for r in tester.results if 11 <= r['case_num'] <= 15],
        "Syntax Errors (16-20)": [r for r in tester.results if 16 <= r['case_num'] <= 20],
        "Complex Code (21-25)": [r for r in tester.results if 21 <= r['case_num'] <= 25],
        "Logic Errors (26-30)": [r for r in tester.results if 26 <= r['case_num'] <= 30],
        "Concepts (31-35)": [r for r in tester.results if 31 <= r['case_num'] <= 35],
        "Vocabulary (36-40)": [r for r in tester.results if 36 <= r['case_num'] <= 40],
        "Emotional (41-45)": [r for r in tester.results if 41 <= r['case_num'] <= 45],
        "Edge Cases (46-50)": [r for r in tester.results if 46 <= r['case_num'] <= 50]
    }
    
    print(f"\n📈 CATEGORY BREAKDOWN:")
    for category, results in categories.items():
        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        percentage = (passed / total) * 100 if total > 0 else 0
        print(f"  {category}: {passed}/{total} ({percentage:.1f}%)")
    
    # Detailed failure analysis
    failures = [r for r in tester.results if not r['passed']]
    if failures:
        print(f"\n❌ FAILED CASES ANALYSIS:")
        for failure in failures:
            print(f"  Case {failure['case_num']}: {failure['description']}")
            print(f"    Expected frustration: {failure['expected_frustration']}, Got: {failure['frustration_score']:.1f}")
            print(f"    Expected concepts: {failure['expected_concepts']}, Got: {failure['concepts']}")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tutor_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            'summary': {
                'total': len(test_cases),
                'passed': tester.passed,
                'failed': tester.failed,
                'success_rate': (tester.passed / len(test_cases)) * 100
            },
            'categories': {k: {'passed': sum(1 for r in v if r['passed']), 'total': len(v)} for k, v in categories.items()},
            'detailed_results': tester.results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print(f"🕒 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return tester.passed / len(test_cases)

if __name__ == "__main__":
    try:
        success_rate = run_comprehensive_tests()
        if success_rate >= 0.8:
            print(f"\n🎉 EXCELLENT! Your AI tutor passed {success_rate*100:.1f}% of tests!")
        elif success_rate >= 0.6:
            print(f"\n👍 GOOD! Your AI tutor passed {success_rate*100:.1f}% of tests!")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT! Your AI tutor passed {success_rate*100:.1f}% of tests!")
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")