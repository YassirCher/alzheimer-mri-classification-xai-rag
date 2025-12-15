"""
Quick Test Script - Verify Chatbot Fixes
Run this after rebuilding vector store to test if fixes work
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rag_gemini import get_rag

def test_chatbot():
    print("=" * 80)
    print("TESTING CHATBOT FIXES")
    print("=" * 80)
    
    rag = get_rag()
    
    test_cases = [
        {
            "question": "What is the accuracy of our model?",
            "should_contain": ["99.98", "0.9998"],
            "should_not_contain": ["90.32", "329 minutes"],
            "name": "Model Accuracy Test"
        },
        {
            "question": "Tell me about our deployed EfficientNet-B0 model performance",
            "should_contain": ["99.98", "deployed", "Model Performance Metrics"],
            "should_not_contain": ["90.32"],
            "name": "Deployed Model Test"
        },
        {
            "question": "How does our model compare to benchmarks?",
            "should_contain": ["99.98", "surpass", "best"],
            "should_not_contain": ["90.32"],
            "name": "Benchmark Comparison Test"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}/{len(test_cases)}: {test['name']}")
        print(f"{'='*80}")
        print(f"📝 Question: {test['question']}")
        
        result = rag.query(test['question'])
        answer = result['answer']
        sources = [s['source'] for s in result['sources']]
        
        print(f"\n💬 Answer Preview:")
        print(f"   {answer[:300]}...")
        print(f"\n📚 Sources: {sources[:3]}")
        
        # Check should_contain
        contains_pass = True
        for phrase in test['should_contain']:
            if phrase.lower() in answer.lower():
                print(f"   ✅ Contains '{phrase}'")
            else:
                print(f"   ❌ Missing '{phrase}'")
                contains_pass = False
        
        # Check should_not_contain
        not_contains_pass = True
        for phrase in test['should_not_contain']:
            if phrase.lower() in answer.lower():
                print(f"   ❌ Incorrectly contains '{phrase}'")
                not_contains_pass = False
            else:
                print(f"   ✅ Correctly excludes '{phrase}'")
        
        # Check for ** formatting issues
        formatting_pass = True
        if answer.startswith('**') or answer.endswith('**'):
            print(f"   ⚠️  Response starts/ends with **")
            formatting_pass = False
        else:
            print(f"   ✅ No ** at start/end")
        
        if '. **' in answer or '**.' in answer:
            print(f"   ⚠️  ** found at sentence boundaries")
            formatting_pass = False
        else:
            print(f"   ✅ No ** at sentence boundaries")
        
        # Overall test result
        if contains_pass and not_contains_pass and formatting_pass:
            print(f"\n   ✅ TEST PASSED")
            passed += 1
        else:
            print(f"\n   ❌ TEST FAILED")
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}/{len(test_cases)}")
    print(f"❌ Failed: {failed}/{len(test_cases)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Chatbot is working correctly!")
        print("\n✅ Ready to use. Start Flask app:")
        print("   python app.py")
    else:
        print("\n⚠️  Some tests failed. Troubleshooting:")
        print("   1. Make sure rebuild_vectorstore.py completed successfully")
        print("   2. Check that data/model_info.txt exists")
        print("   3. Verify GOOGLE_API_KEY is set in .env")
        print("   4. Try deleting data/vectorstore and rebuilding")
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_chatbot()
