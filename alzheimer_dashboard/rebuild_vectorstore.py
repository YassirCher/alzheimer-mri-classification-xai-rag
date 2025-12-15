"""
Rebuild Vector Store with Model Information

This script rebuilds the FAISS vector store to include the model performance metrics.
Run this once to update the RAG system with model information.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rag_gemini import GeminiRAG

def rebuild_vectorstore():
    """Rebuild the vector store with model information"""
    print("=" * 80)
    print("REBUILDING VECTOR STORE WITH MODEL INFORMATION")
    print("=" * 80)
    
    # Remove old vector store completely
    vectorstore_path = "data/vectorstore"
    if os.path.exists(vectorstore_path):
        print(f"\n🗑️  Removing old vector store from {vectorstore_path}...")
        import shutil
        try:
            shutil.rmtree(vectorstore_path)
            print("✅ Old vector store removed")
        except Exception as e:
            print(f"⚠️  Error removing old vector store: {e}")
    
    # Also remove any cached instances
    import utils.rag_gemini as rag_module
    rag_module._rag_instance = None
    
    # Initialize new RAG system (will create new vector store)
    print("\n🔧 Creating new vector store with model information...")
    print("   This will load:")
    print("   - Model Performance Metrics (YOUR 99.98% model)")
    print("   - 6 Research papers")
    rag = GeminiRAG(articles_dir='data/articles')
    
    # Test query about model
    print("\n" + "=" * 80)
    print("TESTING MODEL INFORMATION RETRIEVAL")
    print("=" * 80)
    
    test_queries = [
        "What is the accuracy of our EfficientNet-B0 model?",
        "Tell me about our deployed model performance",
        "What are our model metrics?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = rag.query(query)
        answer = result['answer'][:400].replace('\n', ' ')
        print(f"\n💬 Answer: {answer}...")
        
        # Check if it mentions 99.98%
        if '99.98' in result['answer'] or '0.9998' in result['answer']:
            print("   ✅ Correctly references YOUR model (99.98%)")
        elif '90.32' in result['answer']:
            print("   ❌ ERROR: Still referencing research paper model (90.32%)")
        
        sources = [s['source'] for s in result['sources'][:3]]
        print(f"\n📚 Sources: {sources}")
        
        if 'Model Performance Metrics' in str(sources):
            print("   ✅ Found model info in sources")
        else:
            print("   ⚠️  Model info not in top sources")
    
    print("\n" + "=" * 80)
    print("✅ VECTOR STORE REBUILD COMPLETE")
    print("=" * 80)
    print("\n💡 The chatbot should now answer with 99.98% accuracy!")
    print("⚠️  If it still gives wrong answers, restart Flask app completely")

if __name__ == "__main__":
    rebuild_vectorstore()
