from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import os
import glob
from dotenv import load_dotenv

load_dotenv()

class GeminiRAG:
    """RAG system using local embeddings + Gemini LLM"""
    
    def __init__(self, articles_dir='data/articles'):
        # Try multiple possible paths for articles
        possible_paths = [
            articles_dir,
            '../data/articles',
            '../../data/articles'
        ]
        
        self.articles_dir = None
        for path in possible_paths:
            full_path = os.path.abspath(path)
            if os.path.exists(full_path):
                self.articles_dir = full_path
                break
        
        if not self.articles_dir:
            print(f"⚠️ Articles directory not found in any of: {possible_paths}")
            self.articles_dir = articles_dir  # Use default even if not found
        
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if not self.api_key:
            print("⚠️ GOOGLE_API_KEY not found - LLM responses will be limited")
        
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.initialize()
    
    def initialize(self):
        """Initialize embeddings and vector store"""
        try:
            print("🔧 Initializing RAG system...")
            
            # Use FREE local HuggingFace embeddings (no API, no limits!)
            print("  Loading local embeddings (HuggingFace)...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("  ✓ Local embeddings loaded")
            
            # Load documents
            print("  Loading documents...")
            documents = self.load_and_chunk_documents()
            
            if documents:
                print(f"  Creating vector store with {len(documents)} chunks...")
                self.vectorstore = FAISS.from_documents(
                    documents,
                    self.embeddings
                )
                
                # Save vector store
                try:
                    os.makedirs("data", exist_ok=True)
                    self.vectorstore.save_local("data/vectorstore")
                    print("  ✓ Vector store saved to data/vectorstore")
                except Exception as e:
                    print(f"  ⚠ Could not save vector store: {e}")
                
                # Setup QA chain
                self.setup_qa_chain()
                print("✅ RAG system initialized successfully!")
                print(f"📚 Loaded {len(documents)} document chunks from {len(glob.glob(os.path.join(self.articles_dir, '*.txt')))} articles")
            else:
                print("⚠️ No documents loaded")
        
        except Exception as e:
            print(f"❌ Error initializing RAG: {e}")
            import traceback
            traceback.print_exc()
    
    def load_and_chunk_documents(self) -> list:
        """Load and intelligently chunk documents"""
        all_documents = []
        
        # Load model information first
        model_info_paths = [
            'data/model_info.txt',
            '../data/model_info.txt',
            'alzheimer_dashboard/data/model_info.txt',
            os.path.join(os.path.dirname(self.articles_dir), 'model_info.txt')
        ]
        
        model_info_path = None
        for path in model_info_paths:
            if os.path.exists(path):
                model_info_path = path
                break
        
        if model_info_path:
            try:
                with open(model_info_path, 'r', encoding='utf-8') as f:
                    model_content = f.read()
                
                # Add model info as a special document
                model_doc = Document(
                    page_content=model_content,
                    metadata={
                        'source': 'Model Performance Metrics',
                        'type': 'model_info',
                        'chunk_id': 0,
                        'total_chunks': 1
                    }
                )
                all_documents.append(model_doc)
                print(f"  ✓ Loaded model performance information")
            except Exception as e:
                print(f"  ⚠ Could not load model info: {e}")
        
        if not os.path.exists(self.articles_dir):
            print(f"❌ Articles directory not found: {self.articles_dir}")
            return all_documents  # Return model info if available
        
        article_files = glob.glob(os.path.join(self.articles_dir, '*.txt'))
        print(f"  Found {len(article_files)} article files")
        
        # Smart text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n=====", "\n\n", "\n", ". ", " "],
            length_function=len
        )
        
        for article_file in article_files:
            try:
                with open(article_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': os.path.basename(article_file),
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    )
                    all_documents.append(doc)
                
                print(f"    ✓ {os.path.basename(article_file)}: {len(chunks)} chunks")
            
            except Exception as e:
                print(f"    ✗ Error loading {article_file}: {e}")
        
        return all_documents
    
    def setup_qa_chain(self):
        """Setup QA chain with Gemini LLM"""
        try:
            if not self.api_key:
                print("  ⚠ No API key - using retrieval-only mode")
                self.qa_chain = None
                return
            
            # Initialize Gemini LLM (only for generation, not embeddings)
            print("  Connecting to Gemini LLM...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-flash-latest",
                google_api_key=self.api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            
            # Custom prompt
            prompt_template = """You are a medical AI assistant specialized in Alzheimer's disease and brain imaging. You have access to medical research papers AND information about OUR DEPLOYED EfficientNet-B0 classification model.

IMPORTANT: When answering questions about "the model", "our model", "this model", model accuracy, or model performance, you MUST use information from the "Model Performance Metrics" source that shows 99.98% accuracy. DO NOT use accuracy numbers from research papers.

Research Context:
{context}

Question: {question}

CRITICAL FORMATTING RULES:
1. NEVER start or end a sentence with ** or *
2. NEVER use **text** for bold - just write the text normally
3. NEVER use *text* for italic - just write the text normally
4. DO NOT use markdown formatting at all
5. Use plain text with clear paragraphs
6. For lists, start each item on a new line with a number or bullet
7. Use simple, clean language
8. Let the frontend handle all formatting

Content Rules:
- When asked about OUR model performance, cite the "Model Performance Metrics" source
- For medical questions, use the research papers
- Clearly distinguish between our deployed model and models mentioned in research papers
- Be specific: "Our deployed EfficientNet-B0 model achieves 99.98% accuracy" (not 90.32% from papers)
- Always mention which source you're using

Answer (plain text only, no markdown):"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with custom retriever
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}  # Increased to include model info
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            print("  ✓ Gemini LLM connected")
        
        except Exception as e:
            print(f"  ⚠ Could not connect to Gemini LLM: {e}")
            print("  Using retrieval-only mode (will return relevant documents)")
            self.qa_chain = None
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        try:
            if not self.vectorstore:
                return {
                    'answer': 'RAG system not initialized. Please check if articles are in data/articles/',
                    'sources': []
                }
            
            # If we have LLM, use it
            if self.qa_chain:
                try:
                    result = self.qa_chain({'query': question})
                    
                    sources = []
                    for doc in result.get('source_documents', []):
                        source_info = {
                            'source': doc.metadata.get('source', 'Unknown'),
                            'chunk_id': doc.metadata.get('chunk_id', 0),
                            'preview': doc.page_content[:200] + '...'
                        }
                        sources.append(source_info)
                    
                    return {
                        'answer': result.get('result', 'No answer generated'),
                        'sources': sources
                    }
                except Exception as e:
                    print(f"LLM error: {e}, falling back to retrieval")
                    # Fall through to retrieval-only mode
            
            # Fallback: retrieval-only mode (no LLM)
            print(f"  Retrieval-only query: {question[:50]}...")
            docs = self.vectorstore.similarity_search(question, k=4)
            
            # Format retrieved documents
            answer_parts = ["**Note:** AI service is currently busy. Here is the relevant information found in the documents:\n"]
            
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content.strip()
                
                # Clean up content - remove excessive newlines
                content = " ".join(content.split())
                
                # Truncate if too long
                if len(content) > 400:
                    content = content[:400] + "..."
                
                answer_parts.append(f"\n### {source}")
                answer_parts.append(f"{content}\n")
            
            answer = "\n".join(answer_parts)
            
            sources = [{
                'source': doc.metadata.get('source', 'Unknown'),
                'chunk_id': doc.metadata.get('chunk_id', 0),
                'preview': doc.page_content[:200] + '...'
            } for doc in docs]
            
            return {
                'answer': answer,
                'sources': sources
            }
        
        except Exception as e:
            print(f"❌ Query error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f'Error processing query: {str(e)}',
                'sources': []
            }
    
    @classmethod
    def load_existing(cls, articles_dir='data/articles'):
        """Load existing vector store"""
        instance = cls.__new__(cls)
        instance.articles_dir = articles_dir
        instance.api_key = os.getenv('GOOGLE_API_KEY')
        
        try:
            print("🔄 Loading existing vector store...")
            
            # Use local embeddings (same as initialization)
            instance.embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            instance.vectorstore = FAISS.load_local(
                "data/vectorstore",
                instance.embeddings,
                allow_dangerous_deserialization=True
            )
            
            instance.setup_qa_chain()
            print("✅ Vector store loaded from disk!")
            return instance
        
        except Exception as e:
            print(f"⚠️ Could not load existing vector store: {e}")
            print("Creating new vector store...")
            return cls(articles_dir)

# Global instance
_rag_instance = None

def get_rag():
    """Get or create RAG instance"""
    global _rag_instance
    if _rag_instance is None:
        if os.path.exists("data/vectorstore"):
            try:
                _rag_instance = GeminiRAG.load_existing()
            except:
                _rag_instance = GeminiRAG()
        else:
            _rag_instance = GeminiRAG()
    return _rag_instance
