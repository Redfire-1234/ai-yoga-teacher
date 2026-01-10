import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from typing import List, Tuple
import logging
from config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        """Initialize vector store - Load ONCE at startup"""
        self.embedding_model = None
        self.index = None
        self.metadata = None
        self.load_vectorstore()
    
    def load_vectorstore(self):
        """Load embedding model, FAISS index, and metadata"""
        try:
            # Load local embedding model ONCE
            logger.info("Loading local SentenceTransformer model...")
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("✅ Embedding model loaded successfully")
            
            # Download FAISS index from HuggingFace
            logger.info(f"Downloading FAISS index: {settings.HF_REPO_ID}")
            faiss_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=settings.FAISS_INDEX_FILE,
                repo_type="model"
            )
            
            # Download metadata from HuggingFace
            logger.info(f"Downloading metadata: {settings.METADATA_FILE}")
            metadata_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=settings.METADATA_FILE,
                repo_type="model"
            )
            
            # Load FAISS index
            logger.info("Loading FAISS index...")
            self.index = faiss.read_index(faiss_path)
            
            # Load metadata
            logger.info("Loading metadata...")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info("=" * 60)
            logger.info("✅ VectorStore initialized successfully!")
            logger.info(f"   Index size: {self.index.ntotal} vectors")
            logger.info(f"   Metadata entries: {len(self.metadata)}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Convert text to embedding using LOCAL model"""
        try:
            embedding = self.embedding_model.encode([query], show_progress_bar=False)
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Search for similar documents using FAISS"""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        try:
            # Embed query using LOCAL model
            query_vector = self.embed_query(query)
            
            # Search FAISS index
            distances, indices = self.index.search(query_vector, top_k)
            
            # Prepare results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    similarity = 1 / (1 + distance)
                    
                    if similarity >= settings.SIMILARITY_THRESHOLD:
                        doc_text = self.metadata[idx]
                        results.append((doc_text, float(similarity)))
            
            logger.info(f"Found {len(results)} documents for: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def get_context(self, query: str, top_k: int = None) -> str:
        """Get formatted context from search results"""
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def get_sources(self, query: str, top_k: int = None) -> List[str]:
        """Get source texts for citations"""
        results = self.search(query, top_k)
        return [doc[:100] + "..." if len(doc) > 100 else doc for doc, _ in results]
