import faiss
import pickle
import numpy as np
import requests
from huggingface_hub import hf_hub_download
from typing import List, Tuple
import logging
from config import settings
import os

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        """Initialize the vector store by loading FAISS index and metadata from HuggingFace"""
        self.index = None
        self.metadata = None
        self.hf_token = os.getenv("HF_TOKEN", None)  # Optional: for rate limits
        self.load_vectorstore()
    
    def load_vectorstore(self):
        """Load FAISS index and metadata from HuggingFace repository"""
        try:
            logger.info(f"Downloading FAISS index from HuggingFace: {settings.HF_REPO_ID}")
            faiss_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=settings.FAISS_INDEX_FILE,
                repo_type="model"
            )
            
            logger.info(f"Downloading metadata from HuggingFace: {settings.HF_REPO_ID}")
            metadata_path = hf_hub_download(
                repo_id=settings.HF_REPO_ID,
                filename=settings.METADATA_FILE,
                repo_type="model"
            )
            
            logger.info("Loading FAISS index...")
            self.index = faiss.read_index(faiss_path)
            
            logger.info("Loading metadata...")
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"VectorStore loaded successfully!")
            logger.info(f"Index size: {self.index.ntotal} vectors")
            logger.info(f"Metadata entries: {len(self.metadata)}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert query text to embedding vector using HuggingFace Inference API
        This uses NO local RAM for model loading!
        """
        try:
            api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
            # api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{settings.EMBEDDING_MODEL}"
            
            headers = {}
            if self.hf_token:
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": query, "options": {"wait_for_model": True}}
            )
            
            if response.status_code != 200:
                raise Exception(f"HF API error: {response.status_code} - {response.text}")
            
            embedding = np.array(response.json())
            
            # Handle different response formats
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            elif len(embedding.shape) == 3:
                # Some models return [batch, tokens, dims] - take mean over tokens
                embedding = embedding.mean(axis=1)
            
            return embedding.astype('float32')
            
        except Exception as e:
            logger.error(f"Error embedding query via HF API: {e}")
            raise
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Search for similar documents in the vector store
        
        Args:
            query: Search query string
            top_k: Number of top results to return (default from settings)
        
        Returns:
            List of tuples (document_text, similarity_score)
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        try:
            # Embed the query using HF API (no local RAM needed!)
            query_vector = self.embed_query(query)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, top_k)
            
            # Prepare results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.metadata):
                    # Convert FAISS L2 distance to similarity score (inverse)
                    # Lower distance = higher similarity
                    similarity = 1 / (1 + distance)
                    
                    # Filter by similarity threshold
                    if similarity >= settings.SIMILARITY_THRESHOLD:
                        doc_text = self.metadata[idx]
                        results.append((doc_text, float(similarity)))
            
            logger.info(f"Found {len(results)} relevant documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise
    
    def get_context(self, query: str, top_k: int = None) -> str:
        """
        Get formatted context string from search results
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
        
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Source {i}] (Relevance: {score:.2f})\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def get_sources(self, query: str, top_k: int = None) -> List[str]:
        """
        Get list of source texts for citation
        
        Args:
            query: Search query
            top_k: Number of sources to retrieve
        
        Returns:
            List of source texts
        """
        results = self.search(query, top_k)
        return [doc[:100] + "..." if len(doc) > 100 else doc for doc, _ in results]

