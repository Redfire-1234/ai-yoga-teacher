from groq import Groq
from typing import List, Dict, Tuple
import logging
from vectorstore import VectorStore
from config import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self):
        """Initialize RAG Engine with Groq client and VectorStore"""
        try:
            # Initialize Groq client
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.client = Groq(api_key=settings.GROQ_API_KEY)
            logger.info("Groq client initialized successfully")
            
            # Initialize VectorStore
            self.vectorstore = VectorStore()
            logger.info("RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG Engine: {e}")
            raise
    
    def format_conversation_history(self, history: List[Dict]) -> List[Dict]:
        """
        Format conversation history for Groq API
        
        Args:
            history: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Formatted history for Groq API
        """
        # Keep only last N exchanges
        max_messages = settings.MAX_HISTORY_LENGTH * 2
        if len(history) > max_messages:
            history = history[-max_messages:]
        
        return history
    
    def build_prompt(self, query: str, context: str, conversation_history: List[Dict]) -> List[Dict]:
        """
        Build the complete prompt with system message, context, and history
        
        Args:
            query: User's current query
            context: Retrieved context from vector store
            conversation_history: Previous conversation messages
        
        Returns:
            List of messages for Groq API
        """
        messages = [
            {
                "role": "system",
                "content": settings.SYSTEM_PROMPT
            }
        ]
        
        # Add conversation history
        formatted_history = self.format_conversation_history(conversation_history)
        messages.extend(formatted_history)
        
        # Add current query with context
        user_message = f"""Based on the following context from the yoga knowledge base, please answer the question.

Context:
{context}

Question: {query}

Please provide a helpful, accurate response based on the context above. If the context doesn't contain enough information to fully answer the question, you can supplement with your general yoga knowledge, but prioritize the provided context."""
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def get_response(
        self, 
        query: str, 
        conversation_history: List[Dict] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate response using RAG pipeline
        
        Args:
            query: User's question
            conversation_history: Previous conversation messages
        
        Returns:
            Tuple of (response_text, sources_list)
        """
        try:
            if conversation_history is None:
                conversation_history = []
            
            # Retrieve relevant context from vector store
            logger.info(f"Retrieving context for query: {query[:50]}...")
            context = self.vectorstore.get_context(query)
            sources = self.vectorstore.get_sources(query)
            
            # Build prompt with context and history
            messages = self.build_prompt(query, context, conversation_history)
            
            # Generate response using Groq
            logger.info("Generating response with Groq...")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=settings.GROQ_MODEL,
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
                top_p=1,
                stream=False
            )
            
            response = chat_completion.choices[0].message.content
            logger.info("Response generated successfully")
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return a fallback response
            fallback_response = "I apologize, but I encountered an error processing your question. Please try again or rephrase your question."
            return fallback_response, []
    
    def get_streaming_response(
        self, 
        query: str, 
        conversation_history: List[Dict] = None
    ):
        """
        Generate streaming response using RAG pipeline (for future use)
        
        Args:
            query: User's question
            conversation_history: Previous conversation messages
        
        Yields:
            Response chunks
        """
        try:
            if conversation_history is None:
                conversation_history = []
            
            # Retrieve context
            context = self.vectorstore.get_context(query)
            
            # Build prompt
            messages = self.build_prompt(query, context, conversation_history)
            
            # Stream response
            stream = self.client.chat.completions.create(
                messages=messages,
                model=settings.GROQ_MODEL,
                temperature=settings.GROQ_TEMPERATURE,
                max_tokens=settings.GROQ_MAX_TOKENS,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield "I apologize, but I encountered an error. Please try again."