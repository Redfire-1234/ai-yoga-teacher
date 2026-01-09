from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from rag_engine import RAGEngine
from config import settings
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Yoga Teacher API",
    description="RAG-powered Yoga guidance with conversation memory",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
rag_engine = None

# In-memory conversation storage (use Redis/DB for production)
conversations: Dict[str, List[Dict]] = {}


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    
class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[str] = []


# Startup event
@app.on_event("startup")
async def startup_event():
    global rag_engine
    try:
        logger.info("Initializing RAG Engine...")
        rag_engine = RAGEngine()
        logger.info("RAG Engine initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Engine: {e}")
        raise


# Health check endpoint
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "AI Yoga Teacher API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_engine": "initialized" if rag_engine else "not initialized"
    }


# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG Engine not initialized")
        
        session_id = request.session_id
        user_message = request.message.strip()
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Initialize conversation history for new sessions
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Get conversation history
        history = conversations[session_id]
        
        # Get RAG response
        response, sources = rag_engine.get_response(
            query=user_message,
            conversation_history=history
        )
        
        # Update conversation history
        conversations[session_id].append({
            "role": "user",
            "content": user_message
        })
        conversations[session_id].append({
            "role": "assistant",
            "content": response
        })
        
        # Keep only last 10 exchanges (20 messages) to manage memory
        if len(conversations[session_id]) > 20:
            conversations[session_id] = conversations[session_id][-20:]
        
        logger.info(f"Session {session_id}: Response generated successfully")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get conversation history
@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    if session_id not in conversations:
        return {"session_id": session_id, "history": []}
    
    return {
        "session_id": session_id,
        "history": conversations[session_id]
    }


# Clear conversation history
@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    if session_id in conversations:
        del conversations[session_id]
        return {"message": f"Conversation {session_id} cleared"}
    
    return {"message": f"Conversation {session_id} not found"}


# Get all active sessions
@app.get("/sessions")
async def get_sessions():
    return {
        "active_sessions": list(conversations.keys()),
        "total_sessions": len(conversations)
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
