# AI Yoga Teacher Backend 🧘‍♂️

RAG-powered Yoga guidance API with conversation memory using FastAPI, Groq, and FAISS.

## Project Structure

```
ai-yoga-teacher/
│
├── app.py              # FastAPI routes and endpoints
├── rag_engine.py       # RAG logic with Groq LLM
├── vectorstore.py      # FAISS vector store loader
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
├── .env.example        # Environment template
└── README.md           # This file
```

## Features

- ✅ RAG-based responses using FAISS vector store
- ✅ Groq LLM integration (llama-3.1-70b-versatile)
- ✅ Conversation history/memory management
- ✅ HuggingFace model loading
- ✅ FastAPI with CORS support
- ✅ Ready for Render deployment

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd ai-yoga-teacher
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your_actual_groq_api_key_here
```

**Get your Groq API key:** https://console.groq.com/keys

### 5. Run Locally

```bash
# Development mode with auto-reload
python app.py

# Or with uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: `http://localhost:8000`

## API Endpoints

### Health Check
```bash
GET /
GET /health
```

### Chat
```bash
POST /chat
Content-Type: application/json

{
  "message": "What is Surya Namaskar?",
  "session_id": "user123"
}
```

Response:
```json
{
  "response": "Namaste 🙏 Surya Namaskar, or Sun Salutation...",
  "session_id": "user123",
  "sources": ["Source text 1...", "Source text 2..."]
}
```

### Get Conversation History
```bash
GET /conversation/{session_id}
```

### Clear Conversation
```bash
DELETE /conversation/{session_id}
```

### List Active Sessions
```bash
GET /sessions
```

## Deployment on Render

### 1. Create `render.yaml`

```yaml
services:
  - type: web
    name: ai-yoga-teacher
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
```

### 2. Deploy Steps

1. Push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. New → Web Service
4. Connect your GitHub repository
5. Add environment variable: `GROQ_API_KEY`
6. Deploy!

### 3. Update Frontend

Update your HTML to point to Render URL:
```javascript
const API_URL = "https://your-app.onrender.com";
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | - | Your Groq API key |
| `GROQ_MODEL` | No | llama-3.1-70b-versatile | Groq model name |
| `GROQ_TEMPERATURE` | No | 0.7 | Response randomness (0-1) |
| `TOP_K_RESULTS` | No | 3 | Number of docs to retrieve |
| `MAX_HISTORY_LENGTH` | No | 10 | Conversation memory size |

## Testing the API

### Using cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about meditation", "session_id": "test123"}'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "What is pranayama?",
        "session_id": "user123"
    }
)
print(response.json())
```

## Troubleshooting

### Issue: "GROQ_API_KEY not found"
**Solution:** Make sure `.env` file exists and contains `GROQ_API_KEY=your_key`

### Issue: "Failed to download from HuggingFace"
**Solution:** Check internet connection. The app downloads vector store files on first run.

### Issue: "No module named 'faiss'"
**Solution:** Install FAISS: `pip install faiss-cpu`

### Issue: Low similarity scores
**Solution:** Adjust `SIMILARITY_THRESHOLD` in config.py (lower = more results)

## Vector Store Info

- **HuggingFace Repo:** `Redfire-1234/ai-yoga-teacher`
- **Index File:** `hatha_yoga.bin` (FAISS index)
- **Metadata:** `hatha_yoga.pkl` (Document texts)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`

## Development

### Adding New Features

1. **Add new endpoint** in `app.py`
2. **Update RAG logic** in `rag_engine.py`
3. **Modify vector search** in `vectorstore.py`
4. **Change settings** in `config.py`

### Code Structure

- **app.py**: API routes, request/response handling
- **rag_engine.py**: Groq integration, prompt building, response generation
- **vectorstore.py**: FAISS loading, embedding, similarity search
- **config.py**: All configuration and settings

## License

MIT

## Support

For issues or questions:
- Create GitHub issue
- Check Render logs for deployment errors
- Verify Groq API key is valid