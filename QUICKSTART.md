# Quick Start Guide 🚀

Get your AI Yoga Teacher running in 5 minutes!

## Prerequisites

- Python 3.9+ installed
- Groq API key ([Get one here](https://console.groq.com/keys))
- Git installed

## Step 1: Setup Project

```bash
# Clone or create project folder
mkdir ai-yoga-teacher
cd ai-yoga-teacher

# Create all the Python files (copy from artifacts above)
# - app.py
# - rag_engine.py
# - vectorstore.py
# - config.py
# - requirements.txt
# - .env
```

## Step 2: Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Configure Environment

Create `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## Step 5: Run the Server

```bash
python app.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     RAG Engine initialized successfully!
```

## Step 6: Test the API

### Option A: Use the test script
```bash
python test_api.py
```

### Option B: Use cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is yoga?","session_id":"test"}'
```

### Option C: Open the HTML frontend
1. Open `index.html` in your browser
2. Make sure `API_URL` is set to `http://localhost:8000`
3. Start chatting!

## Step 7: Deploy to Render

1. **Create GitHub repository** and push your code
2. **Go to Render** → https://dashboard.render.com/
3. **New Web Service** → Connect your repo
4. **Add Environment Variable**: 
   - Key: `GROQ_API_KEY`
   - Value: Your actual Groq API key
5. **Deploy!**
6. **Update frontend**: Change `API_URL` in `index.html` to your Render URL

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt --force-reinstall
```

### "GROQ_API_KEY not found"
- Check `.env` file exists in project root
- No spaces around `=` in `.env`
- API key is valid from Groq Console

### "Failed to download from HuggingFace"
- Check internet connection
- Files download automatically on first run
- May take 1-2 minutes for initial setup

### Vector store not loading
- Verify HuggingFace repo is public: `Redfire-1234/ai-yoga-teacher`
- Check files exist: `hatha_yoga.bin` and `hatha_yoga.pkl`

## File Checklist

Before running, make sure you have:
- ✅ `app.py`
- ✅ `rag_engine.py`
- ✅ `vectorstore.py`
- ✅ `config.py`
- ✅ `requirements.txt`
- ✅ `.env` (with your GROQ_API_KEY)
- ✅ `index.html` (frontend)
- ⭐ `test_api.py` (optional but recommended)
- ⭐ `render.yaml` (for Render deployment)

## Next Steps

1. ✅ Test locally
2. ✅ Deploy to Render
3. ✅ Update frontend with Render URL
4. ✅ Share with users!

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/chat` | POST | Send message |
| `/conversation/{id}` | GET | Get history |
| `/conversation/{id}` | DELETE | Clear history |
| `/sessions` | GET | List sessions |

## Support

- 📧 Check Render logs if deployment fails
- 🔍 Use `test_api.py` to debug issues
- 📚 Read the full README.md for detailed info

Happy coding! 🧘‍♂️