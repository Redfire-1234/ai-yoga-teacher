"""
Test script for AI Yoga Teacher API
Run this after starting the server to verify everything works
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your Render URL when deployed
SESSION_ID = "test_session_123"

def test_health_check():
    """Test health check endpoint"""
    print("\n🔍 Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"✅ RAG Engine: {data['rag_engine']}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_chat(message):
    """Test chat endpoint"""
    print(f"\n💬 User: {message}")
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={
                "message": message,
                "session_id": SESSION_ID
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"🤖 AI: {data['response'][:200]}...")
            print(f"📚 Sources: {len(data['sources'])} documents retrieved")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return False

def test_conversation_history():
    """Test conversation history endpoint"""
    print("\n📖 Testing Conversation History...")
    try:
        response = requests.get(f"{BASE_URL}/conversation/{SESSION_ID}")
        data = response.json()
        print(f"✅ Retrieved {len(data['history'])} messages")
        for i, msg in enumerate(data['history'][-4:], 1):  # Show last 4 messages
            role = msg['role'].capitalize()
            content = msg['content'][:50]
            print(f"   {i}. {role}: {content}...")
        return True
    except Exception as e:
        print(f"❌ History retrieval failed: {e}")
        return False

def test_sessions():
    """Test sessions endpoint"""
    print("\n🔐 Testing Sessions...")
    try:
        response = requests.get(f"{BASE_URL}/sessions")
        data = response.json()
        print(f"✅ Active sessions: {data['total_sessions']}")
        print(f"✅ Session IDs: {data['active_sessions']}")
        return True
    except Exception as e:
        print(f"❌ Sessions check failed: {e}")
        return False

def test_clear_conversation():
    """Test clear conversation endpoint"""
    print("\n🗑️  Testing Clear Conversation...")
    try:
        response = requests.delete(f"{BASE_URL}/conversation/{SESSION_ID}")
        data = response.json()
        print(f"✅ {data['message']}")
        return True
    except Exception as e:
        print(f"❌ Clear conversation failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧘‍♂️ AI Yoga Teacher API Test Suite")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Chat - Basic Question", lambda: test_chat("What is yoga?")),
        ("Chat - Specific Pose", lambda: test_chat("Tell me about Surya Namaskar")),
        ("Chat - Follow-up", lambda: test_chat("How do I practice it safely?")),
        ("Conversation History", test_conversation_history),
        ("Active Sessions", test_sessions),
        ("Clear Conversation", test_clear_conversation),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            time.sleep(1)  # Small delay between tests
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {name}")
    
    print(f"\n🎯 Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working perfectly!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    print("Starting API tests...")
    print(f"Target URL: {BASE_URL}")
    print(f"Session ID: {SESSION_ID}\n")
    
    run_all_tests()