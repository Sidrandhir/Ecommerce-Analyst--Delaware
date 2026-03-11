"""
diagnose.py — Run this to find which models your API key supports.
Usage: python diagnose.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GOOGLE_API_KEY not set in .env")
    exit(1)

print(f"API Key found: {api_key[:8]}...")
print()

# Test 1: List available models via new SDK
print("=== Available models (google-genai SDK) ===")
try:
    from google import genai
    client = genai.Client(api_key=api_key)
    models = client.models.list()
    chat_models = []
    embed_models = []
    for m in models:
        name = m.name if hasattr(m, 'name') else str(m)
        supported = getattr(m, 'supported_actions', []) or []
        if 'generateContent' in str(supported) or 'GENERATE_CONTENT' in str(supported):
            chat_models.append(name)
        if 'embedContent' in str(supported) or 'EMBED_CONTENT' in str(supported):
            embed_models.append(name)

    print("Chat models:")
    for m in chat_models: print(f"  {m}")
    print("Embedding models:")
    for m in embed_models: print(f"  {m}")
except Exception as e:
    print(f"New SDK listing failed: {e}")

print()

# Test 2: Probe specific model names directly
print("=== Direct model probe ===")
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage

chat_candidates = [
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
    "gemini-1.5-flash", "gemini-1.5-flash-001",
    "gemini-1.5-pro", "gemini-1.5-pro-001",
]
embed_candidates = [
    "models/text-embedding-004", "text-embedding-004",
    "models/embedding-001", "embedding-001",
]

print("Chat models:")
for model in chat_candidates:
    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0, max_output_tokens=5, google_api_key=api_key)
        llm.invoke([HumanMessage(content="hi")])
        print(f"  WORKS: {model}")
    except Exception as e:
        print(f"  FAIL:  {model} — {type(e).__name__}")

print("Embedding models:")
for model in embed_candidates:
    try:
        emb = GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)
        emb.embed_query("test")
        print(f"  WORKS: {model}")
    except Exception as e:
        print(f"  FAIL:  {model} — {type(e).__name__}")