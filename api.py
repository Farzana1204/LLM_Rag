from dotenv import load_dotenv
import os
from fastapi import FastAPI 
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from RAG import MODEL_NAME, load_pages, chunk_pages, embed_texts, retrieve, file_path
import uvicorn
import pickle
import numpy as np

load_dotenv()

print ("key loaded:",bool(os.getenv("GROQ_API_KEY")))

state = {}
CACHE_FILE = r"C:\Projects\LLM_Rag\cache_embeddings.pkl"

def save_cache(chunks, vectors):
    """Save chunks and vectors to cache file"""
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "vectors": vectors}, f)
    print("[+] Cache saved!")

def load_cache():
    """Load chunks and vectors from cache file"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            print("[+] Cache loaded!")
            return data["chunks"], data["vectors"]
        except Exception as e:
            print(f"[-] Cache load failed: {e}")
    return None, None
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(">>> STARTUP: building things")

    model = SentenceTransformer(MODEL_NAME)
    state["model"] = model
    pages = load_pages(file_path)
    chunks = chunk_pages(pages)
    print("Chunks:" ,chunks)
    print(type(chunks))
    vector = embed_texts(model, [c["text"] for c in chunks])
    save_cache(chunks, vector)
    state["chunks"] = chunks
    state["vector"] = vector

    print(">>> COMPLETED!")
    
    yield
    print(">>> SHUTDOWN: cleaning up")
    state.clear()

app = FastAPI(title="LLM RAG API", lifespan=lifespan)

@app.get("/")
async def root():
    return {"title": "RAG", "message": "Welcome to LLM RAG API"}

@app.post("/query")
async def query_rag(question: str):
    model = state.get("model")
    chunks = state.get("chunks")
    vector = state.get("vector")
    
    if not model or not chunks or vector is None:
        return {"error": "RAG system not initialized"}
    
    results = retrieve(model, vector, chunks, question)
    return {"question": question, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)




