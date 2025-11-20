import logging
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Gravity Backend")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text" # Fallback to llama3 if needed, but user preferred this

# --- Dummy Data ---
# 20 data points across Tech, Nature, Space, Food
RAW_DATA = [
    {"id": 1, "text": "Artificial Intelligence", "category": "Tech"},
    {"id": 2, "text": "Machine Learning", "category": "Tech"},
    {"id": 3, "text": "React JS Framework", "category": "Tech"},
    {"id": 4, "text": "Python Programming", "category": "Tech"},
    {"id": 5, "text": "Quantum Computing", "category": "Tech"},
    {"id": 6, "text": "Forest Ecosystem", "category": "Nature"},
    {"id": 7, "text": "Ocean Conservation", "category": "Nature"},
    {"id": 8, "text": "Mountain Hiking", "category": "Nature"},
    {"id": 9, "text": "Wildlife Photography", "category": "Nature"},
    {"id": 10, "text": "Renewable Energy", "category": "Nature"},
    {"id": 11, "text": "Mars Exploration", "category": "Space"},
    {"id": 12, "text": "Black Holes", "category": "Space"},
    {"id": 13, "text": "International Space Station", "category": "Space"},
    {"id": 14, "text": "Galaxies and Stars", "category": "Space"},
    {"id": 15, "text": "NASA Missions", "category": "Space"},
    {"id": 16, "text": "Pizza Margherita", "category": "Food"},
    {"id": 17, "text": "Sushi Roll", "category": "Food"},
    {"id": 18, "text": "Chocolate Cake", "category": "Food"},
    {"id": 19, "text": "Spicy Tacos", "category": "Food"},
    {"id": 20, "text": "Fresh Salad", "category": "Food"},
]

# Cache for data embeddings to avoid re-computing them every time (in a real app, use a vector DB)
DATA_EMBEDDINGS = {}

def get_embedding(text: str) -> np.ndarray:
    """Fetch embedding from Ollama."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL_NAME, "prompt": text},
            timeout=10
        )
        response.raise_for_status()
        embedding = response.json().get("embedding")
        if not embedding:
            raise ValueError("No embedding found in response")
        return np.array(embedding)
    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama connection failed: {e}")
        # Fallback for testing if Ollama isn't running/model missing (return random vector)
        # IMPORTANT: In production, we should fail hard, but for this prototype we want to be robust.
        # However, the prompt implies we MUST use Ollama. I will raise HTTP 503 if it fails.
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable or model '{MODEL_NAME}' missing. Ensure 'ollama serve' is running and 'ollama pull {MODEL_NAME}' was executed.")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

# Pre-compute embeddings on startup
@app.on_event("startup")
def startup_event():
    logger.info("Pre-computing embeddings for dummy data...")
    try:
        # Check if ollama is reachable first
        requests.get("http://localhost:11434", timeout=2)
        
        for item in RAW_DATA:
            try:
                DATA_EMBEDDINGS[item["id"]] = get_embedding(item["text"])
            except Exception as e:
                logger.warning(f"Failed to embed '{item['text']}': {e}")
        logger.info(f"Computed {len(DATA_EMBEDDINGS)} embeddings.")
    except Exception:
        logger.warning("Ollama not detected on startup. Embeddings will be fetched on demand (and might fail).")

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: SearchRequest):
    query_text = request.query
    if not query_text:
        return []

    try:
        query_vec = get_embedding(query_text)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during embedding")

    results = []
    for item in RAW_DATA:
        item_vec = DATA_EMBEDDINGS.get(item["id"])
        
        # If startup failed to get embeddings, try now (lazy load)
        if item_vec is None:
            try:
                item_vec = get_embedding(item["text"])
                DATA_EMBEDDINGS[item["id"]] = item_vec
            except:
                similarity = 0.0 # Default to 0 if we can't get embedding
                results.append({**item, "similarity": similarity})
                continue

        similarity = cosine_similarity(query_vec, item_vec)
        results.append({**item, "similarity": similarity})

    # Sort by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results

@app.get("/health")
def health():
    return {"status": "ok"}
