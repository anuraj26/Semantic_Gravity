# Semantic Gravity

A visual search engine where the "gravity" of an object is determined by its semantic relevance to a user's query.

## Prerequisites

1.  **Ollama**: You must have [Ollama](https://ollama.com/) installed and running.
2.  **Model**: You need the `nomic-embed-text` model.
    ```bash
    ollama pull nomic-embed-text
    ```

## How to Run

You need to run the Backend and Frontend in separate terminals.

### 1. Backend (Python FastAPI)

```bash
cd backend
# Create venv if not exists
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run server
uvicorn main:app --reload
```
The backend runs on `http://localhost:8000`.

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```
The frontend runs on `http://localhost:5173`.

## Troubleshooting

- **Ollama Connection Failed**: Ensure Ollama is running (`ollama serve`) and you have pulled the model (`ollama pull nomic-embed-text`).
- **Backend 404**: If the backend says 404 for Ollama, check your Ollama version.
