# Semantic Gravity: Physics-Based Neural Search Engine

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20D3.js-61DAFB)
![Python](https://img.shields.io/badge/Backend-FastAPI%20%2B%20NumPy-3776AB)
![AI](https://img.shields.io/badge/AI-Ollama%20(Local)-000000)

**Semantic Gravity** is an experimental search interface that reimagines data visualization. Instead of static lists, it treats information as physical matter. Using **Local LLMs (Ollama)** and **Vector Embeddings**, the system calculates the semantic relevance of data points and maps that score to gravitational force in a physics simulation.

> **The Core Concept:** In this universe, **Relevance = Gravity**.
> Highly relevant data physically falls to the center of the user's attention, while irrelevant data floats into the void.

---

## 🏗 Architecture

This project utilizes a decoupled 3-tier architecture to separate the Neural Compute layer from the Physics Rendering layer.

```mermaid
graph TD
    subgraph "Frontend Layer (Client)"
        UI[React UI] --> |User Query| API[Fetch API]
        D3[D3.js Physics Engine] --> |Render @ 60FPS| Canvas[HTML5 Canvas]
    end

    subgraph "Logic Layer (Server)"
        FastAPI[Python FastAPI] --> |Orchestration| VectorEngine
        FastAPI --> |JSON Response| UI
    end

    subgraph "Intelligence Layer (Local AI)"
        VectorEngine[NumPy Vector Store] --> |Text| Ollama
        Ollama[Ollama Service] --> |Embeddings (768 dim)| VectorEngine
    end
