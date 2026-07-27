# LocalizeRAG

## Project Overview

LocalizeRAG is a production-quality platform for hybrid GraphRAG-based content generation. This repository contains the initial project scaffold with a FastAPI backend and a React frontend.

The current setup includes a health check API and a static homepage. No business logic or AI features are implemented yet.

## Tech Stack

### Backend
- Python 3.11
- FastAPI
- Uvicorn

### Frontend
- React
- TypeScript
- Vite
- TailwindCSS
- shadcn/ui

## Folder Structure

```
LocalizeRAG/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   ├── routers/
│   │   ├── services/
│   │   ├── rag/
│   │   ├── graph/
│   │   ├── llm/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── utils/
│   └── tests/
├── frontend/
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       ├── hooks/
│       └── types/
├── docs/
├── datasets/
└── README.md
```

## Getting Started

### Backend

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Health check: `GET http://localhost:8000/`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

## Future Modules

The following modules are planned for future development:

- **RAG** — Retrieval-augmented generation pipeline
- **Graph** — Knowledge graph construction and querying
- **LLM** — Language model integration and orchestration
- **Routers** — API route definitions
- **Services** — Business logic layer
- **Models / Schemas** — Data models and request/response schemas
