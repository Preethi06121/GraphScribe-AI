import logging

from fastapi import FastAPI

from app.routers import content, documents, graph, retrieval, research

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="LocalizeRAG")

app.include_router(documents.router)
app.include_router(graph.router)
app.include_router(retrieval.router)
app.include_router(content.router)
app.include_router(research.router)


@app.get("/")
def health_check():
    return {"project": "LocalizeRAG", "status": "running"}
