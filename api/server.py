"""HTTP API for the web client.

Three routes and no state. The model is loaded once on the first request and
kept in memory; scoring is arithmetic, so there is nothing to scale beyond
adding processes.

    uvicorn api.server:app --reload
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from anitaste.model import ModelUnavailable
from anitaste.recommend import Ask, InvalidAnswer, questions, recommend, status

app = FastAPI(
    title="anitaste",
    description="Anime recommendations from a model trained on the community "
                "recommendation graph.",
    version="1.0.0",
)

# The web client is served from somewhere else in development. Locked to the
# configured origins rather than "*", so this stays safe to expose.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "ANITASTE_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AskBody(BaseModel):
    answers: dict[str, list[str]] = Field(default_factory=dict)
    text: str = ""
    exclude: list[int] = Field(default_factory=list)
    limit: int = Field(default=6, ge=1, le=24)


@app.get("/api/questions")
def get_questions() -> dict:
    """The questionnaire, plus whether there is a model behind it."""
    report = status()
    return {
        "available": report["available"],
        "questions": questions() if report["available"] else [],
        "titles": report.get("titles", 0),
        "trainedAt": report.get("trainedAt", ""),
    }


@app.get("/api/status")
def get_status() -> dict:
    """Model metadata and the training metrics, for the about panel."""
    return status()


@app.post("/api/recommend")
def post_recommend(body: AskBody) -> dict:
    try:
        results = recommend(
            Ask(
                answers=body.answers,
                text=body.text,
                exclude=frozenset(body.exclude),
                limit=body.limit,
            )
        )
    except InvalidAnswer as error:
        # The caller's fault, and the message says exactly what to fix.
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ModelUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"results": [item.as_dict() for item in results]}
