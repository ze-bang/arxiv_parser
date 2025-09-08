from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException

from ..data.arxiv_client import ArxivClient
from ..models.baseline import load_artifacts, predict_df


app = FastAPI(title="arXiv citation forecast API")


MODEL_PATH = os.getenv("MODEL_PATH", "models/cslg_h12.joblib")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(arxiv_id: str, model_path: Optional[str] = None):
    path = model_path or MODEL_PATH
    try:
        art = load_artifacts(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
    arx = ArxivClient()
    paper = arx.get_by_id(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="arXiv paper not found")
    row = {
        "arxiv_id": paper["arxiv_id"],
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "categories": paper.get("categories", []),
        "authors": paper.get("authors", []),
        "published": dt.datetime.utcnow(),
        "n_authors": len(paper.get("authors", [])),
        "primary_cat": (paper.get("categories", [None])[0] if paper.get("categories") else None),
        "month": dt.datetime.utcnow().month,
    }
    df = pd.DataFrame([row])
    yhat = float(predict_df(df, art)[0])
    return {"arxiv_id": arxiv_id, "pred_citations_12m": yhat}
