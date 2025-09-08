import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

from arxiv_cite_forecast.data.arxiv_client import ArxivClient
from arxiv_cite_forecast.models.baseline import load_artifacts, predict_df


st.set_page_config(page_title="arXiv 12m citation forecast", layout="wide")
st.title("arXiv 12-month citation forecast (demo)")

model_path = st.text_input("Path to trained model (.joblib)", "models/cslg_h12.joblib")
arxiv_id = st.text_input("arXiv ID", "2401.00001")

if st.button("Predict"):
    try:
        art = load_artifacts(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
    else:
        arx = ArxivClient()
        paper = arx.get_by_id(arxiv_id)
        if not paper:
            st.error("Paper not found")
        else:
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
            try:
                yhat = predict_df(df, art)[0]
                st.success(f"Predicted citations in 12m: {yhat:.2f}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
