import argparse
import datetime as dt

import pandas as pd

from ..data.arxiv_client import ArxivClient
from ..models.baseline import load_artifacts, predict_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv-id", required=True)
    p.add_argument("--model", required=True)
    args = p.parse_args()

    art = load_artifacts(args.model)
    arx = ArxivClient()
    e = arx.get_by_id(args.arxiv_id)
    if not e:
        raise SystemExit("arXiv id not found")
    row = {
        "arxiv_id": e["arxiv_id"],
        "title": e.get("title", ""),
        "abstract": e.get("abstract", ""),
        "categories": e.get("categories", []),
        "authors": e.get("authors", []),
        "published": dt.datetime.utcnow(),  # used for month feature in online prediction
        "n_authors": len(e.get("authors", [])),
        "primary_cat": (e.get("categories", [None])[0] if e.get("categories") else None),
        "month": dt.datetime.utcnow().month,
    }
    df = pd.DataFrame([row])
    yhat = predict_df(df, art)[0]
    print({"arxiv_id": args.arxiv_id, "pred_citations_12m": float(yhat)})


if __name__ == "__main__":
    main()
