import argparse
from pathlib import Path

import pandas as pd

from ..models.baseline import save_artifacts, train_baseline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Parquet dataset path")
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--model-out", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    artifacts, metrics = train_baseline(df)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    save_artifacts(artifacts, args.model_out)
    print({"metrics": metrics, "model": args.model_out})


if __name__ == "__main__":
    main()
