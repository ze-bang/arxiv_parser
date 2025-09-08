import argparse
from pathlib import Path

import pandas as pd

from ..data.dataset import build_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    df = build_dataset(args.category, args.start_date, args.end_date, limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
