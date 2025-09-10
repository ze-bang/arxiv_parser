# arxiv-citation-forecast

Predict 12-month citations for newly submitted arXiv papers using text and metadata. Includes:
- Data loaders for arXiv and OpenAlex with local caching
- Feature pipeline (TF-IDF + SVD text embeddings, meta features)
- Baseline model (scikit-learn HistGradientBoostingRegressor)
- Time-based evaluation and metrics
- CLI tools to build datasets, train, and predict
- Streamlit demo app

## Quick start

Prereqs: Python 3.10+

1) Create a virtual environment and install deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure your User-Agent (recommended by OpenAlex) via `.env`:

```bash
cp .env.example .env
# edit .env to set CONTACT_EMAIL to your email (used in API User-Agent)
```

3) Build a dataset (example: cs.LG papers submitted in 2021-2022 with valid 12m targets)

```bash
python -m arxiv_cite_forecast.cli.build_dataset \
  --category cs.LG \
  --start-date 2021-01-01 \
  --end-date 2022-12-31 \
  --limit 500 \
  --out data/datasets/csLG_2021_2022.parquet
```

4) Train a model

```bash
python -m arxiv_cite_forecast.cli.train \
  --data data/datasets/csLG_2021_2022.parquet \
  --horizon 12 \
  --model-out models/cslg_h12.joblib
```

5) Predict for a paper by arXiv ID

```bash
python -m arxiv_cite_forecast.cli.predict \
  --arxiv-id 2401.00001 \
  --model models/cslg_h12.joblib
```

6) Streamlit demo

```bash
streamlit run ui/streamlit_app.py
```

## Notes
- Targets are defined as citations within N months after submission using OpenAlex counts_by_year. We compute 12m targets if at least 365 days have passed since submission.
- To avoid data leakage, the feature pipeline uses only submission-time metadata (title, abstract, categories, author count, submission month).
- APIs are rate-limited; responses are cached under `data_cache/`.

## State‑of‑the‑Art Enhancement Roadmap (2025)

These are incremental, resume‑worthy upgrades aligned with current citation / scholarly impact prediction research trends (drawing on recent 2025 work leveraging OpenAlex scale, transformer text embeddings, and graph representations of collaboration networks):

### 1. Rich Text & Semantic Features
- Replace TF‑IDF+SVD with sentence transformer embeddings (e.g. `sentence-transformers/all-MiniLM-L12-v2` for speed; upgrade to `bge-base-en-v1.5` or domain models).
- Concatenate title + abstract embedding plus (optionally) concept / keyword embeddings extracted via zero‑shot classification.
- Use embedding similarity to a curated set of historically high‑impact papers ("proximity-to-impact" feature: mean/max cosine similarity to top decile prior papers by field/year).

### 2. Author & Institution Graph Features
- Build a co‑authorship graph (nodes=authors, edges=co-auth counts, weighted, time-filtered pre-cutoff).
- Compute fast graph statistics per paper (mean/median author PageRank, eigenvector centrality, clustering coefficient, author h-index proxy from historical citations, institution diversity index).
- (Stretch) Train a lightweight GNN (GraphSAGE / GAT) over the author-paper bipartite graph to produce author embeddings; aggregate (mean + attention pooling) into paper representation and fuse with text embedding.

### 3. Field / Category Normalization
- Normalize raw target with field & year baselines: `norm_y = y12 / (epsilon + median_y12[primary_cat, year])` for cross-field comparability; optionally predict both raw and normalized then invert.
- Add entropy / interdisciplinarity signals: number of distinct primary OpenAlex concepts in top-K, concept entropy, distance between arXiv category and OpenAlex concept cluster centroid.

### 4. Advanced Target Modeling
- Zero-Inflated Negative Binomial (ZINB) or hurdle models to better handle citation zero-inflation.
- Multi-horizon multi-task head (predict 6m, 12m, 24m jointly) to share representation and improve early horizon stability.
- Quantile regression (LightGBM quantile or NGBoost) + Conformal calibration to produce well-calibrated prediction intervals (coverage metric reported).

### 5. Ranking & Discovery Objectives
- Complement regression loss with LambdaRank / pairwise loss for ordering papers within the same weekly submission cohort.
- Evaluate NDCG@K (K=20/50/100) and Precision@TopK for identifying potential high-impact papers.

### 6. Similarity & Neighborhood Signals
- kNN over embedding space: aggregate historical citation stats of top-N nearest older papers (median, max, variance); include recency-weighted neighbor citation growth slope.
- Expert proximity: compute maximum author similarity to a set of domain experts (predefined by prolific/highly cited author ids from pre-training period).

### 7. Temporal Drift Handling
- Rolling retraining every quarter; maintain model registry with performance decay tracking.
- Feature drift detection (e.g., population stability index (PSI) on core numeric features and embedding norm distribution) to trigger refresh.

### 8. Evaluation Enhancements
- Add calibration curves for prediction intervals.
- Slice metrics: early-career (authors all < N prior papers), single-institution vs multi, cross-category generalization.
- Report both raw and field-normalized R^2 / Spearman.

### 9. Reproducibility & MLOps polish
- Hydra or Pydantic settings for experiment configs.
- MLflow or Weights & Biases tracking (params, metrics, artifacts, model lineage).
- GitHub Actions: run tests + lint + small smoke train on a tiny synthetic dataset.
- Model card (motivation, data scope, limitations, bias considerations, leakage checks) + data card.

### 10. Deployment / Serving
- Batch scoring script to forecast for newest weekly submissions (cron / GitHub Action schedule) and push JSON to a lightweight static dashboard.
- FastAPI endpoint extension: `/explain` returning top SHAP/global importance stats (compute once per model, cache).

## Suggested Folder Extensions
```
src/arxiv_cite_forecast/
  graph/            # graph extraction & centrality/GNN utilities
  embeddings/       # sentence-transformer inference & caching
  ranking/          # pairwise/pointwise ranking trainers
  explain/          # SHAP / feature importance wrappers
  workflows/        # scheduled batch jobs / rolling retrain scripts
configs/            # hydra-style experiment configs
mlruns/             # (gitignored) MLflow tracking store
model_registry/     # versioned serialized models + metadata
```

## Incremental Implementation Plan
1. Swap text backend to sentence-transformers with on-disk embedding cache.
2. Add author graph extraction (pre-cutoff) + centrality features.
3. Introduce field-normalized target variant; train dual-head model.
4. Implement LightGBM quantile + conformal post-processing for intervals.
5. Add ranking objective training path (e.g., LightGBM ranker by weekly cohort).
6. GNN prototype (GraphSAGE) for author embeddings; ablation to show lift.
7. Calibration & SHAP reporting utilities; integrate into Streamlit panel.
8. Rolling window evaluation script with drift metrics & HTML report.
9. CI workflow + MLflow tracking; produce model card on each release.
10. Scheduled batch job to score latest submissions and publish artifact.

## High-Value Ablation Ideas
- Text model: TF-IDF vs sentence transformer vs hybrid
- Add/remove author graph features
- Neighbor citation features on/off
- Field normalization vs raw target
- Multi-task (6/12/24m) vs single horizon

## Risk & Mitigation Summary
| Risk | Impact | Mitigation |
|------|--------|-----------|
| API rate limits | Slow ingestion | Caching, exponential backoff, parallel buckets with polite sleep |
| Author disambiguation noise | Feature dilution | Use OpenAlex author IDs only; drop ambiguous (missing id) authors |
| Leakage (post-submission signals) | Inflated metrics | Strict cutoff snapshots; assert feature timestamps ≤ submission |
| Field imbalance | Biased ranking | Field-normalized targets + stratified metrics |
| Zero inflation | Poor regression fit | ZINB / hurdle loss + quantile intervals |

## Minimal Next Step (If You Continue Now)
Implement Step 1 (sentence-transformer embeddings) by adding an `embeddings/encoder.py` that wraps a model from `sentence-transformers` with local caching, then swap it into the training pipeline (keeping TF-IDF as a fallback flag).


## Project structure

- src/arxiv_cite_forecast/
  - data/: API clients and dataset builder
  - features/: feature engineering
  - models/: baseline models
  - eval/: metrics and backtesting helpers
  - cli/: command-line tools
- ui/streamlit_app.py: interactive demo
- tests/: unit tests
- data_cache/: API cache (gitignored)
- data/datasets/: saved datasets (gitignored)

## License
MIT