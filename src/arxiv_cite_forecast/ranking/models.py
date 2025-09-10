from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def create_ranking_groups(df: pd.DataFrame, group_by: str = 'submission_week') -> np.ndarray:
    """Create ranking groups for pairwise training (e.g., by submission week)."""
    if group_by == 'submission_week':
        df = df.copy()
        df['pub_date'] = pd.to_datetime(df['published'])
        df['submission_week'] = df['pub_date'].dt.to_period('W')
        
        # Encode groups as integers
        unique_weeks = df['submission_week'].unique()
        week_to_id = {week: i for i, week in enumerate(unique_weeks)}
        groups = df['submission_week'].map(week_to_id).values
        
    elif group_by == 'category':
        unique_cats = df['primary_cat'].unique()
        cat_to_id = {cat: i for i, cat in enumerate(unique_cats) if pd.notna(cat)}
        groups = df['primary_cat'].map(lambda x: cat_to_id.get(x, -1)).values
        
    else:
        # Single group (rank all together)
        groups = np.zeros(len(df))
    
    return groups


def train_ranking_model(
    X: np.ndarray,
    y: np.ndarray, 
    groups: np.ndarray,
    random_state: int = 42
):
    """Train LightGBM ranker model."""
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM required for ranking models")
    
    # Compute group sizes for LightGBM
    unique_groups = np.unique(groups)
    group_sizes = [np.sum(groups == g) for g in unique_groups]
    
    ranker = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        boosting_type='gbdt',
        num_leaves=31,
        learning_rate=0.1,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=random_state
    )
    
    ranker.fit(X, y, group=group_sizes)
    return ranker


def evaluate_ranking_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    groups: np.ndarray,
    k_values: List[int] = [10, 20, 50]
) -> Dict[str, float]:
    """Evaluate ranking metrics including NDCG@K and Spearman correlation."""
    metrics = {}
    
    # Overall Spearman correlation
    spearman_corr, _ = spearmanr(y_true, y_pred)
    metrics['spearman_overall'] = spearman_corr
    
    # NDCG scores
    for k in k_values:
        try:
            # Compute NDCG@k per group and average
            group_ndcgs = []
            unique_groups = np.unique(groups)
            
            for g in unique_groups:
                mask = groups == g
                if np.sum(mask) >= k:  # Need at least k samples
                    y_true_g = y_true[mask].reshape(1, -1)
                    y_pred_g = y_pred[mask].reshape(1, -1)
                    ndcg_k = ndcg_score(y_true_g, y_pred_g, k=k)
                    group_ndcgs.append(ndcg_k)
            
            if group_ndcgs:
                metrics[f'ndcg@{k}'] = np.mean(group_ndcgs)
            else:
                metrics[f'ndcg@{k}'] = 0.0
                
        except Exception as e:
            print(f"Warning: Could not compute NDCG@{k}: {e}")
            metrics[f'ndcg@{k}'] = 0.0
    
    # Precision@K (fraction of top-K predictions that are in top-K ground truth)
    for k in k_values:
        if len(y_true) >= k:
            top_k_true_idx = np.argsort(y_true)[-k:]
            top_k_pred_idx = np.argsort(y_pred)[-k:]
            precision_k = len(np.intersect1d(top_k_true_idx, top_k_pred_idx)) / k
            metrics[f'precision@{k}'] = precision_k
        else:
            metrics[f'precision@{k}'] = 0.0
    
    return metrics


def rank_papers_by_impact_potential(
    df: pd.DataFrame,
    predictions: np.ndarray,
    top_k: int = 50
) -> pd.DataFrame:
    """Rank papers by predicted impact and return top-K."""
    df = df.copy()
    df['predicted_citations'] = predictions
    df['impact_rank'] = df['predicted_citations'].rank(ascending=False)
    
    # Select top-K
    top_papers = df.nsmallest(top_k, 'impact_rank')
    
    # Add percentile scores
    top_papers['impact_percentile'] = (
        (len(df) - top_papers['impact_rank'] + 1) / len(df) * 100
    )
    
    return top_papers[['arxiv_id', 'title', 'primary_cat', 'predicted_citations', 
                     'impact_rank', 'impact_percentile']].reset_index(drop=True)
