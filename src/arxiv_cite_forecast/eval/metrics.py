from __future__ import annotations

import numpy as np
from sklearn.metrics import ndcg_score


def spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    import scipy.stats as st

    coeff, _ = st.spearmanr(y_true, y_pred)
    return float(coeff)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 20) -> float:
    # scikit-learn ndcg expects shape (1, n_samples)
    true = np.asarray(y_true).reshape(1, -1)
    pred = np.asarray(y_pred).reshape(1, -1)
    return float(ndcg_score(true, pred, k=k))
