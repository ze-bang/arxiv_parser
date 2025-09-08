from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class MetaFeaturizer:
    def __init__(self):
        # scikit-learn >=1.4 uses sparse_output; keep sparse output for hstack efficiency
        try:
            self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            # fallback for older versions
            self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        self.scaler = StandardScaler(with_mean=False)
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        X_cat = df[["primary_cat", "month"]].astype("category")
        self.cat_encoder.fit(X_cat)
        X_num = df[["n_authors"]].astype(float)
        self.scaler.fit(X_num)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("MetaFeaturizer must be fitted before transform().")
        X_cat = self.cat_encoder.transform(df[["primary_cat", "month"]].astype("category"))
        X_num = self.scaler.transform(df[["n_authors"]].astype(float))
        return X_cat, X_num

    @staticmethod
    def hstack_sparse_dense(sparse_mtx, dense_arr):
        from scipy import sparse

        if not sparse.issparse(sparse_mtx):
            raise ValueError("sparse_mtx must be a scipy.sparse matrix")
        if dense_arr.ndim == 1:
            dense_arr = dense_arr.reshape(-1, 1)
        return sparse.hstack([sparse_mtx, dense_arr], format="csr")
