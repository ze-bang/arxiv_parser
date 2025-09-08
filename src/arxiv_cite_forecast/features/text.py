from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


class TextFeaturizer:
    """TF-IDF over title+abstract followed by SVD to a compact embedding.

    Designed to be light-weight for a demo without heavy GPU models.
    """

    def __init__(self, max_features: int = 20000, n_components: int = 256, min_df: int = 1, random_state: int = 42):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df)
        self.desired_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, titles: Iterable[str], abstracts: Iterable[str]):
        docs = [f"{t} \n {a}" for t, a in zip(titles, abstracts)]
        try:
            X = self.vectorizer.fit_transform(docs)
        except ValueError as e:
            # Handle tiny datasets where min_df prunes all tokens
            if "After pruning, no terms remain" in str(e):
                self.vectorizer = TfidfVectorizer(
                    max_features=self.vectorizer.max_features,
                    ngram_range=self.vectorizer.ngram_range,
                    min_df=1,
                )
                X = self.vectorizer.fit_transform(docs)
            else:
                raise
        # Ensure n_components <= n_features
        n_feat = X.shape[1]
        n_comp = max(1, min(self.desired_components, n_feat))
        if self.svd.n_components != n_comp:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.svd.random_state)
        self.svd.fit(X)
        self._fitted = True
        return self

    def transform(self, titles: Iterable[str], abstracts: Iterable[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TextFeaturizer must be fitted before transform().")
        docs = [f"{t} \n {a}" for t, a in zip(titles, abstracts)]
        X = self.vectorizer.transform(docs)
        Z = self.svd.transform(X)
        # Pad with zeros to maintain desired dimensionality for downstream models/tests
        if Z.shape[1] < self.desired_components:
            pad = self.desired_components - Z.shape[1]
            Z = np.hstack([Z, np.zeros((Z.shape[0], pad), dtype=Z.dtype)])
        return Z
