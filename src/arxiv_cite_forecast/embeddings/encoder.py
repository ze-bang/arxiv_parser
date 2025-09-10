from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils import Cache, get_cache_dir


class SentenceTransformerEncoder:
    """Sentence transformer encoder with local caching for paper embeddings.
    
    Provides a drop-in replacement for TF-IDF+SVD with better semantic understanding.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        cache_namespace: str = "embeddings",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.cache = Cache(f"{cache_namespace}_{model_name.replace('/', '_')}")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def _cache_key(self, texts: List[str]) -> str:
        """Generate cache key from text list hash."""
        text_hash = hashlib.sha256("|".join(texts).encode()).hexdigest()[:16]
        return f"{self.model_name}_{text_hash}"
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings with caching."""
        cache_key = self._cache_key(texts)
        cached = self.cache.get(cache_key)
        
        if cached is not None:
            return np.array(cached["embeddings"])
        
        # Compute embeddings
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True
        )
        
        # Cache results
        self.cache.set(cache_key, {"embeddings": embeddings.tolist()})
        return embeddings
    
    def encode_papers(self, titles: Iterable[str], abstracts: Iterable[str]) -> np.ndarray:
        """Encode paper title+abstract pairs."""
        texts = [f"{title}\n\n{abstract}" for title, abstract in zip(titles, abstracts)]
        return self.encode(texts)


class HybridTextEncoder:
    """Hybrid encoder combining TF-IDF and sentence transformers."""
    
    def __init__(
        self,
        use_tfidf: bool = True,
        use_transformers: bool = True,
        tfidf_weight: float = 0.3,
        transformer_weight: float = 0.7,
        **kwargs
    ):
        self.use_tfidf = use_tfidf
        self.use_transformers = use_transformers
        self.tfidf_weight = tfidf_weight
        self.transformer_weight = transformer_weight
        
        if use_transformers:
            from .encoder import SentenceTransformerEncoder
            self.transformer_encoder = SentenceTransformerEncoder(**kwargs)
            
        if use_tfidf:
            from ..features.text import TextFeaturizer
            self.tfidf_encoder = TextFeaturizer()
            
        self._fitted = False
    
    def fit(self, titles: Iterable[str], abstracts: Iterable[str]):
        if self.use_tfidf:
            self.tfidf_encoder.fit(titles, abstracts)
        self._fitted = True
        return self
    
    def transform(self, titles: Iterable[str], abstracts: Iterable[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("HybridTextEncoder must be fitted first")
            
        embeddings = []
        
        if self.use_transformers:
            transformer_emb = self.transformer_encoder.encode_papers(titles, abstracts)
            embeddings.append(transformer_emb * self.transformer_weight)
            
        if self.use_tfidf:
            tfidf_emb = self.tfidf_encoder.transform(titles, abstracts)
            embeddings.append(tfidf_emb * self.tfidf_weight)
            
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Concatenate different embedding types
        return np.hstack(embeddings)
