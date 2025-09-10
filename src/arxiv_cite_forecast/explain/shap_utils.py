from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


class ModelExplainer:
    """SHAP-based model explainer for citation prediction models."""
    
    def __init__(self, model: BaseEstimator, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit_explainer(self, X_background: np.ndarray, max_evals: int = 100):
        """Fit SHAP explainer on background dataset."""
        # Use TreeExplainer for tree-based models, KernelExplainer as fallback
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except:
            # Fallback to kernel explainer with sampled background
            if len(X_background) > max_evals:
                background_sample = shap.sample(X_background, max_evals)
            else:
                background_sample = X_background
            self.explainer = shap.KernelExplainer(self.model.predict, background_sample)
    
    def explain_predictions(self, X: np.ndarray, max_samples: int = 100) -> np.ndarray:
        """Compute SHAP values for predictions."""
        if self.explainer is None:
            raise ValueError("Must call fit_explainer first")
            
        # Sample data if too large
        if len(X) > max_samples:
            X_sample = X[:max_samples]
        else:
            X_sample = X
            
        self.shap_values = self.explainer.shap_values(X_sample)
        return self.shap_values
    
    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get global feature importance from SHAP values."""
        if self.shap_values is None:
            self.explain_predictions(X)
            
        # Mean absolute SHAP values
        importance = np.mean(np.abs(self.shap_values), axis=0)
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(importance)}
    
    def explain_paper(self, X_paper: np.ndarray, paper_info: Dict[str, Any]) -> Dict[str, Any]:
        """Explain prediction for a single paper."""
        if self.explainer is None:
            raise ValueError("Must call fit_explainer first")
            
        shap_vals = self.explainer.shap_values(X_paper.reshape(1, -1))[0]
        prediction = self.model.predict(X_paper.reshape(1, -1))[0]
        
        # Top positive and negative contributors
        if self.feature_names:
            feature_impacts = list(zip(self.feature_names, shap_vals))
        else:
            feature_impacts = [(f"feature_{i}", val) for i, val in enumerate(shap_vals)]
            
        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'prediction': float(prediction),
            'baseline': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0,
            'top_positive_features': [(name, float(val)) for name, val in feature_impacts if val > 0][:10],
            'top_negative_features': [(name, float(val)) for name, val in feature_impacts if val < 0][:10],
            'paper_info': paper_info
        }


def create_feature_names(text_dim: int, has_graph_features: bool = False) -> List[str]:
    """Create human-readable feature names for different components."""
    names = []
    
    # Text embedding features
    names.extend([f"text_emb_{i}" for i in range(text_dim)])
    
    # Meta features
    names.extend(['n_authors', 'month'])
    
    # One-hot encoded categories (approximate)
    names.extend(['cat_cs_LG', 'cat_cs_AI', 'cat_other'])
    
    # Graph features (if enabled)
    if has_graph_features:
        names.extend([
            'mean_degree_centrality', 'max_degree_centrality',
            'mean_pagerank', 'max_pagerank', 
            'mean_eigenvector_centrality', 'max_eigenvector_centrality',
            'mean_clustering', 'max_clustering',
            'author_graph_diversity', 'coauthor_network_size'
        ])
    
    return names
