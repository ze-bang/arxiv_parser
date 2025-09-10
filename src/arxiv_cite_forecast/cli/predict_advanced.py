import argparse
import datetime as dt

import pandas as pd

from ..data.arxiv_client import ArxivClient
from ..models.advanced import AdvancedArtifacts, predict_with_uncertainty
from ..models.baseline import load_artifacts
from ..explain.shap_utils import ModelExplainer


def main():
    p = argparse.ArgumentParser(description="Advanced prediction with uncertainty and explanations")
    p.add_argument("--arxiv-id", required=True, help="arXiv ID to predict")
    p.add_argument("--model", required=True, help="Path to trained model")
    p.add_argument("--explain", action="store_true", help="Include SHAP explanations")
    p.add_argument("--uncertainty", action="store_true", help="Include uncertainty intervals")
    args = p.parse_args()

    # Load model
    try:
        artifacts = load_artifacts(args.model)
        is_advanced = isinstance(artifacts, AdvancedArtifacts)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return 1

    # Fetch paper
    arx = ArxivClient()
    paper = arx.get_by_id(args.arxiv_id)
    if not paper:
        print(f"Paper {args.arxiv_id} not found")
        return 1

    # Create paper dataframe
    row = {
        'arxiv_id': paper['arxiv_id'],
        'title': paper.get('title', ''),
        'abstract': paper.get('abstract', ''),
        'categories': paper.get('categories', []),
        'authors': paper.get('authors', []),
        'published': dt.datetime.utcnow(),
        'n_authors': len(paper.get('authors', [])),
        'primary_cat': (paper.get('categories', [None])[0] if paper.get('categories') else None),
        'month': dt.datetime.utcnow().month,
    }
    df = pd.DataFrame([row])

    # Make predictions
    result = {'arxiv_id': args.arxiv_id, 'paper': paper}
    
    if is_advanced and args.uncertainty:
        try:
            predictions = predict_with_uncertainty(df, artifacts)
            result['predictions'] = {k: float(v[0]) for k, v in predictions.items()}
        except Exception as e:
            print(f"Uncertainty prediction failed: {e}")
            # Fallback to basic prediction
            from ..models.baseline import predict_df
            pred = predict_df(df, artifacts)[0]
            result['predictions'] = {'mean': float(pred)}
    else:
        # Basic prediction
        if hasattr(artifacts, 'model'):
            from ..models.baseline import predict_df
            pred = predict_df(df, artifacts)[0]
            result['predictions'] = {'mean': float(pred)}
        else:
            print("Cannot make predictions with this model type")
            return 1

    # Add explanations
    if args.explain and hasattr(artifacts, 'feature_names'):
        try:
            # Build feature matrix for explanation
            features = []
            
            # Text features
            if hasattr(artifacts.text_encoder, 'encode_papers'):
                X_text = artifacts.text_encoder.encode_papers(df['title'], df['abstract'])
            else:
                X_text = artifacts.text_encoder.transform(df['title'], df['abstract'])
            features.append(X_text)
            
            # Meta features
            X_cat, X_num = artifacts.meta_encoder.transform(df)
            from scipy import sparse
            features.append(X_cat.toarray() if sparse.issparse(X_cat) else X_cat)
            features.append(X_num)
            
            # Graph features
            if hasattr(artifacts, 'graph_encoder') and artifacts.graph_encoder:
                X_graph = artifacts.graph_encoder.get_paper_graph_features(df)
                features.append(X_graph.values)
            
            import numpy as np
            X = np.hstack(features)
            
            # Create explainer (simplified for demo)
            explainer = ModelExplainer(artifacts.model, artifacts.feature_names)
            explainer.fit_explainer(X)  # Use same sample as background
            
            explanation = explainer.explain_paper(X[0], row)
            result['explanation'] = explanation
            
        except Exception as e:
            print(f"Explanation failed: {e}")
            result['explanation'] = {'error': str(e)}

    # Output results
    import json
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    exit(main())
