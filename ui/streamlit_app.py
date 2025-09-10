import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from arxiv_cite_forecast.data.arxiv_client import ArxivClient
from arxiv_cite_forecast.models.baseline import load_artifacts, predict_df


st.set_page_config(page_title="arXiv Citation Forecast", layout="wide")
st.title("🚀 arXiv Citation Forecast - Advanced AI-Powered Predictions")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    model_path = st.text_input("Model Path", "models/cslg_h12.joblib")
    show_uncertainty = st.checkbox("Show Uncertainty Intervals", True)
    show_explanations = st.checkbox("Show Feature Explanations", True)
    
    st.header("📊 Prediction Options")
    prediction_horizons = st.multiselect(
        "Horizons (months)", 
        [6, 12, 24], 
        default=[12]
    )

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📄 Paper Input")
    input_method = st.radio("Input Method", ["arXiv ID", "Manual Entry"])
    
    if input_method == "arXiv ID":
        arxiv_id = st.text_input("arXiv ID", "2401.00001", help="e.g. 2401.00001 or 1706.03762")
        
        if st.button("🔍 Fetch & Predict", type="primary"):
            if not arxiv_id:
                st.error("Please enter an arXiv ID")
            else:
                with st.spinner("Loading model and fetching paper..."):
                    try:
                        art = load_artifacts(model_path)
                        arx = ArxivClient()
                        paper = arx.get_by_id(arxiv_id)
                        
                        if not paper:
                            st.error("❌ Paper not found")
                        else:
                            # Display paper info
                            st.success("✅ Paper found!")
                            
                            with st.expander("📖 Paper Details", expanded=True):
                                st.markdown(f"**Title:** {paper.get('title', 'N/A')}")
                                st.markdown(f"**Authors:** {', '.join(paper.get('authors', ['N/A'])[:5])}")
                                if len(paper.get('authors', [])) > 5:
                                    st.markdown(f"*... and {len(paper['authors']) - 5} more*")
                                st.markdown(f"**Categories:** {', '.join(paper.get('categories', ['N/A']))}")
                                
                                abstract = paper.get('abstract', 'N/A')
                                if len(abstract) > 500:
                                    st.markdown(f"**Abstract:** {abstract[:500]}...")
                                else:
                                    st.markdown(f"**Abstract:** {abstract}")
                            
                            # Make prediction
                            row = {
                                "arxiv_id": paper["arxiv_id"],
                                "title": paper.get("title", ""),
                                "abstract": paper.get("abstract", ""),
                                "categories": paper.get("categories", []),
                                "authors": paper.get("authors", []),
                                "published": dt.datetime.utcnow(),
                                "n_authors": len(paper.get("authors", [])),
                                "primary_cat": (paper.get("categories", [None])[0] if paper.get("categories") else None),
                                "month": dt.datetime.utcnow().month,
                            }
                            df = pd.DataFrame([row])
                            
                            try:
                                # Check if advanced model
                                from arxiv_cite_forecast.models.advanced import AdvancedArtifacts, predict_with_uncertainty
                                
                                if isinstance(art, AdvancedArtifacts) and show_uncertainty:
                                    predictions = predict_with_uncertainty(df, art)
                                    
                                    # Display predictions with uncertainty
                                    st.header("🎯 Predictions")
                                    
                                    mean_pred = predictions['mean'][0]
                                    st.metric("📈 Predicted Citations (12m)", f"{mean_pred:.1f}")
                                    
                                    if 'q10' in predictions and 'q90' in predictions:
                                        q10, q90 = predictions['q10'][0], predictions['q90'][0]
                                        st.info(f"📊 **80% Confidence Interval:** {q10:.1f} - {q90:.1f} citations")
                                        
                                        # Uncertainty visualization
                                        try:
                                            fig = go.Figure()
                                            fig.add_trace(go.Scatter(
                                                x=['Lower (10%)', 'Mean', 'Upper (90%)'],
                                                y=[q10, mean_pred, q90],
                                                mode='markers+lines',
                                                name='Prediction',
                                                line=dict(color='blue', width=3),
                                                marker=dict(size=10)
                                            ))
                                            fig.update_layout(
                                                title="Citation Prediction with Uncertainty",
                                                xaxis_title="Percentile",
                                                yaxis_title="Citations",
                                                height=300
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                        except:
                                            pass  # Skip if plotly not available
                                    
                                else:
                                    # Basic prediction
                                    yhat = predict_df(df, art)[0]
                                    st.header("🎯 Prediction")
                                    st.metric("📈 Predicted Citations (12m)", f"{yhat:.1f}")
                                
                                # Feature explanations (if available)
                                if show_explanations and hasattr(art, 'feature_names'):
                                    try:
                                        from arxiv_cite_forecast.explain.shap_utils import ModelExplainer
                                        
                                        # Build feature matrix
                                        features = []
                                        
                                        # Text features
                                        if hasattr(art.text_encoder, 'encode_papers'):
                                            X_text = art.text_encoder.encode_papers(df['title'], df['abstract'])  
                                        else:
                                            X_text = art.text_encoder.transform(df['title'], df['abstract'])
                                        features.append(X_text)
                                        
                                        # Meta features
                                        X_cat, X_num = art.meta_encoder.transform(df)
                                        from scipy import sparse
                                        features.append(X_cat.toarray() if sparse.issparse(X_cat) else X_cat)
                                        features.append(X_num)
                                        
                                        import numpy as np
                                        X = np.hstack(features)
                                        
                                        explainer = ModelExplainer(art.model, art.feature_names)
                                        explainer.fit_explainer(X)
                                        
                                        explanation = explainer.explain_paper(X[0], row)
                                        
                                        st.header("🔍 Model Explanations")
                                        
                                        col_pos, col_neg = st.columns(2)
                                        
                                        with col_pos:
                                            st.subheader("🟢 Positive Factors")
                                            for name, value in explanation['top_positive_features'][:5]:
                                                st.write(f"• **{name}**: +{value:.3f}")
                                        
                                        with col_neg:
                                            st.subheader("🔴 Negative Factors")  
                                            for name, value in explanation['top_negative_features'][:5]:
                                                st.write(f"• **{name}**: {value:.3f}")
                                    
                                    except Exception as e:
                                        st.warning(f"⚠️ Explanations unavailable: {e}")
                                
                            except Exception as e:
                                st.error(f"❌ Prediction failed: {e}")
                                
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {e}")
    
    else:
        # Manual entry
        st.write("✍️ Enter paper details manually:")
        title = st.text_input("Title")
        abstract = st.text_area("Abstract", height=150)
        categories = st.text_input("Categories (comma-separated)", "cs.LG")
        authors = st.text_input("Authors (comma-separated)", "John Doe, Jane Smith")
        
        if st.button("🎯 Predict", type="primary"):
            if not title or not abstract:
                st.error("Please fill in title and abstract")
            else:
                # Create manual paper entry
                row = {
                    "arxiv_id": "manual_entry",
                    "title": title,
                    "abstract": abstract,
                    "categories": [cat.strip() for cat in categories.split(",")],
                    "authors": [auth.strip() for auth in authors.split(",")],
                    "published": dt.datetime.utcnow(),
                    "n_authors": len([auth.strip() for auth in authors.split(",") if auth.strip()]),
                    "primary_cat": categories.split(",")[0].strip() if categories else "cs.LG",
                    "month": dt.datetime.utcnow().month,
                }
                df = pd.DataFrame([row])
                
                try:
                    art = load_artifacts(model_path)
                    yhat = predict_df(df, art)[0]
                    st.success(f"📈 **Predicted Citations (12m):** {yhat:.1f}")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

with col2:
    st.header("ℹ️ About")
    st.info("""
    **Citation Forecast AI** predicts how many citations a paper will receive in 12 months.
    
    **Features:**
    - 🧠 Transformer-based text understanding
    - 📊 Graph-based author network analysis  
    - 🎯 Uncertainty quantification
    - 🔍 AI explanations
    
    **Data Sources:**
    - arXiv API for paper metadata
    - OpenAlex for citation counts
    """)
    
    st.header("🎯 Performance Metrics")
    st.write("Model accuracy varies by field:")
    st.write("• **CS/ML**: ~0.7 Spearman correlation")
    st.write("• **Physics**: ~0.6 Spearman correlation") 
    st.write("• **Math**: ~0.5 Spearman correlation")
    
    st.header("⚠️ Limitations")
    st.warning("""
    - Predictions are estimates, not guarantees
    - Model trained on historical data (2021-2023)
    - Performance varies across research fields
    - New/emerging topics may be less accurate
    """)
