#!/usr/bin/env python3
"""
Quick script to create demo data for testing advanced features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducible results
np.random.seed(42)
random.seed(42)

# Sample paper data
sample_titles = [
    "Deep Learning for Time Series Forecasting",
    "Transformer Architecture for Natural Language Processing",
    "Graph Neural Networks for Social Network Analysis", 
    "Reinforcement Learning in Robotics Applications",
    "Computer Vision for Medical Image Analysis",
    "Federated Learning with Privacy Preservation",
    "Neural Architecture Search with Evolutionary Algorithms",
    "Attention Mechanisms in Sequence-to-Sequence Models",
    "Adversarial Training for Robust Machine Learning",
    "Meta-Learning for Few-Shot Classification",
    "Explainable AI for Healthcare Decision Making",
    "Quantum Machine Learning Algorithms",
    "Multi-Modal Learning for Video Understanding",
    "Self-Supervised Learning for Representation Learning",
    "Causal Inference in Observational Data",
    "Geometric Deep Learning on Manifolds",
    "Continual Learning without Catastrophic Forgetting",
    "Neural Differential Equations for Time Series",
    "Graph Convolutional Networks for Drug Discovery",
    "Variational Autoencoders for Generative Modeling"
]

sample_abstracts = [
    "This paper presents a novel deep learning approach for time series forecasting that combines attention mechanisms with recurrent neural networks.",
    "We propose a new transformer architecture that improves upon existing models by incorporating specialized attention patterns for natural language tasks.",
    "Our work introduces graph neural networks for analyzing social networks, showing improved performance over traditional methods.",
    "This study explores reinforcement learning applications in robotics, demonstrating effective policy learning in complex environments.",
    "We develop computer vision techniques for medical image analysis, achieving state-of-the-art results on benchmark datasets.",
    "This paper addresses privacy concerns in federated learning by proposing novel differential privacy mechanisms.",
    "We present an evolutionary algorithm approach to neural architecture search that discovers efficient network designs.",
    "Our research focuses on attention mechanisms in sequence-to-sequence models, improving translation quality significantly.",
    "This work develops adversarial training methods that enhance model robustness against various attack strategies.",
    "We propose a meta-learning framework for few-shot classification that generalizes well across different domains.",
    "This paper presents explainable AI techniques specifically designed for healthcare decision support systems.",
    "We explore quantum machine learning algorithms and their potential advantages over classical approaches.",
    "Our work addresses multi-modal learning challenges in video understanding through novel fusion techniques.",
    "This study develops self-supervised learning methods that learn effective representations without labeled data.",
    "We present causal inference methods for observational data that handle confounding variables effectively.",
    "This paper extends deep learning to non-Euclidean domains using geometric deep learning principles.",
    "We address catastrophic forgetting in continual learning through novel regularization techniques.",
    "Our work applies neural differential equations to time series modeling with improved accuracy.",
    "This study uses graph convolutional networks for drug discovery, identifying promising molecular candidates.",
    "We develop variational autoencoders for generative modeling with enhanced sample quality and diversity."
]

sample_authors = [
    ["John Smith", "Alice Johnson"], 
    ["Bob Chen", "Sarah Davis", "Mike Wilson"],
    ["Lisa Wang", "David Brown"],
    ["Emma Garcia", "James Miller", "Anna Taylor"],
    ["Chris Lee", "Maria Rodriguez"],
    ["Kevin Zhang", "Jennifer White", "Michael Kim"],
    ["Sophie Martin", "Ryan Clark", "Ashley Lopez"],
    ["Daniel Thompson", "Rachel Green"],
    ["Alex Turner", "Jessica Yang", "Mark Anderson"],
    ["Laura Chen", "Steven Moore", "Catherine Liu"],
    ["Andrew Davis", "Michelle Kim", "Thomas Wright"],
    ["Nicole Johnson", "Brian Wang", "Jennifer Martinez"],
    ["Patrick Sullivan", "Amanda Chen", "Jason Liu"],
    ["Rebecca Taylor", "Matthew Kim", "Sarah Chang"],
    ["Jonathan Smith", "Emily Rodriguez", "Kevin Park"],
    ["Stephanie Brown", "Anthony Wang", "Diana Lee"],
    ["Christopher Garcia", "Melissa Chen", "Robert Kim"],
    ["Amanda Thompson", "Joseph Martinez", "Nicole Zhang"],
    ["Brandon Davis", "Hannah Kim", "Tyler Chen"],
    ["Samantha Rodriguez", "Nathan Smith", "Ashley Wang"]
]

sample_categories = ["cs.LG", "cs.AI", "cs.CV", "cs.CL", "cs.RO", "stat.ML"]

# Generate synthetic data
n_papers = 100
data = []

base_date = datetime(2023, 1, 1)

for i in range(n_papers):
    # Random submission date in the past 2 years
    days_ago = random.randint(30, 730)  # 30 days to 2 years ago
    pub_date = base_date + timedelta(days=days_ago)
    
    # Select random paper attributes
    title_idx = i % len(sample_titles)
    title = sample_titles[title_idx] + f" - Study {i+1}"
    abstract = sample_abstracts[title_idx] + f" Experimental results show promising performance with method {i+1}."
    authors = sample_authors[i % len(sample_authors)]
    category = random.choice(sample_categories)
    
    # Generate synthetic features
    abstract_len = len(abstract)
    title_len = len(title) 
    n_authors = len(authors)
    
    # Generate citation counts (recent papers have lower citations typically)
    months_since_pub = (datetime.now() - pub_date).days / 30.0
    # Older papers tend to have more citations, with some randomness
    base_citations = max(0, int(np.random.poisson(months_since_pub * 0.5) + np.random.normal(5, 3)))
    
    # Citations at different horizons
    citations_6mo = max(0, int(base_citations * random.uniform(0.2, 0.4)))
    citations_12mo = max(0, int(base_citations * random.uniform(0.5, 0.7)))  
    citations_24mo = base_citations
    
    paper_data = {
        'arxiv_id': f'2023.{i+1:05d}v1',
        'title': title,
        'abstract': abstract,
        'authors': ', '.join(authors),
        'categories': category,
        'published': pub_date.isoformat() + 'Z',
        'abstract_length': abstract_len,
        'title_length': title_len,
        'author_count': n_authors,
        'primary_category': category,
        'primary_cat': category,  # For MetaFeaturizer compatibility
        'month': pub_date.month,  # For MetaFeaturizer compatibility
        'n_authors': n_authors,  # For MetaFeaturizer compatibility
        'citations_6mo': citations_6mo,
        'citations_12mo': citations_12mo, 
        'citations_24mo': citations_24mo
    }
    
    data.append(paper_data)

# Create DataFrame and save
df = pd.DataFrame(data)
print(f"Generated {len(df)} synthetic papers")
print(f"Citation stats - 6mo: {df['citations_6mo'].mean():.1f}, 12mo: {df['citations_12mo'].mean():.1f}, 24mo: {df['citations_24mo'].mean():.1f}")

# Save to parquet
df.to_parquet('data/demo_dataset.parquet', index=False)
print("Saved to data/demo_dataset.parquet")

# Print sample
print("\nSample data:")
print(df[['title', 'authors', 'citations_6mo', 'citations_12mo', 'citations_24mo']].head())
