from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class AuthorGraphBuilder:
    """Build co-authorship graphs and extract centrality features."""
    
    def __init__(self, cutoff_date: Optional[dt.datetime] = None):
        self.cutoff_date = cutoff_date
        self.graph = nx.Graph()
        self.author_stats = {}
        
    def build_from_papers(self, papers_df: pd.DataFrame) -> nx.Graph:
        """Build co-authorship graph from papers dataframe."""
        author_papers = defaultdict(list)
        paper_authors = {}
        
        for idx, row in papers_df.iterrows():
            if pd.isna(row.get('published')):
                continue
                
            pub_date = pd.to_datetime(row['published'])
            if self.cutoff_date and pub_date > self.cutoff_date:
                continue
                
            authors = row.get('authors', [])
            if not authors or len(authors) < 2:
                continue
                
            paper_id = row.get('arxiv_id', idx)
            paper_authors[paper_id] = authors
            
            # Track papers per author
            for author in authors:
                author_papers[author].append((paper_id, pub_date))
        
        # Build co-authorship edges
        edge_weights = defaultdict(int)
        
        for paper_id, authors in paper_authors.items():
            for i, author1 in enumerate(authors):
                for author2 in authors[i+1:]:
                    edge = tuple(sorted([author1, author2]))
                    edge_weights[edge] += 1
        
        # Create graph
        self.graph = nx.Graph()
        for (author1, author2), weight in edge_weights.items():
            self.graph.add_edge(author1, author2, weight=weight)
            
        # Add isolated nodes for single-author papers
        for author, papers in author_papers.items():
            if author not in self.graph:
                self.graph.add_node(author)
                
        return self.graph
    
    def compute_centralities(self) -> Dict[str, Dict[str, float]]:
        """Compute various centrality measures for all authors."""
        if not self.graph.nodes():
            return {}
            
        centralities = {}
        
        # Degree centrality (normalized)
        centralities['degree'] = nx.degree_centrality(self.graph)
        
        # PageRank (handles disconnected components)
        try:
            centralities['pagerank'] = nx.pagerank(self.graph, weight='weight')
        except:
            centralities['pagerank'] = {node: 0.0 for node in self.graph.nodes()}
            
        # Eigenvector centrality (only for largest connected component)
        try:
            if nx.is_connected(self.graph):
                centralities['eigenvector'] = nx.eigenvector_centrality(self.graph, weight='weight')
            else:
                # Use largest connected component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                eig_cent = nx.eigenvector_centrality(subgraph, weight='weight')
                centralities['eigenvector'] = {node: eig_cent.get(node, 0.0) for node in self.graph.nodes()}
        except:
            centralities['eigenvector'] = {node: 0.0 for node in self.graph.nodes()}
            
        # Clustering coefficient
        centralities['clustering'] = nx.clustering(self.graph, weight='weight')
        
        self.author_stats = centralities
        return centralities
    
    def get_paper_graph_features(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """Extract graph-based features for papers."""
        if not self.author_stats:
            self.compute_centralities()
            
        features = []
        
        for idx, row in papers_df.iterrows():
            authors = row.get('authors', [])
            
            if not authors:
                # No authors - use zeros
                paper_features = {
                    'mean_degree_centrality': 0.0,
                    'max_degree_centrality': 0.0,
                    'mean_pagerank': 0.0,
                    'max_pagerank': 0.0,
                    'mean_eigenvector_centrality': 0.0,
                    'max_eigenvector_centrality': 0.0,
                    'mean_clustering': 0.0,
                    'max_clustering': 0.0,
                    'author_graph_diversity': 0.0,
                    'coauthor_network_size': 0,
                }
            else:
                # Aggregate centrality stats across authors
                degree_vals = [self.author_stats.get('degree', {}).get(a, 0.0) for a in authors]
                pagerank_vals = [self.author_stats.get('pagerank', {}).get(a, 0.0) for a in authors]
                eigen_vals = [self.author_stats.get('eigenvector', {}).get(a, 0.0) for a in authors]
                cluster_vals = [self.author_stats.get('clustering', {}).get(a, 0.0) for a in authors]
                
                # Count unique co-authors
                coauthor_set = set()
                for author in authors:
                    if author in self.graph:
                        coauthor_set.update(self.graph.neighbors(author))
                
                paper_features = {
                    'mean_degree_centrality': np.mean(degree_vals),
                    'max_degree_centrality': np.max(degree_vals),
                    'mean_pagerank': np.mean(pagerank_vals),
                    'max_pagerank': np.max(pagerank_vals),
                    'mean_eigenvector_centrality': np.mean(eigen_vals),
                    'max_eigenvector_centrality': np.max(eigen_vals),
                    'mean_clustering': np.mean(cluster_vals),
                    'max_clustering': np.max(cluster_vals),
                    'author_graph_diversity': np.std(degree_vals) if len(degree_vals) > 1 else 0.0,
                    'coauthor_network_size': len(coauthor_set),
                }
            
            features.append(paper_features)
        
        return pd.DataFrame(features)


def compute_field_normalized_targets(df: pd.DataFrame, target_col: str = 'y12') -> pd.DataFrame:
    """Compute field and year normalized citation targets."""
    df = df.copy()
    
    # Extract year from published date
    df['pub_year'] = pd.to_datetime(df['published']).dt.year
    
    # Compute field-year medians
    field_year_medians = df.groupby(['primary_cat', 'pub_year'])[target_col].median()
    
    # Normalize targets
    normalized_targets = []
    for idx, row in df.iterrows():
        field = row.get('primary_cat')
        year = row.get('pub_year')
        target = row.get(target_col, 0)
        
        if pd.isna(field) or pd.isna(year):
            normalized_targets.append(0.0)
        else:
            field_median = field_year_medians.get((field, year), 1.0)
            # Add small epsilon to avoid division by zero
            normalized = target / (field_median + 0.1)
            normalized_targets.append(normalized)
    
    df[f'{target_col}_normalized'] = normalized_targets
    return df
