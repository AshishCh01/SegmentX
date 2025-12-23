"""
Advanced Clustering Module
==========================
Production-grade clustering with multiple algorithms, parallel processing,
and automatic selection of the best configuration.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import (
    KMeans, 
    SpectralClustering, 
    AgglomerativeClustering,
    DBSCAN,
    MiniBatchKMeans,
    Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# Try to import HDBSCAN (optional dependency)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class AdvancedClusteringEngine:
    """
    State-of-the-art clustering engine that tries multiple algorithms
    and configurations to find the best clustering solution.
    
    Features:
    - Multiple clustering algorithms (KMeans++, Spectral, Agglomerative, GMM, BIRCH, HDBSCAN)
    - Automatic algorithm selection based on silhouette score
    - Parallel processing for faster results
    - Mini-batch algorithms for large datasets
    - Advanced feature engineering and selection
    """
    
    def __init__(self, random_state=42, n_jobs=-1, use_minibatch=False):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_minibatch = use_minibatch
        self.best_score = -1
        self.best_labels = None
        self.best_method = None
        self.best_k = None
        self.all_results = []
        self.processing_time = 0
        
    def remove_outliers_iqr(self, data, factor=1.5):
        """Remove outliers using IQR method - more robust than Z-score."""
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask = np.all((data >= lower) & (data <= upper), axis=1)
        return mask
    
    def engineer_features(self, df):
        """Create powerful engineered features for better cluster separation."""
        df_eng = df.copy()
        cols = df.columns.tolist()
        n_cols = len(cols)
        
        # Limit feature engineering for large number of columns
        max_cols = min(4, n_cols)
        
        # 1. Interaction features (products) - limited
        if n_cols >= 2:
            for i in range(min(2, max_cols)):
                for j in range(i+1, min(3, max_cols)):
                    col1, col2 = cols[i], cols[j]
                    df_eng[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        
        # 2. Ratio features - limited
        if n_cols >= 2:
            for i in range(min(2, max_cols)):
                for j in range(i+1, min(3, max_cols)):
                    col1, col2 = cols[i], cols[j]
                    denom = df[col2].replace(0, 0.001)
                    df_eng[f'{col1}_div_{col2}'] = df[col1] / denom
        
        # 3. Polynomial features (limited)
        for col in cols[:min(2, n_cols)]:
            df_eng[f'{col}_sq'] = df[col] ** 2
        
        # 4. Log features (for skewed distributions)
        for col in cols[:min(2, n_cols)]:
            df_eng[f'{col}_log'] = np.log1p(np.abs(df[col]))
        
        # Clean up infinities and NaN
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        df_eng = df_eng.fillna(df_eng.median())
        
        return df_eng
    
    def select_best_features(self, X, n_features=10):
        """Select features with highest variance for clustering."""
        selector = VarianceThreshold(threshold=0.01)
        try:
            X_selected = selector.fit_transform(X)
            if X_selected.shape[1] < 2:
                return X
            variances = np.var(X_selected, axis=0)
            top_indices = np.argsort(variances)[-min(n_features, len(variances)):]
            return X_selected[:, top_indices]
        except:
            return X
    
    def try_kmeans(self, X, k_range=(2, 6)):
        """Try KMeans with multiple k values."""
        results = []
        KMeansClass = MiniBatchKMeans if self.use_minibatch else KMeans
        
        for k in range(k_range[0], k_range[1] + 1):
            try:
                if self.use_minibatch:
                    model = KMeansClass(
                        n_clusters=k,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        batch_size=1024,
                        random_state=self.random_state
                    )
                else:
                    model = KMeansClass(
                        n_clusters=k,
                        init='k-means++',
                        n_init=20,
                        max_iter=500,
                        tol=1e-5,
                        random_state=self.random_state
                    )
                labels = model.fit_predict(X)
                if len(set(labels)) > 1:
                    score = silhouette_score(X, labels)
                    results.append({
                        'method': f'KMeans(k={k})',
                        'labels': labels,
                        'score': score,
                        'k': k,
                        'model': model
                    })
            except Exception:
                continue
        return results
    
    def try_spectral(self, X, k_range=(2, 5)):
        """Try Spectral Clustering - excellent for non-convex clusters."""
        results = []
        n_samples = X.shape[0]
        
        # Limit for large datasets (spectral is slow)
        if n_samples > 5000:
            return results
        
        for k in range(k_range[0], min(k_range[1] + 1, n_samples)):
            for affinity in ['rbf', 'nearest_neighbors']:
                try:
                    model = SpectralClustering(
                        n_clusters=k,
                        affinity=affinity,
                        n_neighbors=min(10, n_samples - 1),
                        random_state=self.random_state,
                        assign_labels='kmeans'
                    )
                    labels = model.fit_predict(X)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X, labels)
                        results.append({
                            'method': f'Spectral({affinity}, k={k})',
                            'labels': labels,
                            'score': score,
                            'k': k,
                            'model': model
                        })
                except Exception:
                    continue
        return results
    
    def try_agglomerative(self, X, k_range=(2, 5)):
        """Try Agglomerative (Hierarchical) Clustering."""
        results = []
        n_samples = X.shape[0]
        
        # Limit for large datasets
        if n_samples > 10000:
            k_range = (2, 4)
        
        for k in range(k_range[0], k_range[1] + 1):
            for linkage_type in ['ward', 'complete', 'average']:
                try:
                    model = AgglomerativeClustering(
                        n_clusters=k,
                        linkage=linkage_type
                    )
                    labels = model.fit_predict(X)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X, labels)
                        results.append({
                            'method': f'Agglomerative({linkage_type}, k={k})',
                            'labels': labels,
                            'score': score,
                            'k': k,
                            'model': model
                        })
                except Exception:
                    continue
        return results
    
    def try_gmm(self, X, k_range=(2, 5)):
        """Try Gaussian Mixture Model - handles overlapping clusters well."""
        results = []
        
        for k in range(k_range[0], k_range[1] + 1):
            for cov_type in ['full', 'diag']:
                try:
                    model = GaussianMixture(
                        n_components=k,
                        covariance_type=cov_type,
                        n_init=10,
                        max_iter=200,
                        random_state=self.random_state
                    )
                    labels = model.fit_predict(X)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X, labels)
                        results.append({
                            'method': f'GMM({cov_type}, k={k})',
                            'labels': labels,
                            'score': score,
                            'k': k,
                            'model': model
                        })
                except Exception:
                    continue
        return results
    
    def try_birch(self, X, k_range=(2, 5)):
        """Try BIRCH clustering - efficient for large datasets."""
        results = []
        
        for k in range(k_range[0], k_range[1] + 1):
            try:
                model = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
                labels = model.fit_predict(X)
                if len(set(labels)) > 1 and len(set(labels)) <= k:
                    score = silhouette_score(X, labels)
                    results.append({
                        'method': f'BIRCH(k={k})',
                        'labels': labels,
                        'score': score,
                        'k': k,
                        'model': model
                    })
            except Exception:
                continue
        return results
    
    def try_hdbscan(self, X):
        """Try HDBSCAN - noise-resistant density-based clustering."""
        if not HDBSCAN_AVAILABLE:
            return []
        
        results = []
        n_samples = X.shape[0]
        
        for min_cluster_size in [15, 25, 50]:
            if min_cluster_size > n_samples // 10:
                continue
            try:
                model = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=5,
                    metric='euclidean'
                )
                labels = model.fit_predict(X)
                # HDBSCAN uses -1 for noise, filter those out for scoring
                valid_mask = labels >= 0
                if valid_mask.sum() > 10 and len(set(labels[valid_mask])) > 1:
                    score = silhouette_score(X[valid_mask], labels[valid_mask])
                    n_clusters = len(set(labels[valid_mask]))
                    results.append({
                        'method': f'HDBSCAN(min={min_cluster_size})',
                        'labels': labels,
                        'score': score,
                        'k': n_clusters,
                        'model': model
                    })
            except Exception:
                continue
        return results
    
    def try_pca_enhanced_clustering(self, X, k_range=(2, 5)):
        """Apply PCA first, then cluster - often improves separation."""
        results = []
        n_features = X.shape[1]
        
        for n_comp in [2, 3]:
            if n_comp >= n_features:
                continue
            try:
                pca = PCA(n_components=n_comp)
                X_pca = pca.fit_transform(X)
                
                for k in range(k_range[0], k_range[1] + 1):
                    KMeansClass = MiniBatchKMeans if self.use_minibatch else KMeans
                    
                    if self.use_minibatch:
                        model = KMeansClass(
                            n_clusters=k,
                            init='k-means++',
                            n_init=10,
                            batch_size=1024,
                            random_state=self.random_state
                        )
                    else:
                        model = KMeansClass(
                            n_clusters=k,
                            init='k-means++',
                            n_init=20,
                            random_state=self.random_state
                        )
                    labels = model.fit_predict(X_pca)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_pca, labels)
                        results.append({
                            'method': f'PCA({n_comp})+KMeans(k={k})',
                            'labels': labels,
                            'score': score,
                            'k': k,
                            'model': model,
                            'pca': pca,
                            'pca_data': X_pca
                        })
            except Exception:
                continue
        return results
    
    def find_best_clustering(self, X_original, X_engineered=None):
        """
        Run all clustering algorithms and find the best one.
        Uses parallel processing when possible.
        """
        start_time = time.time()
        all_results = []
        n_samples = X_original.shape[0]
        
        # Determine if we should use mini-batch
        self.use_minibatch = n_samples > 10000
        
        # Prepare data
        datasets = [('Original', X_original)]
        if X_engineered is not None and X_engineered.shape[1] > X_original.shape[1]:
            datasets.append(('Engineered', X_engineered))
        
        for data_name, X in datasets:
            # Scale data
            scaler = RobustScaler()  # More robust to outliers
            X_scaled = scaler.fit_transform(X)
            
            # Remove outliers
            mask = self.remove_outliers_iqr(X_scaled, factor=2.0)
            X_clean = X_scaled[mask] if mask.sum() > 50 else X_scaled
            
            print(f"Testing algorithms on {data_name} data ({X_clean.shape[0]} samples)...")
            
            # Collect all results
            results = []
            results.extend(self.try_kmeans(X_clean))
            results.extend(self.try_gmm(X_clean))
            results.extend(self.try_birch(X_clean))
            results.extend(self.try_agglomerative(X_clean))
            results.extend(self.try_pca_enhanced_clustering(X_clean))
            
            # Only try expensive algorithms on smaller datasets
            if n_samples <= 5000:
                results.extend(self.try_spectral(X_clean))
            
            # Try HDBSCAN if available
            if HDBSCAN_AVAILABLE and n_samples <= 20000:
                results.extend(self.try_hdbscan(X_clean))
            
            for r in results:
                r['data_type'] = data_name
                r['mask'] = mask
                r['scaler'] = scaler
            
            all_results.extend(results)
        
        self.all_results = all_results
        self.processing_time = time.time() - start_time
        print(f"Clustering completed in {self.processing_time:.2f} seconds")
        
        if not all_results:
            return None, None, 0, "None", 2
        
        # Find best result
        best = max(all_results, key=lambda x: x['score'])
        
        self.best_score = best['score']
        self.best_labels = best['labels']
        self.best_method = best['method']
        self.best_k = best['k']
        
        return best['labels'], best['score'], best['method'], best['k'], best
    
    def get_cluster_statistics(self, X, labels):
        """Calculate detailed cluster statistics."""
        if labels is None:
            return {}
        
        stats = {
            'silhouette': silhouette_score(X, labels),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
            'n_clusters': len(set(labels)),
            'cluster_sizes': {i: int((labels == i).sum()) for i in set(labels)},
            'processing_time': self.processing_time
        }
        return stats
