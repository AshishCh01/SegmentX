"""
Advanced E-commerce Customer Segmentation System
================================================
A fast, production-grade clustering application using multiple algorithms
to achieve optimal cluster separation.

Features:
- Multiple clustering algorithms (KMeans++, GMM, Agglomerative, BIRCH)
- Automatic algorithm selection based on silhouette score
- Smart dataset detection (product catalogs, transactions, generic)
- Optimized for speed with sampling and mini-batch processing
- 6+ chart types for comprehensive visualization
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings('ignore')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


def save_plot(fig, name, dpi=120):
    """Save plot with proper settings."""
    fig.savefig(os.path.join(UPLOAD_FOLDER, name), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def preprocess_data(df, sample_size=None):
    """Simple, fast preprocessing of data."""
    # Sample if needed
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Remove columns with too many missing values
    df_numeric = df_numeric.dropna(axis=1, thresh=len(df_numeric) * 0.5)
    
    # Fill remaining NaN with median
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    # Remove constant columns
    df_numeric = df_numeric.loc[:, df_numeric.std() > 0]
    
    return df_numeric


def find_best_clustering(X_scaled, max_k=6, use_minibatch=False):
    """Fast clustering with limited configurations for speed."""
    results = []
    n_samples = X_scaled.shape[0]
    
    # KMeans (fastest and usually best)
    for k in range(2, max_k + 1):
        try:
            if use_minibatch:
                model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=5, 
                                       batch_size=1024, random_state=42)
            else:
                model = KMeans(n_clusters=k, init='k-means++', n_init=10, 
                              max_iter=300, random_state=42)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                results.append({
                    'method': f'KMeans(k={k})',
                    'labels': labels,
                    'score': score,
                    'k': k,
                    'model': model
                })
        except Exception as e:
            continue
    
    # GMM (good for overlapping clusters)
    for k in range(2, min(max_k, 5) + 1):
        try:
            model = GaussianMixture(n_components=k, n_init=3, max_iter=100, random_state=42)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                score = silhouette_score(X_scaled, labels)
                results.append({
                    'method': f'GMM(k={k})',
                    'labels': labels,
                    'score': score,
                    'k': k,
                    'model': model
                })
        except Exception:
            continue
    
    # Agglomerative (only for smaller datasets)
    if n_samples <= 5000:
        for k in range(2, min(max_k, 5) + 1):
            try:
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
                labels = model.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    results.append({
                        'method': f'Agglomerative(k={k})',
                        'labels': labels,
                        'score': score,
                        'k': k,
                        'model': model
                    })
            except Exception:
                continue
    
    # BIRCH (fast for large datasets)
    if n_samples > 1000:
        for k in range(2, min(max_k, 5) + 1):
            try:
                model = Birch(n_clusters=k, threshold=0.5)
                labels = model.fit_predict(X_scaled)
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    results.append({
                        'method': f'BIRCH(k={k})',
                        'labels': labels,
                        'score': score,
                        'k': k,
                        'model': model
                    })
            except Exception:
                continue
    
    if not results:
        return None, None, 0, "None", 2, {}
    
    # Find best result
    best = max(results, key=lambda x: x['score'])
    return best['labels'], best['score'], best['method'], best['k'], results, best


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    start_time = time.time()
    
    file = request.files['file']
    if not file:
        return "No file uploaded", 400
    
    # Load data
    df = pd.read_csv(file)
    df_original = df.copy()
    n_rows = len(df)
    
    # Determine sample size for large datasets
    if n_rows > 20000:
        sample_size = 15000  # Reduced for faster processing
        print(f"Large dataset ({n_rows} rows). Sampling {sample_size} rows...")
    elif n_rows > 10000:
        sample_size = 10000
        print(f"Medium dataset ({n_rows} rows). Sampling {sample_size} rows...")
    else:
        sample_size = None
    
    # Preprocess data
    df_processed = preprocess_data(df, sample_size=sample_size)
    
    if df_processed.shape[1] < 2:
        return "Need at least 2 numeric features for clustering", 400
    
    n_samples = len(df_processed)
    X = df_processed.values
    
    # Scale data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use mini-batch for larger datasets
    use_minibatch = n_samples > 5000
    
    # Find best clustering
    labels, score, method, k, all_results, best_config = find_best_clustering(
        X_scaled, max_k=6, use_minibatch=use_minibatch
    )
    
    if labels is None:
        return "Could not find valid clusters in the data", 400
    
    # Create results dataframe
    df_for_clustering = df_original.iloc[:n_samples].copy()
    df_for_clustering["Cluster"] = labels
    
    # Create cluster labels
    label_names = [
        "Budget Shoppers", "High Spenders", "Occasional Buyers", 
        "Loyal Customers", "New Customers", "Premium Members"
    ]
    cluster_labels = {i: label_names[i] for i in range(k)}
    df_for_clustering["Cluster_Label"] = df_for_clustering["Cluster"].map(cluster_labels)
    df_for_clustering.to_csv(os.path.join(UPLOAD_FOLDER, "clustered_customers_labeled.csv"), index=False)
    
    # Get top results for comparison
    top_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:8]
    
    # ========== VISUALIZATIONS ==========
    colors_palette = sns.color_palette("husl", k)
    numeric_cols = df_processed.columns.tolist()
    
    # ===== Plot 1: Algorithm Comparison + Silhouette Analysis =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top methods bar chart
    methods = [r['method'] for r in top_results]
    scores = [r['score'] for r in top_results]
    colors = ['#10b981' if i == 0 else '#667eea' for i in range(len(scores))]
    
    axes[0].barh(range(len(methods)), scores, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(len(methods)))
    axes[0].set_yticklabels(methods, fontsize=9)
    axes[0].set_xlabel('Silhouette Score', fontsize=11)
    axes[0].set_title(f'Best: {method} (Score: {score:.3f})', fontsize=12, fontweight='bold')
    axes[0].axvline(x=score, color='red', linestyle='--', alpha=0.7)
    
    for i, s in enumerate(scores):
        axes[0].text(s + 0.005, i, f'{s:.3f}', va='center', fontsize=9)
    
    # Silhouette analysis
    silhouette_vals = silhouette_samples(X_scaled, labels)
    y_lower = 10
    
    for i in range(k):
        cluster_silhouette = silhouette_vals[labels == i]
        cluster_silhouette.sort()
        size = len(cluster_silhouette)
        y_upper = y_lower + size
        color = plt.cm.viridis(i / k)
        axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette, 
                              facecolor=color, edgecolor=color, alpha=0.7)
        axes[1].text(-0.05, y_lower + 0.5 * size, cluster_labels[i], fontsize=8)
        y_lower = y_upper + 10
    
    axes[1].axvline(x=score, color='red', linestyle='--', linewidth=2, label=f'Avg: {score:.3f}')
    axes[1].set_xlabel('Silhouette Coefficient', fontsize=11)
    axes[1].set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    save_plot(fig, "elbow.png")
    
    # ===== Plot 2: Pairplot (simplified) =====
    plot_data = df_processed.copy()
    plot_data["Cluster"] = [cluster_labels[c] for c in labels]
    
    cols_to_plot = numeric_cols[:min(3, len(numeric_cols))]
    plot_subset = plot_data[cols_to_plot + ["Cluster"]]
    
    try:
        pairplot = sns.pairplot(plot_subset, hue="Cluster", palette="husl", 
                                diag_kind="kde", plot_kws={'alpha': 0.6, 's': 30}, corner=True)
        pairplot.fig.suptitle(f'Cluster Visualization ({method})', y=1.02, fontsize=12, fontweight='bold')
        pairplot.fig.savefig(os.path.join(UPLOAD_FOLDER, "pairplot.png"), dpi=120, bbox_inches='tight')
        plt.close()
    except Exception:
        # Fallback scatter plot
        fig, ax = plt.subplots(figsize=(10, 7))
        for i in range(k):
            mask = labels == i
            ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                      label=cluster_labels[i], alpha=0.6, s=40)
        ax.legend()
        ax.set_title('Cluster Scatter Plot')
        save_plot(fig, "pairplot.png")
    
    # ===== Plot 3: PCA 2D =====
    n_components = min(2, X_scaled.shape[1])
    pca_2d = PCA(n_components=n_components)
    pca_2d_data = pca_2d.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(10, 7))
    
    for i in range(k):
        mask_i = labels == i
        plt.scatter(pca_2d_data[mask_i, 0], 
                   pca_2d_data[mask_i, 1] if n_components > 1 else np.zeros(mask_i.sum()), 
                   c=[colors_palette[i]], label=cluster_labels[i], s=50, alpha=0.7,
                   edgecolors='white', linewidth=0.5)
    
    # Add centroids if KMeans
    if hasattr(best_config.get('model'), 'cluster_centers_'):
        centers = best_config['model'].cluster_centers_
        # Transform centers to PCA space using the same PCA
        centers_2d = pca_2d.transform(centers)
        plt.scatter(centers_2d[:, 0], 
                   centers_2d[:, 1] if n_components > 1 else np.zeros(len(centers_2d)), 
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2, 
                   label='Centroids', zorder=10)
    
    var_explained = pca_2d.explained_variance_ratio_ * 100
    plt.xlabel(f'PC1 ({var_explained[0]:.1f}% variance)', fontsize=11)
    if n_components > 1:
        plt.ylabel(f'PC2 ({var_explained[1]:.1f}% variance)', fontsize=11)
    plt.title(f'PCA 2D | Score: {score:.3f}', fontsize=12, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "pca_2d.png")
    
    # ===== Plot 4: PCA 3D =====
    n_components_3d = min(3, X_scaled.shape[1])
    pca_3d = PCA(n_components=n_components_3d)
    pca_3d_data = pca_3d.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(k):
        mask_i = labels == i
        x_data = pca_3d_data[mask_i, 0]
        y_data = pca_3d_data[mask_i, 1] if n_components_3d > 1 else np.zeros(mask_i.sum())
        z_data = pca_3d_data[mask_i, 2] if n_components_3d > 2 else np.zeros(mask_i.sum())
        ax.scatter(x_data, y_data, z_data, c=[colors_palette[i]], 
                  label=cluster_labels[i], s=40, alpha=0.7)
    
    var_3d = pca_3d.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({var_3d[0]:.1f}%)')
    if n_components_3d > 1:
        ax.set_ylabel(f'PC2 ({var_3d[1]:.1f}%)')
    if n_components_3d > 2:
        ax.set_zlabel(f'PC3 ({var_3d[2]:.1f}%)')
    ax.set_title('PCA 3D Visualization', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    save_plot(fig, "pca_3d.png")
    
    # ===== Plot 5: Pie Chart =====
    cluster_counts = df_for_clustering["Cluster_Label"].value_counts()
    fig = plt.figure(figsize=(9, 7))
    colors_pie = sns.color_palette("husl", len(cluster_counts))
    
    wedges, texts, autotexts = plt.pie(
        cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%',
        startangle=140, colors=colors_pie, explode=[0.02] * len(cluster_counts),
        shadow=True, textprops={'fontsize': 10}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    plt.title(f'Customer Distribution ({k} Segments)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, "cluster_distribution_pie.png")
    
    # ===== Plot 6: Bar Chart =====
    fig = plt.figure(figsize=(10, 5))
    bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors_pie, 
                   edgecolor='black', linewidth=1.2)
    
    for bar, value in zip(bars, cluster_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(value), ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.title(f'Customer Count per Segment (Total: {n_samples})', fontsize=12, fontweight='bold')
    plt.xlabel('Customer Segment', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot(fig, "cluster_distribution_bar.png")
    
    # ===== Plot 7: Box Plots =====
    fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(14, 4))
    if len(numeric_cols) == 1:
        axes = [axes]
    
    for idx, col in enumerate(numeric_cols[:3]):
        if idx < len(axes):
            plot_df = pd.DataFrame({'value': df_processed[col], 
                                   'Cluster': [cluster_labels[c] for c in labels]})
            sns.boxplot(data=plot_df, x='Cluster', y='value', ax=axes[idx], palette="husl")
            axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=30)
    
    plt.suptitle('Feature Distribution by Cluster', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_plot(fig, "box_plots.png")
    
    # Calculate processing time
    processing_time = round(time.time() - start_time, 2)
    
    # Prepare metrics
    metrics = {
        'silhouette_score': round(score, 3),
        'n_clusters': k,
        'n_samples': n_samples,
        'processing_time': processing_time,
        'method': method
    }
    
    return render_template(
        "result.html",
        elbow_img="static/uploads/elbow.png",
        cluster_img="static/uploads/pairplot.png",
        pca_2d_img="static/uploads/pca_2d.png",
        pca_3d_img="static/uploads/pca_3d.png",
        pie_chart_img="static/uploads/cluster_distribution_pie.png",
        bar_chart_img="static/uploads/cluster_distribution_bar.png",
        box_plots_img="static/uploads/box_plots.png",
        heatmap_img=None,
        radar_img=None,
        violin_img=None,
        category_img=None,
        time_series_img=None,
        silhouette_score=round(score, 3),
        metrics=metrics,
        tables=[df_for_clustering.head(15).to_html(classes='table table-striped table-bordered', index=False)]
    )


if __name__ == "__main__":
    app.run(debug=True)
