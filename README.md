# SegmentX 🎯

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?style=for-the-badge&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered E-commerce Customer Segmentation System using advanced clustering algorithms**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Algorithms](#-algorithms) • [Screenshots](#-screenshots) • [API](#-api-reference)

</div>

---

## 🌟 Overview

SegmentX is a production-grade customer segmentation application that leverages multiple machine learning algorithms to identify distinct customer groups based on their purchasing behavior. Upload your transactional data and get actionable insights within seconds.

### Key Highlights

- 🚀 **Real-time Analysis** - Get clustering results in seconds, not hours
- 🧠 **Multiple Algorithms** - Automatically selects the best clustering approach
- 📊 **Rich Visualizations** - Interactive 2D/3D PCA plots, distribution charts, and more
- 💼 **RFM Analysis** - Recency, Frequency, Monetary analysis for marketing insights
- 📈 **Sales Analytics** - Comprehensive sales performance metrics
- 🎨 **Modern UI** - Beautiful glassmorphism design with animations

---

## ✨ Features

### 🔬 Advanced Clustering Engine

The system tries multiple clustering algorithms and configurations to find the optimal customer segments:

| Algorithm | Description |
|-----------|-------------|
| **KMeans** | Classic partitioning with multiple k values |
| **Mini-Batch KMeans** | Optimized for large datasets |
| **Spectral Clustering** | Excellent for non-convex clusters |
| **Agglomerative** | Hierarchical clustering approach |
| **Gaussian Mixture Model** | Handles overlapping clusters |
| **BIRCH** | Efficient for large datasets |
| **HDBSCAN** | Density-based, noise-resistant (optional) |

### 📊 RFM Analysis

Automatically segments customers based on:
- **Recency** - How recently a customer made a purchase
- **Frequency** - How often they purchase
- **Monetary** - How much they spend

Customer segments include: Champions, Loyal Customers, Potential Loyalists, New Customers, At Risk, and more.

### 🎨 Visualizations

- **2D & 3D PCA Scatter Plots** - Visualize cluster separation
- **Cluster Distribution** - See customer count per segment
- **Feature Distributions** - Understand spending patterns
- **Correlation Heatmaps** - Identify feature relationships
- **Silhouette Analysis** - Measure cluster quality

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/AshishCh01/SegmentX.git
   cd SegmentX
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   ```
   http://localhost:5000
   ```

### Optional: Enable HDBSCAN

For density-based clustering with noise resistance:
```bash
pip install hdbscan>=0.8.29
```

---

## 📖 Usage

### Preparing Your Data

Your CSV file should contain customer transaction data with columns like:
- Customer ID (optional - auto-detected)
- Purchase Amount / Spending
- Number of Items / Quantity
- Purchase Date (for RFM analysis)
- Any other numerical features

**Example CSV format:**
```csv
CustomerID,TotalSpending,NumItems,Frequency,AvgOrderValue
C001,1500.00,25,12,125.00
C002,450.50,8,3,150.17
C003,3200.00,45,24,133.33
```

### Steps

1. Navigate to `http://localhost:5000`
2. Drag and drop your CSV file or click to browse
3. Click **"Analyze Customers"**
4. Review the segmentation results and visualizations
5. Download the clustered data with segment labels

---

## 🔧 Project Structure

```
SegmentX/
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python version specification
│
├── analysis/                  # Analysis modules
│   ├── __init__.py
│   ├── preprocessing.py      # Data preprocessing & cleaning
│   ├── clustering.py         # Advanced clustering engine
│   ├── rfm_analysis.py       # RFM segmentation
│   └── sales_analytics.py    # Sales performance metrics
│
├── templates/                 # HTML templates
│   ├── index.html            # Upload page
│   └── result.html           # Results dashboard
│
├── static/                    # Static assets
│   └── uploads/              # Uploaded files & generated plots
│
└── data/                      # Sample datasets
    ├── amazon.csv
    ├── customer_shopping_data.csv
    └── sample_customers.csv
```

---

## 🧮 Algorithms

### Clustering Pipeline

1. **Data Preprocessing**
   - Handle missing values
   - Remove duplicates
   - Normalize/standardize features
   - Outlier detection (IQR method)

2. **Feature Engineering**
   - Create derived features (e.g., spending ratios)
   - PCA for dimensionality reduction
   - Feature selection based on variance

3. **Algorithm Selection**
   - Run multiple algorithms in parallel
   - Evaluate using Silhouette Score
   - Select configuration with best separation

4. **Labeling**
   - Assign meaningful segment names
   - Generate descriptive statistics per cluster

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Silhouette Score** | Measures cluster cohesion and separation (-1 to 1) |
| **Davies-Bouldin Index** | Lower values indicate better clustering |
| **Calinski-Harabasz Index** | Higher values indicate better-defined clusters |

---

## 📡 API Reference

### Endpoints

#### `GET /`
Renders the upload page.

#### `POST /analyze`
Analyzes the uploaded CSV file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (CSV file)

**Response:**
Renders result page with:
- Cluster assignments
- Visualization plots
- Segment statistics
- Downloadable clustered CSV

---

## 📊 Sample Results

When you run the analysis, you'll get:

- **Cluster Distribution Chart** - Visual breakdown of customer segments
- **2D PCA Plot** - Scatter plot showing cluster separation
- **3D PCA Plot** - Interactive 3D visualization
- **Feature Importance** - Which features drive segmentation
- **Segment Profiles** - Detailed statistics for each customer group

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **scikit-learn** - Machine learning library
- **Flask** - Web framework
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualizations
- **TailwindCSS** - UI styling

---

<div align="center">

**Built with ❤️ for data-driven business decisions**

[⬆ Back to Top](#segmentx-)

</div>
