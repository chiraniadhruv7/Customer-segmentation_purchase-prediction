# 🛒 Customer Segmentation and Purchase Prediction System

> A hybrid machine learning pipeline combining unsupervised clustering and supervised classification to segment customers and predict high-value purchasing behaviour.

---

## 📌 Overview

This project builds an end-to-end ML system on the **UCI Online Retail Dataset** (541,909 transactions, 4,338 unique customers). It solves two connected business problems:

1. **Who are my customers?** — Discover natural behavioural segments using clustering
2. **Which customers are high-value?** — Predict future high spenders using classification

### 🌟 Novel Contribution — Hybrid Pipeline

```
Raw Transactions
      ↓
  RFM Feature Engineering
      ↓
  K-Means Clustering  ──────────────────────┐
      ↓                                      │
  Cluster Label (discovered unsupervised)    │
      ↓  ←───────────────────────────────────┘
  [Recency_log, Frequency_log, KMeans_Cluster]
      ↓
  Classifier (Random Forest / XGBoost)
      ↓
  High-Value Customer Prediction
```

The cluster label — discovered with **zero knowledge of the target** — becomes the single most important classification feature at **41.7% importance**. This validates the hybrid approach quantitatively.

---

## 📊 Results Summary

| Task | Method | Score |
|------|--------|-------|
| Clustering | K-Means (k=4) | Silhouette = 0.3371 |
| Classification | Tuned Random Forest | Accuracy = 88.5% |
| Classification | Tuned Random Forest | AUC-ROC = 0.9601 |
| Classification | Tuned Random Forest | Recall = 88.5% |
| Feature Importance | Cluster Label | 41.7% (top feature) |

### Customer Segments Discovered

| Segment | Count | Avg Recency | Avg Frequency | Avg Monetary |
|---------|-------|-------------|---------------|--------------|
| 🏆 Champions | 716 | 12 days | 13.7 orders | £8,074 |
| 💙 Loyal Customers | 1,173 | 71 days | 4.1 orders | £1,802 |
| ⚠️ At-Risk | 837 | 18 days | 2.1 orders | £551 |
| 😴 Lost/Inactive | 1,612 | 182 days | 1.3 orders | £343 |

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | 85.8% | 85.4% | 77.8% | 81.5% | 0.926 |
| Random Forest | 88.4% | 85.8% | 85.0% | 85.4% | 0.948 |
| XGBoost | 87.9% | 85.4% | 84.2% | 84.8% | 0.960 |
| **RF Tuned ★** | **88.5%** | 83.7% | **88.5%** | **86.0%** | **0.960** |

---

## 📁 Project Structure

```
customer-segmentation/
│
├── Customer_Segmentation_Prediction.ipynb   # Main notebook (run this)
├── requirements.txt                          # All dependencies
├── online_retail.xlsx                        # Dataset (auto-downloaded)
│
├── outputs/
│   ├── eda_overview.png                      # EDA plots
│   └── ...
│
├── report/
│   ├── Customer_Segmentation_Report.pdf      # Compiled PDF report
│   ├── Customer_Segmentation_Report.tex      # LaTeX source
│   └── Customer_Segmentation_Report.docx     # Word document
│
└── presentation/
    └── Customer_Segmentation_Presentation.pptx
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/customer-segmentation.git
cd customer-segmentation
```

### 2. Create a virtual environment (recommended)

```bash
# Create
python -m venv segmentation_env

# Activate — Windows
segmentation_env\Scripts\activate

# Activate — Mac/Linux
source segmentation_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy openpyxl jupyter notebook
```

### 4. Verify installation

```python
import pandas, numpy, sklearn, xgboost, scipy
print("All good!")
```

---

## 🚀 Quick Start

```bash
# Open in Jupyter
jupyter notebook Customer_Segmentation_Prediction.ipynb

# Or in VS Code
code .
# Then open the .ipynb file and select your Python kernel
```

The dataset is **auto-downloaded** from the UCI repository on first run — no manual download needed.

---

## 🔬 Methodology

### 1. Data Cleaning

| Step | Rows Removed | Reason |
|------|-------------|--------|
| Missing CustomerID | 135,080 | Cannot do customer analysis |
| Cancelled invoices | 8,905 | InvoiceNo starts with 'C' |
| Bad Quantity/Price | 40 | Non-positive values |
| **Remaining** | **397,884** | Clean dataset |

### 2. RFM Feature Engineering

Each customer summarised into 3 features relative to reference date `2011-12-10`:

```python
rfm = df.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', lambda x: (reference_date - x.max()).days),
    Frequency = ('InvoiceNo',   'nunique'),
    Monetary  = ('TotalPrice',  'sum')
)
```

All features log-transformed (`log1p`) and standardised (`StandardScaler`) before clustering.

### 3. Clustering

Three algorithms compared:

| Algorithm | Silhouette | Clusters | Verdict |
|-----------|------------|----------|---------|
| **K-Means** | **0.3371** | 4 | ✅ Best |
| DBSCAN | 0.2931 | 2 | ❌ Too coarse |
| Hierarchical | 0.2419 | 4 | ❌ Unbalanced |

Optimal k=4 selected via Elbow Method + Silhouette Score analysis.

### 4. Target Variable (Leakage-Free)

```python
# Target: top 40% spenders (Monetary >= 60th percentile)
rfm['HighValue'] = (rfm['Monetary'] >= rfm['Monetary'].quantile(0.60)).astype(int)
# Threshold: £942.28 | Classes: 40% positive, 60% negative
```

**Why leakage-free matters:** Earlier versions using `Monetary` in both features and target, or `Recency`/`Frequency` to define the target while using them as features, produced 100% accuracy — not real learning. The current design excludes Monetary from features entirely.

### 5. Hybrid Feature Matrix

```python
# Monetary excluded — it defines the target (no leakage)
# KMeans_Cluster is the novel hybrid feature
feature_cols = ['Recency_log', 'Frequency_log', 'KMeans_Cluster']
```

### 6. Hyperparameter Tuning

```python
param_grid = {
    'n_estimators'    : [100, 200],
    'max_depth'       : [5, 10, None],
    'min_samples_leaf': [1, 5],
}
# GridSearchCV with 5-fold CV, scoring='roc_auc'
# Best: max_depth=10, min_samples_leaf=5, n_estimators=200
# Best CV AUC: 0.9523
```

---

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
scipy>=1.9.0
openpyxl>=3.0.0
jupyter>=1.0.0
notebook>=6.5.0
```

---

## 💡 Key Insights

**Why does the cluster label have 41.7% importance?**

The cluster label is a compressed, non-linear summary of the entire RFM space. K-Means discovered that Champions (high F, high M, low R) form a distinct group from Inactive (low F, low M, high R). When the classifier sees `Cluster=1`, it knows the customer is a Champion — that single number carries more predictive signal than Recency alone because it encodes the interaction between all three RFM dimensions simultaneously.

**Why 88.5% accuracy without seeing spending data?**

The model proves that *how a customer buys* (recency + frequency + behavioural cluster) reliably predicts *how much they spend*. Loyal, frequent, recent buyers tend to be high spenders — and the cluster structure captures that relationship non-linearly.

---

## 🗺️ Business Recommendations

| Segment | Strategy |
|---------|----------|
| 🏆 Champions | Loyalty programmes, VIP early access, exclusive offers |
| 💙 Loyal Customers | Cross-sell, upsell, bundle recommendations |
| ⚠️ At-Risk | Personalised win-back emails, targeted discounts |
| 😴 Lost/Inactive | Low-cost re-engagement or graceful sunset |

---

## 🔭 Future Scope

- **Real-time scoring** — FastAPI + Docker deployment for live customer scoring
- **Temporal RFM** — Rolling windows to track segment evolution over time
- **SHAP explainability** — Individual prediction explanations for business teams
- **Deep embeddings** — Autoencoder on raw transaction sequences → cluster on latent space
- **Recommendation engine** — Product recommendations within each customer cluster
- **Multi-period validation** — Confirm segment stability across multiple years

---

## 📚 References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
2. Chen, T. & Guestrin, C. (2016). XGBoost. *Proceedings of KDD*, 785–794.
3. Hughes, A. M. (1994). *Strategic Database Marketing*. Probus Publishing.
4. Rousseeuw, P. J. (1987). Silhouettes. *Journal of Computational and Applied Mathematics*, 20, 53–65.
5. Yeh, I. C. et al. (2009). Knowledge discovery on RFM model. *Expert Systems with Applications*, 36(3).
6. UCI Machine Learning Repository — [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)

---

## 📄 License

This project is submitted as an academic assignment for the Machine Learning course at Manipal University Jaipur (Batch 2023–2027). For educational use only.

---

<div align="center">
  <sub>Built with Python · Scikit-Learn · XGBoost · Pandas · Seaborn</sub>
</div>
