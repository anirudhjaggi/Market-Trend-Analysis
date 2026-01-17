 # 🛒 **AI FOR MARKET TREND ANALYSIS**

**Author:** Anirudh Jaggi  

---

## 📊 Project Overview

An end-to-end AI-powered analytics dashboard for retail businesses featuring:
- Revenue Forecasting (XGBoost, LSTM, Prophet, SARIMA)
- Customer Segmentation (RFM Analysis)
- Clustering Analysis (HDBSCAN)
- Anomaly Detection (Isolation Forest)
- Model Explainability (SHAP)

---

## 🎯 Features

### 1. Revenue Forecasting
- Multiple ML models for 7-90 day predictions
- 83.5% accuracy with XGBoost
- Interactive confidence intervals
- Trend analysis and insights

### 2. Customer Segmentation
- RFM (Recency, Frequency, Monetary) analysis
- 6 distinct customer segments
- Actionable marketing strategies
- 3D visualization

### 3. Clustering Analysis
- HDBSCAN unsupervised learning
- Natural customer grouping
- Outlier detection
- 2D visualization with t-SNE

### 4. Anomaly Detection
- Isolation Forest algorithm
- Risk scoring (0-1 scale)
- Fraud detection
- High-risk customer identification

### 5. Model Explainability
- SHAP value analysis
- Feature importance ranking
- 97.3% business-driven model
- Transparent AI decisions

---

## 📁 Project Structure

- RETAILGPT/
- ├── dashboard/          # Streamlit dashboard
- │   ├── app.py         # Main application
- │   └── data/          # Dashboard data files
- ├── notebooks/          # Jupyter notebooks
- │   ├── 01_data_prep.ipynb
- │   ├── 02_xgboost_forecast.ipynb
- │   └── 03_export_dashboard_data.ipynb
- ├── data/              # Analysis outputs & visualizations
- ├── requirements.txt    # Python dependencies
- └── README.md          # This file

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone repository
git clone https://github.com/[your-username]/retail-analytics-ai.git
cd retail-analytics-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

