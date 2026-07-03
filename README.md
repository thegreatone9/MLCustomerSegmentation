# Customer Segmentation — ML Pipeline 🎯

An end-to-end Machine Learning pipeline for customer segmentation using **K-Means Clustering**. This project identifies five distinct customer segments based on annual income and spending behavior, enabling data-driven marketing strategies.

## Problem Statement

Retail businesses need to understand their customer base to develop targeted marketing campaigns. This project segments mall customers into five actionable groups using unsupervised learning, providing specific marketing strategies for each segment.

## Key Features

- **5-Stage ML Pipeline**: Data Ingestion → Validation → Transformation → Training → Evaluation
- **StandardScaler Preprocessing**: Proper feature scaling for distance-based clustering
- **Configurable via YAML**: All hyperparameters and paths managed through config files
- **Flask Web App**: Premium UI for real-time customer segment prediction
- **Docker Support**: Containerized for easy deployment
- **One-Click Deployment**: Ready for Render (free tier)

## Architecture

```mermaid
flowchart LR
    A[Mall_Customers.csv] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[metrics.json]
    E --> H[model.joblib]
    D --> I[scaler.joblib]
    H --> J[Flask App]
    I --> J
    J --> K[Prediction]
```

## Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1. Data Ingestion | `DataIngestion` | Copies raw CSV to artifacts directory |
| 2. Data Validation | `DataValidation` | Validates column names against schema |
| 3. Data Transformation | `DataTransformation` | Applies StandardScaler to features |
| 4. Model Training | `ModelTrainer` | Trains KMeans (k=5) on scaled features |
| 5. Model Evaluation | `ModelEvaluation` | Computes silhouette, calinski-harabasz, inertia |

## Customer Segments

| Segment | Description | Strategy |
|---------|-------------|----------|
| 🛡️ Careful | Low Income, Low Spending | Budget-friendly products, discounts |
| 📊 Standard | Moderate Income, Moderate Spending | Balanced product range, loyalty programs |
| ⭐ Target | High Income, High Spending | Premium products, VIP services |
| 💳 Careless | Low Income, High Spending | Installment plans, credit options |
| 🧠 Sensible | High Income, Low Spending | Quality emphasis, investment products |

## UI Walkthrough

A step-by-step visual guide to the web application's prediction flow.

### Step 1 — Enter Customer Data

The landing page presents a clean prediction form. Enter the customer's **Annual Income** (in thousands) and their **Spending Score** (1–100), then click the gradient button to classify them.

![Step 1: Prediction form with input fields for Annual Income and Spending Score](docs/screenshots/step1_prediction_form.jpg)

### Step 2 — Explore the Five Segments

Scroll down to see all five customer segments the model identifies. Each card shows the segment name, income/spending profile, and a marketing tag. The highlighted **Target** segment (high income, high spenders) represents the most valuable customers. Use the **Run Pipeline** button to retrain the model on updated data.

![Step 2: Five segment cards and the Train Model section](docs/screenshots/step2_segments_overview.jpg)

### Step 3 — View Prediction Results

After submitting, the results page shows:
- **Cluster ID** inside an animated ring
- **Segment name** and behavioral description
- The customer's **input values** for reference
- A **recommended marketing strategy** tailored to that segment

Click "Predict Another Customer" to return to the form, or "View Model Metrics" to see the evaluation scores.

![Step 3: Prediction result with cluster ID, segment details, and marketing strategy](docs/screenshots/step3_prediction_result.jpg)

## Tech Stack

- **Python 3.10+**, scikit-learn, pandas, numpy
- **Flask** (web framework), Jinja2 (templates)
- **Docker** for containerization
- **Render** for deployment

## Dataset

The **Mall Customers** dataset contains 200 records with 5 features:

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | int | Unique identifier |
| Gender | object | Male/Female |
| Age | int | Customer age (18–70) |
| Annual Income (k$) | int | Income in thousands (15–137) |
| Spending Score (1-100) | int | Mall-assigned spending score |

Only **Annual Income** and **Spending Score** are used for clustering.

## Project Structure

```
MLCustomerSegmentation/
├── app.py                      # Flask web application
├── main.py                     # Pipeline orchestrator
├── setup.py                    # Package setup
├── Dockerfile                  # Docker configuration
├── render.yaml                 # Render deployment config
├── requirements.txt            # Dependencies
├── params.yaml                 # KMeans hyperparameters
├── schema.yaml                 # Data schema
├── config/
│   └── config.yaml             # Pipeline configuration
├── data/
│   └── Mall_Customers.csv      # Raw dataset
├── src/
│   └── customerSegmentation/
│       ├── __init__.py          # Logger setup
│       ├── constants/           # Path constants
│       ├── entity/              # Dataclass configs
│       ├── config/              # ConfigurationManager
│       ├── utils/               # Utility functions
│       ├── components/          # Pipeline components
│       └── pipeline/            # Prediction pipeline
├── research/                    # EDA & algorithm comparison scripts
├── templates/                   # HTML templates
├── static/                      # CSS assets
└── artifacts/                   # Generated pipeline outputs
```

## Research & Analysis

The `research/` directory contains two comprehensive analysis scripts:

### Exploratory Data Analysis (`research/eda.py`)

```bash
python research/eda.py
```

Generates 7 publication-quality visualizations:
- Feature distributions with mean/median markers
- Gender analysis (distribution, income, spending)
- Correlation heatmap
- Pairwise scatter plots
- Income vs Spending scatter (colored by age)
- Box plots with outlier detection
- Age group spending patterns

### Algorithm Comparison (`research/algorithm_comparison.py`)

```bash
python research/algorithm_comparison.py
```

Compares **4 clustering algorithms** side by side:

| Algorithm | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|-----------|:---:|:---:|:---:|
| **K-Means (k=5)** | **0.555** | **0.572** | **248.6** |
| Hierarchical (Ward) | 0.554 | 0.578 | 244.4 |
| GMM (k=5) | 0.554 | 0.576 | 244.9 |
| DBSCAN | 0.354 | 0.736 | 69.1 |

Also includes:
- **Elbow method** with silhouette score per k
- **Dendrogram** for hierarchical clustering
- **Cluster stability analysis** (20 runs with different seeds — verdict: **highly stable**)

All figures are saved to `research/figures/`.

## Getting Started


### Prerequisites

- Python 3.10+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/thegreatone9/MLCustomerSegmentation.git
cd MLCustomerSegmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python main.py

# Start the web app
python app.py
```

The app will be available at `http://localhost:8080`

### Docker

```bash
# Build the image
docker build -t customer-segmentation .

# Run the container
docker run -p 8080:8080 customer-segmentation
```

## API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Landing page with prediction form |
| `/train` | GET | Trigger full training pipeline |
| `/predict` | POST | Predict customer segment |
| `/metrics` | GET | Return evaluation metrics (JSON) |

### Example Prediction Request

```bash
curl -X POST http://localhost:8080/predict \
  -d "annual_income=60&spending_score=75"
```

## Deployment (Render — Free)

1. Push code to GitHub
2. Go to [render.com](https://render.com) and create a new Web Service
3. Connect your GitHub repository
4. Select **Docker** as the environment
5. Deploy — Render will build and serve automatically

## Configuration

### `params.yaml` — Model Hyperparameters

```yaml
KMeans:
  n_clusters: 5
  init: k-means++
  n_init: 10
  max_iter: 300
  random_state: 42
```

### `config/config.yaml` — Pipeline Paths

All artifact paths and data sources are configured here. Modify to change data sources or output locations.

### `schema.yaml` — Data Schema

Defines expected column names and types for validation.

## Acknowledgments

This project is based on the original work by [**Tensor Titans**](https://github.com/TensorTitans01) who provided the dataset and the overall project idea. The codebase has been restructured into a modular ML pipeline architecture with proper config management, Docker support, and deployment readiness.
