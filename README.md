# 🚀 MLOps ML Project

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![MLOps](https://img.shields.io/badge/MLOps-Ready-orange.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

_A production-ready machine learning baseline project demonstrating MLOps best practices with reproducible pipelines, automated artifact generation, and comprehensive Git versioning._

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Overview

This project serves as a **comprehensive MLOps baseline** for building, training, and deploying machine learning models with industry-standard practices. It provides a structured framework for reproducible ML workflows, from data preprocessing to model evaluation and artifact management.

### 🎯 Key Highlights

- ✅ **Reproducible Training Pipeline** - Consistent results across environments
- ✅ **Configuration-Driven** - Easy hyperparameter tuning via YAML
- ✅ **Automated Artifact Generation** - Models, metrics, and visualizations
- ✅ **Git-Friendly Structure** - Clean separation of code and outputs
- ✅ **Extensible Architecture** - Modular design for custom datasets and models

---

## 🏗️ Project Architecture

```
mlops-ml-project/
├── 📁 config/
│   └── train.yaml              # 🔧 Training configuration & hyperparameters
│
├── 📁 src/                     # 🧠 Core ML modules
│   ├── __init__.py
│   ├── data.py                 # 📊 Data loading and validation
│   ├── features.py             # 🔄 Preprocessing pipeline
│   └── model.py                # 🤖 Model architecture and training
│
├── 📁 scripts/                 # 🎬 Execution scripts
│   ├── train.py                # 🏋️ Model training workflow
│   └── evaluate.py             # 📈 Model evaluation workflow
│
├── 📁 tests/                   # ✅ Test suite
│   └── test_config.py          # 🧪 Configuration validation tests
│
├── 📁 artifacts/               # 📦 Generated outputs (gitignored)
│   ├── model.joblib            # 💾 Serialized trained model
│   ├── metrics.json            # 📊 Performance metrics
│   ├── confusion_matrix.png    # 🎨 Confusion matrix visualization
│   └── report.json             # 📄 Detailed classification report
│
├── .gitignore                  # 🚫 Git exclusion rules
├── README.md                   # 📖 This file
└── requirements.txt            # 📚 Python dependencies
```

---

## ✨ Features

### 🔄 Automated ML Pipeline

- **Data Loading**: Seamless integration with scikit-learn datasets and custom CSV files
- **Preprocessing**: Configurable feature engineering and data transformation
- **Training**: Automated model training with configurable hyperparameters
- **Evaluation**: Comprehensive model assessment with multiple metrics

### 📊 Rich Artifact Generation

- **Serialized Models**: Production-ready model files (`.joblib` format)
- **Performance Metrics**: JSON-formatted accuracy and F1 scores
- **Visualizations**: Confusion matrices for model interpretation
- **Detailed Reports**: Per-class precision, recall, and F1-score breakdowns

### 🛠️ Developer Experience

- **Configuration Management**: YAML-based configuration for easy experimentation
- **Modular Codebase**: Clean separation of concerns for maintainability
- **Testing Infrastructure**: Built-in test suite for configuration validation
- **Documentation**: Comprehensive inline documentation and type hints

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Quick Start

1️⃣ **Clone the repository**

```bash
git clone https://github.com/YOUHAD08/mlops-ml-project.git
cd mlops-ml-project
```

2️⃣ **Set up virtual environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate (Git Bash on Windows)
source .venv/Scripts/activate

# Activate (Linux/macOS)
# source .venv/bin/activate
```

3️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

4️⃣ **Verify installation**

```bash
python -c "import sklearn, pandas, numpy; print('✅ All dependencies installed successfully!')"
```

---

## 💻 Usage

### Training a Model

Train the model using the default configuration:

```bash
python scripts/train.py
```

**📤 Generated Artifacts:**

| Artifact                | Description                   | Location                         |
| ----------------------- | ----------------------------- | -------------------------------- |
| 🤖 **Trained Model**    | Serialized scikit-learn model | `artifacts/model.joblib`         |
| 📊 **Metrics**          | Accuracy and macro F1 score   | `artifacts/metrics.json`         |
| 🎨 **Confusion Matrix** | Visual performance heatmap    | `artifacts/confusion_matrix.png` |

**Example Output:**

```json
{
  "accuracy": 0.9667,
  "f1_macro": 0.9655,
  "timestamp": "2026-01-17T10:30:00"
}
```

---

### Evaluating a Model

Generate a detailed evaluation report:

```bash
python scripts/evaluate.py
```

**📤 Generated Artifacts:**

| Artifact                     | Description                   | Location                |
| ---------------------------- | ----------------------------- | ----------------------- |
| 📄 **Classification Report** | Per-class performance metrics | `artifacts/report.json` |

**Example Output:**

```json
{
  "class_0": {
    "precision": 1.0,
    "recall": 1.0,
    "f1-score": 1.0
  },
  "class_1": {
    "precision": 0.95,
    "recall": 0.9,
    "f1-score": 0.92
  }
}
```

---

## ⚙️ Configuration

Customize your ML pipeline by editing `config/train.yaml`:

```yaml
# Dataset Configuration
dataset:
  name: "iris" # Dataset to use
  test_size: 0.2 # Train/test split ratio
  random_state: 42 # Reproducibility seed

# Model Configuration
model:
  type: "RandomForestClassifier" # Model architecture
  n_estimators: 100 # Number of trees
  max_depth: 5 # Maximum tree depth
  random_state: 42 # Model seed

# Output Configuration
artifacts:
  directory: "artifacts/" # Output directory
  save_model: true # Save trained model
  save_metrics: true # Save performance metrics
  save_plots: true # Generate visualizations
```

### 🔧 Supported Configurations

- **Datasets**: Iris (default), custom CSV files
- **Models**: Random Forest, Logistic Regression, SVM (extensible)
- **Metrics**: Accuracy, F1-score, Precision, Recall
- **Visualizations**: Confusion Matrix, Feature Importance (coming soon)

---

## 📦 Artifacts Directory

All generated files are stored in `artifacts/` (excluded from Git):

```
artifacts/
├── 🤖 model.joblib              # Trained model (serialized with joblib)
├── 📊 metrics.json              # Overall performance metrics
├── 🎨 confusion_matrix.png      # Confusion matrix heatmap
└── 📄 report.json               # Detailed classification report
```

### Artifact Details

#### 🤖 `model.joblib`

- **Format**: Joblib-serialized scikit-learn model
- **Usage**: Load with `joblib.load('artifacts/model.joblib')`
- **Size**: Typically 10-50 KB for baseline models

#### 📊 `metrics.json`

- **Format**: JSON with top-level metrics
- **Includes**: Accuracy, F1-macro, timestamp
- **Purpose**: Quick performance overview

#### 🎨 `confusion_matrix.png`

- **Format**: PNG image (300 DPI)
- **Dimensions**: 800x600 pixels
- **Purpose**: Visual model performance analysis

#### 📄 `report.json`

- **Format**: JSON with per-class metrics
- **Includes**: Precision, Recall, F1-score for each class
- **Purpose**: Detailed performance breakdown

---

## 📊 Dataset Information

### Default: Iris Dataset

The project uses the **Iris flower dataset** as a baseline:

- **Source**: Built into scikit-learn (no download required)
- **Samples**: 150 (50 per class)
- **Features**: 4 (sepal length/width, petal length/width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Type**: Multi-class classification

### 🔄 Custom Datasets

To use your own dataset, modify `config/train.yaml`:

```yaml
dataset:
  name: "custom"
  path: "data/your_dataset.csv"
  target_column: "label"
  feature_columns: ["feat1", "feat2", "feat3"]
```

**Requirements:**

- CSV format with headers
- Numerical features (or encode categorical features)
- Clear target/label column

---

## 🧪 Testing

Run the test suite to validate configuration and setup:

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_config.py -v
```

**Test Coverage:**

- ✅ Configuration file validation
- ✅ Data loading functionality
- ✅ Model initialization
- ✅ Artifact generation

---

## 🛠️ Tech Stack

| Component           | Technology          |
| ------------------- | ------------------- |
| **Language**        | Python 3.8+         |
| **ML Framework**    | scikit-learn        |
| **Data Processing** | pandas, numpy       |
| **Visualization**   | matplotlib, seaborn |
| **Configuration**   | PyYAML              |
| **Testing**         | pytest              |
| **Serialization**   | joblib              |

---

## 🗺️ Roadmap

### ✅ Current Features

- [x] Reproducible training pipeline
- [x] Automated artifact generation
- [x] Configuration management
- [x] Basic model evaluation

### 🔜 Upcoming Features

- [ ] Docker containerization
- [ ] CI/CD pipeline integration
- [ ] Hyperparameter tuning (GridSearch/RandomSearch)
- [ ] Model versioning with MLflow
- [ ] Feature importance visualization
- [ ] Cross-validation support
- [ ] API endpoint for model serving
- [ ] Experiment tracking dashboard

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
