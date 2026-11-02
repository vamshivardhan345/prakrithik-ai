# Prakritik AI - Organic Plant Disease Detection & Recommendation System

A modern agricultural AI dashboard and deep learning system that detects plant leaf diseases using **MobileNetV2** and pairs diagnoses with verified **Organic Pesticide & Biofungicide Recommendations** from agricultural extension guidelines (ICAR / State Agricultural Universities).

---

## 🌟 Key Features

1. **MobileNetV2 Deep Learning Vision Engine**
   - Trained on 54,305 high-resolution PlantVillage images across **38 plant disease and healthy crop classes**.
   - Two-phase training: Head transfer learning followed by deep upper-layer fine-tuning.
   - Evaluated on an independent test dataset: **91.87% Test Accuracy**, **92.21% Precision**, **91.87% Recall**, **91.69% F1-Score**.

2. **Verified Organic Pesticide Integration**
   - Backed by `data/organic_pesticide_recommendations.csv`.
   - Returns active biological agents (e.g., *Trichoderma harzianum*, *Bacillus subtilis*, *Bt*), application guidance, IPM sanitation protocols, and personal protective safety guidelines.
   - Guarded by compliance disclaimers: no invented generic doses; farmers are always guided to check official product labels.

3. **Intelligent Confidence Guardrails**
   - Out-of-distribution and blank/non-leaf image rejection.
   - Low-confidence diagnostic rejection to prevent misleading diagnoses.

4. **Modern Agricultural UI Dashboard**
   - Green and white modern aesthetic with responsive layouts.
   - **Home**: Project highlights and architectural overview.
   - **Disease Detection**: Drag-and-drop file upload, live preview, loading indicator, confidence circle, top-3 probability distribution, and complete organic remedy cards.
   - **Organic Remedies Catalog**: Interactive search and filter tool by crop and disease.
   - **About Project**: Agricultural context and machine learning methodology.

---

## 📁 Project Directory Structure

```
├── app.py                            # Flask application entry point
├── requirements.txt                  # Python dependencies
├── README.md                         # Documentation & instructions
├── prepare_dataset.py                # Dataset extraction, validation & 70/15/15 split
├── train_model.py                    # MobileNetV2 two-phase training & evaluation script
├── test_system.py                    # End-to-end integration test suite
├── data/
│   └── organic_pesticide_recommendations.csv # Verified organic remedy database
├── model/
│   ├── best_plant_model.keras        # Best checkpoint weights (MobileNetV2)
│   └── classes.json                  # Alphabetical list of 38 PlantVillage classes
├── dataset/
│   ├── train/                        # 37,997 training images (70%)
│   ├── validation/                   # 8,129 validation images (15%)
│   └── test/                         # 8,179 independent test images (15%)
├── results/
│   ├── accuracy_graph.png            # Training & validation accuracy curve
│   ├── loss_graph.png                # Training & validation loss curve
│   ├── confusion_matrix.png          # 38-class test confusion matrix heatmap
│   ├── classification_report.txt     # Per-class precision, recall, and F1-score
│   ├── classification_report.json    # JSON formatted metrics
│   ├── dataset_distribution_report.txt # Image counts per split
│   └── error_analysis.txt            # Analysis of lowest performing classes
├── static/
│   ├── css/style.css                 # Modern green-white agricultural stylesheet
│   ├── js/script.js                  # Frontend controller & API integration
│   ├── logo.png                      # Application logo
│   └── uploads/                      # Temporary image uploads
├── templates/
│   └── index.html                    # Unified agricultural dashboard template
└── utils/
    ├── disease_prediction.py         # MobileNetV2 inference engine & guardrails
    └── pesticide_recommendation.py   # Pandas-based organic remedy search engine
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.11)
- **pip** package installer

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Web Application
```bash
python app.py
```

Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)** (or `http://127.0.0.1:5000`)

---

## 🔬 Model Training & Evaluation

To re-train the model from scratch or regenerate evaluation artifacts:

```bash
python train_model.py
```

All evaluation metrics, confusion matrices, and training curves are automatically saved to `results/`.

---

## 🧪 Running Integration Tests

Run the full system test suite covering predictions, organic recommendation lookups, fallback behavior, and guardrails:

```bash
python test_system.py
```
