<div align="center">

# 🩺 Biobuilders — Women's Disease Prediction AI

### Advanced Medical Diagnostic Platform for Women's Health

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gemini AI](https://img.shields.io/badge/Gemini_AI-Integrated-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)

> An enterprise-grade AI platform combining machine learning, deep learning image segmentation, automated medical reporting, and a Gemini-powered AI chatbot — built to assist in the early detection of **PCOS**, **Ovarian Cysts**, and **Breast Cancer**.

**Developed by the Biobuilders Team**  
Rachit Rahaman · Prineeta Saha · Ishika Thapa · Ankita Patra · Sudipta Pariary  
*Swami Vivekananda University — April 2026*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Supported Conditions](#-supported-conditions)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Application Pages](#-application-pages)
- [AI Chatbot & Voice Module](#-ai-chatbot--voice-module)
- [Ultrasound Segmentation](#-ultrasound-segmentation)
- [Model Performance](#-model-performance)
- [API Key Setup](#-api-key-setup)
- [Dependencies](#-dependencies)
- [Training From Scratch](#-training-from-scratch)
- [Troubleshooting](#-troubleshooting)

---

## 🔬 Overview

**Biobuilders** is a full-stack medical AI application built with **Streamlit** that combines:

- **Tabular ML Predictions** — Patient data input → disease risk assessment using Logistic Regression, Random Forest & XGBoost
- **GCN Ultrasound Segmentation** — Graph Convolutional Network that precisely detects infected hypoechoic regions (cysts/follicles) in ultrasound scans
- **Disease Classification CNN** — EfficientNet-based deep learning model for 4-class diagnosis
- **Automated PDF Reports** — Professional patient reports with annotated images
- **Biobuilders AI Chat** — Gemini 1.5 Flash chatbot with voice recording and file attachment support

> Designed for **educational and research use**, providing clinicians and researchers an interactive tool for exploring AI-driven diagnostics.

---

## 🎯 Supported Conditions

| Condition | Input Type | Model Used | Accuracy |
|---|---|---|---|
| **PCOS** (Polycystic Ovary Syndrome) | Tabular patient data | Logistic Regression / Random Forest | ~94% |
| **Ovarian Cyst** | Ultrasound image | GCN Segmentation + CNN Classifier | ~94.8% |
| **Breast Cancer** | Tabular / Image data | EfficientNet B3 CNN | ~96.2% |

---

## ✨ Features

### 🤖 AI & Machine Learning
- Multiple classifiers: Logistic Regression, Random Forest, XGBoost
- Deep learning: TensorFlow/Keras + PyTorch (EfficientNet B3)
- GCN-based ultrasound image segmentation (`gcn_weights.pth`)
- Pre-trained serialized models ready for instant inference

### 🖼️ Ultrasound Image Processing
- Upload and preprocess ultrasound scans (OpenCV + PIL)
- **Clinical GCN V24 segmentation** — Dynamically detects 1–3 darkest hypoechoic cyst regions
- Automatic elimination of acoustic shadows, scan border voids, and caliper artifacts
- Annotated heatmap overlay exported for PDF reports

### 📊 Interactive Dashboard
- Gauge charts, pie charts, radar charts (Plotly)
- Patient profile analysis with per-metric status cards
- Real-time EDA with Pandas + Matplotlib/Seaborn
- Confusion matrices and model performance metrics

### 📄 Automated PDF Reports
- ReportLab-powered professional medical reports
- Embedded annotated ultrasound images
- Patient demographics + clinical findings + recommendations
- One-click download directly from the app

### 💬 Biobuilders AI Chatbot
- Powered by **Gemini 1.5 Flash**
- 🎙️ **Voice recording** via `st.audio_input` — record, stop, send
- 📎 **File attachments** — images, PDFs, audio files
- Streaming responses with animated text output
- Full multi-turn conversation history
- API quota handling with retry guidance

### 🎨 Premium UI/UX
- Aurora Borealis animated background
- Glassmorphism cards with hover effects
- Custom cursor, Google Fonts (Outfit + Quicksand)
- Mobile-responsive sidebar with auto-close
- Dark-mode first design

---

## 📁 Project Structure

```
Women_Disease_App_Final/
│
├── 📂 src/                          # Main application source code
│   ├── app.py                       # ⭐ Main Streamlit application
│   ├── gcn_segmentation.py          # GCN-based ultrasound segmentation (V24)
│   ├── disease_classifier.py        # Unified disease prediction engine
│   ├── report_generator.py          # PDF report automation (ReportLab)
│   ├── data_processing.py           # ETL — loading & preprocessing
│   ├── dataset_builder.py           # Dataset construction tools
│   ├── train_models.py              # ML model training pipeline
│   ├── train_pipeline.py            # End-to-end training orchestration
│   ├── train_segmentation.py        # GCN segmentation training
│   ├── train_gcn.py                 # GCN model training script
│   ├── generate_training_data.py    # Synthetic training data generator
│   ├── prepare_data.py              # Feature engineering utilities
│   ├── evaluate.py                  # Metrics, confusion matrices, ROC
│   ├── visualization.py             # EDA charts & performance plots
│   ├── utils.py                     # Helper functions
│   └── __init__.py                  # Package initialization
│
├── 📂 data/                         # Training datasets
│   ├── breast_cancer.csv            # ~120KB — Breast cancer dataset
│   ├── ovarian_cyst.csv             # ~80KB  — Ovarian cyst dataset
│   └── pcos.csv                     # ~119KB — PCOS patient dataset
│
├── 📂 models/                       # Pre-trained model weights
│   ├── gcn_weights.pth              # GCN segmentation model
│   ├── pcos_Logistic_Regression.joblib
│   └── pcos_Random_Forest.joblib
│
├── 📂 .streamlit/                   # Streamlit configuration
│   ├── config.toml                  # Theme & server config
│   └── secrets.toml                 # 🔑 API keys (see setup below)
│
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies
├── svu_logo.jpg                     # University logo
└── README.md                        # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python **3.8 or higher**
- pip package manager
- ~2 GB disk space (for models + dependencies)
- A free [Gemini API key](https://aistudio.google.com/apikey) (for chatbot)

### 1. Open the Project

```bash
cd "Women_Disease_App_Final"
```

### 2. Create a Virtual Environment

```bash
# Create environment
python -m venv .venv

# Activate — Windows
.\.venv\Scripts\activate

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

Create or edit `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

> 🔑 Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 5. Run the App

```bash
streamlit run src/app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📱 Application Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Welcome screen with platform overview |
| 📋 **Patient Prediction** | Enter clinical data → get PCOS/breast cancer risk scores with gauges & radar charts |
| 🔬 **Ultrasound Analysis** | Upload ultrasound → GCN segmentation → infected region detection |
| 🏥 **Disease Classification** | Upload image → CNN classifies disease category with heatmap |
| 📊 **Model Training** | Interactive training configuration and progress tracking |
| 📈 **Original EDA** | Dataset explorer — charts and statistics for all 3 diseases |
| 🤖 **AI Consult** | Gemini AI chatbot with voice input, file upload, streaming responses |
| ℹ️ **About** | Platform credits, tech stack, methodology |

---

## 🎙️ AI Chatbot & Voice Module

Navigate to **🤖 AI Consult** in the sidebar.

### Text Chat
Type your question in the chat bar at the bottom and press **Enter**.

### Voice Recording
1. Click **🎙️ Voice Message — Click to Record** to expand the voice section
2. Press the **microphone button** to begin recording
3. Click **Stop** when you're done speaking
4. Click **📤 Send Voice Message** — Gemini will transcribe and respond

### File Attachments
Use the **📎 paperclip icon** in the chat bar to attach:
- Images (PNG, JPG, JPEG) — the AI can analyze medical scans
- PDFs — the AI can read and discuss report contents
- Audio files (WAV, MP3)

### API Quota Notes
The free Gemini tier has per-minute and daily limits. If you hit a quota:
- Wait ~60 seconds and retry (per-minute limit)
- After the daily limit, wait until the next morning
- Get a second free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) using a different Google account

---

## 🩻 Ultrasound Segmentation

The **Clinical GCN V24** segmentation engine (`gcn_segmentation.py`) uses a multi-stage pipeline to precisely isolate infected regions:

### How It Works

1. **Scan Zone Detection** — Identifies the valid ultrasound fan cone, ignoring the surrounding black void
2. **Bottom Shadow Cutoff** — Automatically removes the deep acoustic attenuation zone (bottom 15%) where uninfected shadows appear
3. **Border Void Elimination** — Strips a safe inner border so sweep-edge darkness is never flagged
4. **Hypoechoic Thresholding** — Applies a pixel intensity threshold (`< 60`) inside the safe inner zone to isolate fluid-filled cysts (which appear dark on ultrasound)
5. **Morphological Bridging** — Reconnects cyst regions that may be fragmented by caliper measurement annotations
6. **Shape & AI Validation** — Each candidate region is evaluated for:
   - **Solidity** (`> 0.45`) — ensures blob-like structure, not thin lines
   - **Aspect Ratio** (`< 1.8`) — rejects vertical acoustic drop shadows
   - **AI Confidence** — GCN neural network confidence score acts as a secondary validator
7. **Ranking & Output** — Top 1–2 darkest confirmed cysts are highlighted with a green contour overlay

### What It Detects
| Condition | Size Range | Appearance |
|---|---|---|
| Ovarian Cyst | Large (5–15cm) | Single large dark rounded pocket |
| PCOS Follicles | Small (2–9mm) | Multiple small dark circles |

---

## 📊 Model Performance

| Model | Task | Accuracy | F1-Score |
|---|---|---|---|
| Logistic Regression | PCOS Risk | ~94% | 0.93 |
| Random Forest | PCOS Risk | ~94% | 0.94 |
| GCN Segmentation | Ultrasound Infection Detection | ~94.8% | Dice: 0.924 |
| EfficientNet B3 CNN | Disease Classification (4-class) | ~96.2% | 0.962 |

**Per-class CNN metrics:**

| Disease | Precision | Recall | F1-Score |
|---|---|---|---|
| Normal | 0.98 | 0.95 | 0.965 |
| Ovarian Cyst | 0.96 | 0.97 | 0.965 |
| PCOS | 0.94 | 0.90 | 0.920 |
| Breast Cancer | 0.97 | 0.96 | 0.965 |

---

## 🔑 API Key Setup

The Gemini AI chatbot requires a Google API key.

**Option A — Streamlit Secrets (recommended):**

Edit `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "AIza..."
```

**Option B — Environment Variable:**
```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "AIza..."
streamlit run src/app.py

# macOS/Linux
export GEMINI_API_KEY="AIza..."
streamlit run src/app.py
```

**Get a free API key:** https://aistudio.google.com/apikey

---

## 📦 Dependencies

```
Core Data & ML
├── pandas >= 1.3.0
├── numpy >= 1.21.0
├── scikit-learn >= 1.0.0
└── xgboost >= 1.5.0

Deep Learning
├── tensorflow >= 2.8.0
├── keras >= 2.8.0
├── torch >= 2.0.0
└── torchvision >= 0.15.0

Image Processing
├── opencv-python >= 4.5.0
├── pillow >= 8.0.0
├── scikit-image >= 0.19.0
└── albumentations >= 1.3.0

Web UI & Visualization
├── streamlit >= 1.39.0
├── plotly >= 5.0.0
├── matplotlib >= 3.4.0
└── seaborn >= 0.11.0

Reporting
├── reportlab >= 4.0.0
└── pypdf >= 3.0.0

AI Integration
├── google-generativeai >= 0.3.0
└── joblib >= 1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training From Scratch

```bash
# Train all ML classifiers (PCOS, Breast Cancer)
python src/train_pipeline.py

# Train GCN segmentation model
python src/train_gcn.py

# Evaluate trained models
python src/evaluate.py
```

> **Note:** Pre-trained weights in `models/` are ready to use out of the box. Training from scratch requires a labelled ultrasound dataset.

---

## 🛠️ Adding a New Disease

1. Add dataset CSV to `data/`
2. Extend `data_processing.py` with a new loader function
3. Add training logic in `train_models.py`
4. Register predictor in `disease_classifier.py`
5. Add a new UI tab/section in `src/app.py`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` on startup | Run `pip install -r requirements.txt` |
| GCN module fails to load | `pip install opencv-python torch torchvision` |
| Chatbot shows "API Key not found" | Add key to `.streamlit/secrets.toml` |
| Voice recording not working | Requires Streamlit ≥ 1.39 — `pip install --upgrade streamlit` |
| Model loading fails | Verify `models/` contains `.joblib` and `.pth` files |
| Port 8501 already in use | `streamlit run src/app.py --server.port 8502` |
| PDF report generation fails | `pip install reportlab pypdf` |
| Gemini quota exceeded | Wait 60s (per-min) or next day (daily) — or use a new free API key |
| Segmentation marks entire image | Image may not be a valid ultrasound scan — try a clearer image |

---

## ⚠️ Disclaimer

> This platform is developed for **educational and research purposes only**. AI predictions are **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

Educational use only. All rights reserved by the Biobuilders team, Swami Vivekananda University, 2026.

---

<div align="center">

**Built with ❤️ by Biobuilders 🩺**

*Rachit Rahaman · Prineeta Saha · Ishika Thapa · Ankita Patra · Sudipta Pariary*

*Swami Vivekananda University · April 2026*

</div>
