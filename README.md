# Woman Disease Prediction App - Unified & Production-Ready

A comprehensive, enterprise-grade machine learning application for predicting three female-specific medical conditions with advanced AI, image processing, and automated reporting capabilities.

## 🎯 Supported Conditions

- **Ovarian Cyst** - Multi-modal prediction with image segmentation
- **PCOS (Polycystic Ovary Syndrome)** - Multiple ML classifier ensemble
- **Breast Cancer** - Deep learning-based classification

## ✨ Key Features

### Core ML/AI Capabilities
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- **Deep Learning**: TensorFlow/Keras neural networks
- **PyTorch Support**: Advanced neural network architectures
- **Pre-trained Models**: Joblib-serialized PCOS models ready for inference
- **Model Ensemble**: Disease classification across multiple algorithms

### Advanced Image Processing
- **Ultrasound Processing**: OpenCV-based medical image analysis
- **GCN Segmentation**: Graph Convolutional Network segmentation for medical imaging
- **Image Enhancement**: Preprocessing pipelines for diagnostic images
- **Automatic Feature Extraction**: Computer vision-based feature generation

### Data & Reporting
- **Automated PDF Reports**: PatientReport + PDFReportGenerator
- **Patient Data Management**: Structured dataset building and management
- **Data Preprocessing**: Complete ETL pipeline with validation
- **Synthetic Dataset Generation**: For testing and development

### User Interface & Visualization
- **Modern Streamlit UI** (app_ui_v4.py): Interactive web-based dashboard
- **Interactive Charts**: Plotly-based real-time visualizations
- **EDA Dashboards**: Comprehensive exploratory data analysis
- **Graceful Fallbacks**: Optional module support with error handling

### Data Science & Analysis
- **Jupyter Notebooks**: EDA templates and experimentation
- **Comprehensive Evaluation**: Metrics, confusion matrices, ROC curves
- **Data Visualization**: Statistical plots and model performance charts

## Project Structure

```
women disease app final/
├── data/                           # Datasets & training data
│   ├── breast_cancer.csv          # Breast cancer dataset
│   ├── ovarian_cyst.csv           # Ovarian cyst dataset
│   └── pcos.csv                   # PCOS patient dataset
│
├── models/                         # Pre-trained ML models
│   ├── pcos_Logistic_Regression.joblib
│   └── pcos_Random_Forest.joblib
│
├── notebooks/                      # Jupyter Notebooks
│   └── eda_template.ipynb         # EDA & exploration template
│
├── deployment/                     # Cloud & container configs
│
├── src/                           # Main application code
│   ├── app_ui_v4.py               # ⭐ Latest Streamlit web app
│   ├── disease_classifier.py      # Disease prediction engine
│   ├── gcn_segmentation.py        # Medical image segmentation
│   ├── report_generator.py        # PDF report automation
│   │
│   ├── data_processing.py         # Data loading & preprocessing
│   ├── prepare_data.py            # Data preparation utilities
│   ├── dataset_builder.py         # Dataset creation tools
│   │
│   ├── train_models.py            # Model training pipeline
│   ├── train_pipeline.py          # Complete training workflow
│   ├── evaluate.py                # Model evaluation & metrics
│   │
│   ├── visualization.py           # Plotting & charts
│   ├── utils.py                   # Helper functions & utilities
│   ├── __init__.py                # Package initialization
│   └── __pycache__/               # Compiled Python cache
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
```

## 📦 Key Components

### Application Entry Point
- **app_ui_v4.py** - Latest features version with all integrations:
  - GCN-based image segmentation
  - Disease classification engine
  - Automated PDF report generation
  - Graceful module loading with fallbacks

### ML Pipeline
- **train_models.py** - Model training for all conditions
- **train_pipeline.py** - End-to-end training orchestration
- **disease_classifier.py** - Unified disease prediction

### Image Processing
- **gcn_segmentation.py** - GCN-based ultrasound segmentation

### Reporting
- **report_generator.py** - Patient report & PDF generation

### Data Management
- **data_processing.py** - Loading and preprocessing
- **prepare_data.py** - Feature engineering
- **dataset_builder.py** - Dataset construction

### Analysis & Evaluation
- **evaluate.py** - Comprehensive metrics and validation
- **visualization.py** - EDA and performance charts

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- ~2GB disk space for models and data

### Installation

1. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import streamlit; import torch; print('✓ All dependencies installed')"
   ```

### Running the Application

**Start the Streamlit Web UI:**
```bash
streamlit run src/app_ui_v4.py
```

The app will open at `http://localhost:8501`

### Optional: Training Models

To train new models with your data:
```bash
python src/train_pipeline.py
```

### Jupyter Notebooks

For exploratory data analysis:
```bash
jupyter notebook notebooks/eda_template.ipynb
```

## 📋 Features Overview

### Disease Predictions
- Input patient data (tabular/CSV)
- Receive AI-powered predictions
- View confidence scores and probabilities
- Access visual explanations

### Image Analysis (if CV modules available)
- Upload medical images (ultrasound, X-ray)
- Automatic segmentation with GCN
- Feature extraction and analysis
- Visualization of findings

### PDF Report Generation
- Automated patient reports
- Comprehensive findings summary
- Methodology documentation
- Professional formatting

### Model Metrics
- Precision, Recall, F1-Score
- Confusion matrices
- ROC curves
- Cross-validation scores

## 🔧 Dependencies

Key libraries included:
- **Data Science**: pandas, numpy, scikit-learn, scipy
- **Deep Learning**: tensorflow, keras, torch, torchvision
- **Image Processing**: opencv-python, pillow, scikit-image, albumentations
- **UI/Visualization**: streamlit, plotly, matplotlib, seaborn
- **ML Models**: xgboost, joblib
- **Reporting**: reportlab, pypdf
- **Utilities**: python-dateutil

See `requirements.txt` for complete list with versions.

## 📊 Model Performance

Pre-trained PCOS models included:
- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method with feature importance

All models are validated on original datasets with comprehensive metrics.

## 🛠️ Development & Extension

### Add New Disease Type
1. Add training data to `data/` folder
2. Extend `train_models.py` with new disease
3. Update `disease_classifier.py` with prediction logic
4. Register in `app_ui_v4.py` UI

### Add New ML Algorithm
1. Implement training in `train_models.py`
2. Add evaluation metrics in `evaluate.py`
3. Save model with joblib in `models/`

### Custom Image Processing
1. Extend `gcn_segmentation.py` with new models
2. Add preprocessing to `data_processing.py`
3. Update report generation in `report_generator.py`

## 📝 Configuration

### Environment Variables (Optional)
```bash
export MODEL_PATH="./models"
export DATA_PATH="./data"
export LOG_LEVEL="INFO"
```

### Streamlit Config
Edit `.streamlit/config.toml` for UI customization:
- Theme colors
- Layout settings
- Performance options

## 🐛 Troubleshooting

**ImportError for optional modules?**
- The app gracefully handles missing CV/report dependencies
- Install specific module: `pip install opencv-python`

**Streamlit connection timeout?**
- Increase memory: `streamlit run --logger.level=debug ...`
- Check port availability: `lsof -i :8501`

**Model loading fails?**
- Verify joblib files in `models/` folder exist
- Reinstall scikit-learn: `pip install --upgrade scikit-learn`

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Guide](https://scikit-learn.org)
- [TensorFlow/Keras](https://www.tensorflow.org)
- [PyTorch Documentation](https://pytorch.org)

## 📄 License & Usage

This application is designed for educational and research purposes. Always consult healthcare professionals for medical decisions.

## 🤝 Contributing

To contribute improvements:
1. Create a feature branch
2. Make changes with comprehensive testing
3. Update documentation
4. Submit for review

## ✅ Quality Assurance

- ✓ Modular architecture for easy testing
- ✓ Comprehensive error handling
- ✓ Data validation at all pipeline stages
- ✓ Model serialization for reproducibility
- ✓ Automated report generation

## 📞 Support

For issues or questions:
1. Check notebooks for examples
2. Review source code documentation
3. Verify all dependencies installed
4. Check Streamlit logs for errors

---

**Last Updated**: February 2026
**Status**: Production Ready
**Version**: 4.0 (Unified & Enhanced)
