# Multi-Task Facial Attribute Prediction
### A Production-Focused Deep Learning Case Study

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-red.svg)

[🎥 Live Demo](#) | [📊 Documentation](#technical-deep-dive) | [📈 Results](#model-performance) | [⚙️ Installation](#installation--usage)

---

## Overview

This project demonstrates a systematic approach to building production-grade machine learning systems through iterative improvement, rigorous testing, and honest evaluation of real-world performance. I developed three progressively sophisticated deep learning models for simultaneous prediction of age, gender, and ethnicity from facial images, achieving **84.9% gender accuracy** and **70.3% ethnicity accuracy** on test data.

**What makes this project valuable:**

- **Systematic methodology**: Clear progression from baseline to production-grade models
- **Problem-solving documentation**: Detailed solutions to class collapse and overfitting
- **Real-world validation**: Honest assessment of the gap between test metrics and deployment performance
- **Production mindset**: Focus on practical deployment challenges and limitations

---

## Key Results

### Performance Evolution

| Model | Gender Accuracy | Ethnicity Accuracy | Age MAE | Status |
|-------|----------------|-------------------|---------|---------|
| **Baseline CNN** | 61.3% | 42.9% | 7.32 years | ❌ Severe overfitting |
| **Improved CNN** | 68.8% | 52.1% | 8.38 years | ✅ Fixed overfitting |
| **VGG-Style CNN** | 84.9% | 70.3% | 6.84 years | ✅ Best test performance |

### Improvement Summary

```
Gender Accuracy:    +23.6 percentage points (+38.5% relative improvement)
Ethnicity Accuracy: +27.4 percentage points (+63.9% relative improvement)
Age Prediction:     -0.48 years MAE improvement
Class Coverage:     2/5 → 5/5 ethnicity classes (fixed critical bug)
```

---

## Technical Achievements

### 1. Resolved Critical Class Collapse
- **Problem**: Initial model only predicted 2 out of 5 ethnicity classes
- **Solution**: Implemented balanced class weighting and adjusted loss functions
- **Impact**: 121% improvement in F1-score (0.175 → 0.563)

### 2. Eliminated Severe Overfitting
- **Problem**: 4x training/validation performance gap
- **Solution**: Strategic dropout layers (0.25-0.5) and batch normalization
- **Impact**: Reduced overfitting gap by 63% (4x → 1.46x)

### 3. Achieved Strong Test Metrics
- Gender classification: 84.9% accuracy (F1: 0.838)
- Ethnicity classification: 70.3% accuracy (F1: 0.563)
- Age regression: 6.84 years mean absolute error

---

## Model Performance Analysis

### Performance Metrics by Model

The following confusion matrices and analysis charts demonstrate the progressive improvement across three model iterations:

#### Baseline CNN Results
- **Gender Classification**: 61.3% accuracy with significant class imbalance
- **Ethnicity Classification**: 42.9% accuracy with catastrophic class collapse (only 2/5 classes predicted)
- **Age Prediction**: 7.32 years MAE with poor performance on children and elderly

#### Improved CNN Results
- **Gender Classification**: 68.8% accuracy (+7.5 percentage points)
- **Ethnicity Classification**: 52.1% accuracy with all 5 classes now predicted
- **Age Prediction**: 8.38 years MAE (slight regression, but more stable)

#### VGG-Style CNN Results (Final Model)
- **Gender Classification**: 84.9% accuracy with balanced precision/recall
- **Ethnicity Classification**: 70.3% accuracy with improved F1-score (0.563)
- **Age Prediction**: 6.84 years MAE with better performance across age ranges

> **Note**: Detailed performance visualizations including confusion matrices, precision-recall curves, and age distribution analyses are available in the project notebooks and can be generated using the evaluation scripts.

### Model Comparison Summary

| Metric | Baseline | Improved | VGG-Style | Improvement |
|--------|----------|----------|-----------|-------------|
| **Gender Accuracy** | 61.3% | 68.8% | 84.9% | +23.6 pp |
| **Gender F1-Score** | 0.603 | 0.681 | 0.838 | +0.235 |
| **Ethnicity Accuracy** | 42.9% | 52.1% | 70.3% | +27.4 pp |
| **Ethnicity F1-Score** | 0.175 | 0.389 | 0.563 | +0.388 |
| **Age MAE (years)** | 7.32 | 8.38 | 6.84 | -0.48 |
| **Classes Predicted** | 2/5 | 5/5 | 5/5 | ✅ Fixed |
| **Train/Val Gap** | 4.0x | 2.1x | 1.46x | -63% |

---

## Real-World Deployment Insights

### The Reality Gap

Despite achieving 84.9% test accuracy, real-world deployment revealed significant challenges. Testing on personal photographs showed critical failure modes that weren't apparent in test metrics.

**Real-World Test Results:**

| Subject | Ground Truth | Model Prediction | Analysis |
|---------|-------------|------------------|----------|
| Adult (36M) | 36, Male, Black | 28, Male, White | Age reasonable, ethnicity incorrect |
| Adult (25F) | 25, Female, Black | 33, Male, White | All attributes incorrect |
| Child (7M) | 7, Male, Black | 83, Male, White | Catastrophic age error |

**Aggregate Performance:** 0% ethnicity accuracy, 33% gender accuracy on real-world samples

### Root Cause Analysis

Through systematic investigation, I identified six critical factors causing the train-test vs. real-world performance gap:

1. **Data Distribution Mismatch**: Training data consists primarily of celebrity photos with consistent lighting and composition
2. **Demographic Representation**: Training set heavily skewed toward certain ethnic groups
3. **Age Distribution Gaps**: Limited representation of children and elderly individuals
4. **Image Quality Variation**: Real-world photos have diverse lighting conditions and resolutions
5. **Domain Adaptation**: No transfer learning or fine-tuning on target distribution
6. **Preprocessing Differences**: Inconsistencies between training and inference pipelines

### Key Learning

**Test accuracy is a necessary but insufficient indicator of production readiness.** This project demonstrates the critical importance of:
- Out-of-distribution validation
- Diverse test datasets
- Real-world performance monitoring
- Bias detection and mitigation strategies

---

## Technical Deep Dive

### Architecture

The final VGG-style model employs a multi-task learning approach with a shared feature extractor and task-specific prediction heads:

```python
class VGGStyleMultiTask(Model):
    def __init__(self):
        super().__init__()
        
        # Shared feature extractor (VGG-inspired)
        self.backbone = Sequential([
            Conv2D(32, (3,3), padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),
            
            Conv2D(64, (3,3), padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),
            
            Conv2D(128, (3,3), padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPooling2D(),
            
            GlobalAveragePooling2D(),
            Dropout(0.5)
        ])
        
        # Task-specific prediction heads
        self.gender_head = Dense(1, activation='sigmoid')
        self.ethnicity_head = Dense(5, activation='softmax')
        self.age_head = Dense(1, activation='linear')
```

### Training Strategy

**Multi-Task Loss Function:**
```python
total_loss = (
    1.0 * binary_crossentropy(gender_true, gender_pred) +
    0.5 * categorical_crossentropy(ethnicity_true, ethnicity_pred) +
    0.1 * mean_absolute_error(age_true, age_pred)
)
```

**Optimization:**
- Optimizer: Adam with initial learning rate of 0.001
- Learning rate schedule: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: Patience of 10 epochs on validation loss
- Batch size: 32

**Regularization:**
- Dropout: 0.25-0.5 across different layers
- Batch normalization after each convolutional layer
- L2 weight regularization (1e-4)

**Data Augmentation:**
- Random horizontal flips
- Random rotations (±15°)
- Brightness adjustments (±20%)
- Contrast variations (±20%)

### Model Comparison

| Model | Parameters | Training Time | GPU Memory | Convergence |
|-------|-----------|---------------|------------|-------------|
| Baseline | 2.1M | 45 min | 4.2 GB | 30 epochs |
| Improved | 2.3M | 52 min | 4.8 GB | 35 epochs |
| VGG-Style | 3.7M | 68 min | 6.1 GB | 42 epochs |

---

## Project Structure

```
facial-attribute-prediction/
├── models/                          # Trained model artifacts
│   ├── baseline_cnn.h5
│   ├── improved_cnn.h5
│   └── vgg_style.h5
├── notebooks/                       # Experimental notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_improved_model.ipynb
│   └── 04_vgg_style_model.ipynb
├── src/                             # Source code modules
│   ├── data_preprocessing.py
│   ├── model_building.py
│   ├── training_utils.py
│   └── evaluation_metrics.py
├── app/                             # Streamlit deployment
│   ├── app.py
│   └── requirements.txt
├── tests/                           # Unit and integration tests
│   ├── test_preprocessing.py
│   └── test_models.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation & Usage

### Prerequisites
- Python 3.12+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/facial-attribute-prediction.git
cd facial-attribute-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train VGG-style model
python src/train_pipeline.py --model vgg_style --epochs 50 --batch-size 32

# Evaluate on test set
python src/evaluate.py --model models/vgg_style.h5

# Generate performance visualizations
python src/visualize_results.py --model models/vgg_style.h5 --output-dir results/
```

### Deployment

```bash
# Launch Streamlit web application
cd app
streamlit run app.py
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/
```

### Dependencies

```
tensorflow>=2.12.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.24.0
scikit-learn>=1.3.0
pillow>=10.0.0
```

---

## Skills Demonstrated

### Machine Learning & Deep Learning
- Multi-task learning architecture design
- Convolutional neural networks (CNNs)
- Regularization techniques (dropout, batch normalization)
- Hyperparameter tuning and optimization
- Class imbalance handling

### MLOps & Production
- Model versioning and experiment tracking
- Performance monitoring and evaluation
- Bias detection and fairness assessment
- Deployment pipeline development
- Real-world validation methodology

### Data Science
- Exploratory data analysis
- Statistical evaluation metrics
- Data visualization and storytelling
- Root cause analysis
- Experimental design

### Software Engineering
- Clean, modular code architecture
- Version control (Git)
- Documentation best practices
- Reproducible research
- Test-driven development

---

## Future Improvements

### Immediate Priorities

1. **Diverse Validation Dataset**
   - Collect images across demographic groups
   - Include varied lighting conditions and angles
   - Ensure balanced age distribution

2. **Advanced Data Augmentation**
   - Implement skin tone augmentation
   - Add synthetic age progression
   - Simulate diverse lighting conditions

3. **Production Monitoring**
   - Prediction distribution tracking
   - Automated bias detection alerts
   - Performance degradation monitoring

### Long-Term Vision

1. **Fairness-First Architecture**
   - Implement demographic parity constraints
   - Apply adversarial debiasing techniques
   - Develop fairness-aware loss functions

2. **Robust Production Pipeline**
   - A/B testing framework
   - Continuous retraining system
   - Canary deployment strategy

3. **Explainable AI Integration**
   - Feature attribution visualization (GradCAM)
   - Confidence calibration
   - Automated error case analysis

---

## What This Project Demonstrates

### For Technical Recruiters

This project showcases:

- **Problem-solving ability**: Diagnosed and fixed critical model failures (class collapse, overfitting)
- **Production mindset**: Focused on deployment challenges, not just test metrics
- **Critical thinking**: Questioned model performance and identified real-world limitations
- **Communication skills**: Clear documentation of both successes and failures
- **Research methodology**: Systematic approach to model improvement

### Core Competencies

✅ Deep learning model development and optimization  
✅ Production ML system design and deployment  
✅ Model debugging and performance analysis  
✅ Bias detection and fairness considerations  
✅ Clear technical communication  
✅ Honest assessment of limitations  
✅ Continuous improvement mindset

---

## Citation

If you use this work in your research or projects, please cite:

```bibtex
@software{facial_attribute_prediction,
  author = {Your Name},
  title = {Multi-Task Facial Attribute Prediction: A Production-Focused Case Study},
  year = {2025},
  url = {https://github.com/yourusername/facial-attribute-prediction}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- UTKFace dataset creators for providing the training data
- TensorFlow and Keras teams for excellent deep learning frameworks
- Streamlit for simplified deployment tooling
- The open-source ML community for inspiration and resources

---

## Connect

⭐ **Star this repository if you found it valuable**

📧 **Contact**: Yunusa Jibrin
💼 **LinkedIn**: linkedin.com/in/yunusajibrin  
🐙 **GitHub**: https://github.com/yunusajib/multitask-age-detection-ml

---

*Last updated: December 2025*