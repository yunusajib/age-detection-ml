# 🎭 Multi-Task Facial Attribute Prediction

Deep learning model for simultaneous prediction of age, gender, and ethnicity from facial images

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Achievements](#-key-achievements)
- [Technical Highlights](#-technical-highlights)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Architecture](#-model-architecture)
- [Development Journey](#-development-journey)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)


## 🎯 Overview

This project implements a **multi-task convolutional neural network** that simultaneously predicts three facial attributes from a single image:
-  **Age** (regression: 0-100+ years)
-  **Gender** (binary classification: Male/Female)
-  **Ethnicity** (5-class classification)

### Problem Statement

Traditional single-task models require separate networks for each attribute, leading to:
-  Higher computational costs (3 separate models)
-  Inefficient feature learning (repeated processing)
-  Deployment complexity (managing multiple models)

**Solution:** A unified multi-task learning architecture that shares feature extraction across all three tasks, improving efficiency and enabling knowledge transfer between related tasks.

---

##  Key Achievements

### Performance Improvements

    | Metric | Baseline | Improved | Gain | % Improvement |
    |--------|----------|----------|------|---------------|
    | **Ethnicity Accuracy** | 42.9% | **54.7%** | +11.8% | **+27.5%**  |
    | **Gender Accuracy** | 61.3% | **71.2%** | +9.9% | **+16.1%** |
    | **Ethnicity F1-Score** | 0.175 | **0.387** | +0.212 | **+121%**  |
    | **Overfitting Reduction** | 4.0x gap | **1.46x gap** | -2.54x | **-63.5%**  |

### Critical Problem Solved: Class Collapse

    **Before:** Model predicted only 2 out of 5 ethnicity classes (catastrophic failure)
    ```
    Classes predicted: [0, 2]  ← 3 classes completely ignored
    ```

    **After:** Model now predicts ALL 5 ethnicity classes with balanced distribution
    ```
    Classes predicted: [0, 1, 2, 3, 4]  ← All classes represented 
    ```


## 🔧 Technical Highlights

### Problems Identified & Solutions Implemented

#### 1. Severe Overfitting (4x training-validation gap)
**Solution:**
    -  Added 7 dropout layers (0.25-0.5 rates) throughout architecture
    -  Implemented data augmentation (rotation, flip, brightness, contrast, zoom)
    -  Reduced learning rate from 1e-3 to 1e-4
    -  Added learning rate scheduler with plateau detection

#### 2. Class Imbalance (6:1 ratio between majority and minority classes)
**Solution:**
    -  Analyzed class distribution and calculated balanced weights
    -  Implemented custom `WeightedSparseCategoricalCrossentropy` loss
    -  Applied class weights: 0.47x (majority) to 2.81x (minority)

#### 3. Limited Generalization
**Solution:**
    -  Data augmentation pipeline with 5 augmentation types
    -  Batch normalization after each convolutional layer
    -  Extended training with increased early stopping patience

### Implementation Details

- **Framework:** TensorFlow/Keras 2.x
- **Dataset:** UTKFace (18,964 training samples)
- **Architecture:** Custom CNN with shared backbone + task-specific heads
- **Training:** 22 epochs, batch size 32, Adam optimizer
- **Regularization:** Dropout (0.5), BatchNormalization, Data Augmentation
- **Loss Functions:** MSE (age) + Weighted Categorical Crossentropy (gender, ethnicity)

---

## 📊 Results

### Model Performance Comparison

![Baseline vs Improved](age-detection-ml-fresh/src/visualizations/baseline_vs_improved.png)<div align="center">
⭐ Star this repo if you find it helpful!
Made with ❤️ and lots of ☕
</div>