# ML Challenge 2025: Smart Product Pricing

## 1. Executive Summary
We developed a **text-centric price prediction system** using extensive feature engineering on product metadata and an ensemble of gradient boosting regression models. Systematic experimentation showed that **TF-IDF–based lexical features consistently outperformed semantic and multimodal approaches**, achieving a best validation **SMAPE** of **0.5674**. Image-based and transformer-based methods provided minimal additional benefit relative to their complexity.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The task was formulated as a **supervised regression problem** to predict product prices from unstructured catalog text and associated images. Exploratory Data Analysis revealed a **highly skewed, heavy-tailed price distribution**, motivating the use of log-transformed targets for model stability.

Analysis of the ```catalog_content``` field indicated that **brand identifiers, quantity indicators (IPQ), and specification keywords** were the strongest pricing signals, while visual similarity across products did not consistently correlate with price.

**Key Observations:**
- Price distribution is long-tailed with significant outliers.
- Most pricing signal is embedded in textual metadata.
- Numeric and quantity cues embedded in text are critical.
- Image information alone is weakly correlated with price.

### 2.2 Solution Strategy
A **feature-engineering–first approach** was adopted, prioritizing interpretable, high-signal textual representations and robust classical machine learning models. Multiple model families were evaluated, with final predictions generated using an ensemble of gradient boosting models, optimized using SMAPE-based validation.

**Approach Type:** Ensemble (tree-based regression)  
**Core Innovation:** Demonstrating that **sparse lexical features combined with domain-specific heuristics outperform complex multimodal embeddings** for large-scale e-commerce price prediction under SMAPE evaluation.

---

## 3. Model Architecture

### 3.1 Architecture Overview
The final pipeline consists of:

**1**. Text preprocessing and normalization
**2**. TF-IDF vectorization and text-derived statistical features
**3**. Training multiple gradient boosting regressors
**4**. Ensemble averaging of predictions
**5**. Inverse transformation of log-scaled outputs


### 3.2 Model Components

**Text Processing Pipeline:**
- [ ] Preprocessing steps: 
    - [] Lowercasing
    - [] Special Character Removal
    - [] Token Normalization
- [ ] Model type: 
    - [] TF-IDF (word- and character-level n-grams)
    - [] Text length and keyword indicators
- [ ] Key parameters: 
    - [] Tuned n-gram ranges
    - [] Feature dimensionality controlled empirically

**Image Processing Pipeline:**
- [ ] Preprocessing steps: 
    - [] Image resizing and normalization
- [ ] Model type: 
    - [] Frozen CLIP / CNN embeddings
- [ ] Key parameters:
    - [] Evaluated during experimentation but excluded from final model due to limited performance gains


---


## 4. Model Performance

### 4.1 `feature_engineering.py` Validation Results
- **LightGBM Cross-Validation**
    - **SMAPE:** 0.5749 ± 0.0052
    - **MAE:** $12.99 ± $0.18
    - **RMSE:** $29.92
- **XGBOOST Cross-Validation** 
    - **SMAPE:** 0.5779 ± 0.0050
    - **MAE:** $13.05 ± $0.18
    - **RMSE:** $29.97
- **CATBOOST Cross-Validation** 
    - **SMAPE:** 0.5762 ± 0.0068
    - **MAE:** $13.12 ± $0.23
    - **RMSE:** $31.23

### 4.2 `clip_feature.py` Validation Results
- **SMAPE:** 56.402

### Performance Notes:
- Ensemble models consistently outperformed individual regressors.
- TF-IDF–based models outperformed BERT-style semantic embeddings.
- Image-based features provided marginal or inconsistent improvement.


## 5. Conclusion
This solution demonstrates that **well-engineered textual features combined with ensemble tree models** provide a strong and reliable approach for large-scale product price prediction. Despite exploring multimodal and deep learning techniques, **simpler, domain-aware methods proved more effective** under the SMAPE metric. The project highlights the importance of feature quality, metric alignment, and systematic experimentation in applied machine learning.

---

## Appendix

### A. Code artefacts
Complete source code repository including feature engineering, model training, inference pipeline, and documentation.


### B. Additional Results
Feature importance visualizations, comparative experiment results, and error analysis across price ranges.

---
