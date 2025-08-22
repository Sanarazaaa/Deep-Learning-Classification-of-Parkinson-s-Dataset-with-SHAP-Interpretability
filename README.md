# Deep-Learning-Classification-of-Parkinson-Dataset-with-SHAP-Interpretability

This project implements an **interpretable neural network model** for classifying samples from a Parkinson’s dataset.  
It combines **deep learning (TensorFlow/Keras)** with **explainable AI (SHAP, Random Forest feature importance)** to balance predictive performance and interpretability.  


##  Project Overview
- Built a **feedforward neural network classifier** to predict class labels from a Parkinson’s dataset  
- Preprocessed data: handled missing values, one-hot encoding, normalization, stratified train–test split  
- Evaluated performance with **accuracy, confusion matrices, ROC curves, and classification reports**  
- Applied **SHAP (SHapley Additive exPlanations)** and **Random Forest feature importance** to interpret key predictors  
- Visualized results with **t-SNE**, ROC plots, and SHAP summary plots  


##  Methods

### 1. Data Preprocessing
- Removed empty columns  
- Encoded categorical variables (one-hot, label encoding)  
- Normalized numerical features  
- Stratified train–test split  

### 2. Modeling
- Framework: TensorFlow/Keras  
- Architecture: `Dense(32) → Dense(16) → Dense(output, softmax)`  
- Loss: `sparse_categorical_crossentropy`  
- Optimizer: `Adam`  

### 3. Evaluation
- Metrics: accuracy, ROC-AUC, confusion matrix  
- Visualization: ROC curves, t-SNE  

### 4. Interpretability
- Random Forest feature importance  
- SHAP (KernelExplainer) for neural network predictions  

