
# 🧪 Cervical Cancer Risk Prediction Using Machine Learning

This project builds and evaluates machine learning models to predict **cancer risk** based on patient health and lifestyle data. It is designed as a data science and machine learning assignment, following a full pipeline from data cleaning to model evaluation.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Tech Stack](#tech-stack)
4. [Installation](#installation)
5. [Project Workflow](#project-workflow)
6. [Models Implemented](#models-implemented)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [How to Run](#how-to-run)
10. [Future Improvements](#future-improvements)

---

## 📌 Project Overview

Cancer is one of the leading global health challenges. Early detection of Cervical cancer risk can significantly improve patient outcomes.
This project uses **machine learning** techniques to analyze patient health data and predict whether a patient is at risk of cancer.

The notebook includes:

* Data preprocessing (cleaning, imputing missing values, encoding).
* Feature selection (excluding target-leakage features like screening test results).
* Training multiple machine learning classifiers.
* Comparing performance with confusion matrices, ROC curves, and cross-validation.

---

## 📂 Dataset

The dataset includes patient health and lifestyle attributes such as:

* **Demographics:** Age, Gender.
* **Health Factors:** Smoking, Alcohol intake, Obesity, Diet.
* **Screening Results:** Cancer test outcomes (removed to prevent leakage).
* **Target Variable:** Whether the patient is at risk of cancer (`Cancer` = 0 or 1).

👉 **Note:** The dataset used in this project was provided for coursework and may not be publicly available.

---

## 🛠 Tech Stack

* **Language:** Python (Jupyter Notebook)
* **Libraries:**

  * `pandas`, `numpy` – data handling
  * `matplotlib`, `seaborn` – visualization
  * `scikit-learn` – ML models and evaluation
  * `imblearn` – handling imbalanced datasets

---

## 🔄 Project Workflow

1. **Data Loading & Exploration**

   * Load dataset and inspect structure.
   * Check missing values, class balance, and distributions.

2. **Data Preprocessing**

   * Handle missing values with `SimpleImputer`.
   * Encode categorical variables.
   * Drop features causing **data leakage** (e.g., direct cancer test results).
   * Split data into **train (80%)** and **test (20%)** sets with stratification.

3. **Model Training & Evaluation**

   * Build ML pipelines for Logistic Regression, Random Forest, and Support Vector Machine.
   * Use cross-validation for robust performance estimation.
   * Evaluate with accuracy, precision, recall, F1-score, ROC-AUC.

4. **Visualization**

   * Confusion matrices for classification errors.
   * ROC curves to compare model performance.
   * Feature importance plot (Random Forest).

---

## 🤖 Models Implemented

1. **Logistic Regression** (with class balancing)
2. **Random Forest Classifier** (ensemble learning, feature importance)
3. **Support Vector Machine (SVM)** (with RBF kernel & probability outputs)

---

## 📊 Evaluation Metrics

The following metrics were used to assess model performance:

* **Accuracy** – Overall correctness.
* **Precision** – Ability to correctly identify cancer-positive patients.
* **Recall (Sensitivity)** – Ability to detect all actual cancer cases.
* **F1-score** – Balance of precision and recall.
* **ROC-AUC** – Area under the ROC curve (discrimination ability).

---

## ✅ Results

* **Random Forest** achieved the best balance of recall and precision.
* **Logistic Regression** performed consistently and explained feature contributions.
* **SVM** worked well after scaling, though slower on large datasets.

📌 Key Insights:

* Lifestyle and health-related factors strongly influence cancer risk predictions.
* Handling **missing values** and preventing **data leakage** are critical for trustworthy results.

---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/cancer-risk-prediction.git
   cd cancer-risk-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `ml.ipynb` and run all cells sequentially.

---

## 🔮 Future Improvements

* Include **additional datasets** (e.g., genetic and environmental factors).
* Try **advanced models** like XGBoost, LightGBM, or Neural Networks.
* Deploy as a **web app** using Flask/Django + React for healthcare providers.
* Implement **explainability tools** (e.g., SHAP, LIME) for model interpretation.


