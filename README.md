# Telco Customer Churn Prediction

A machine learning project to predict **customer churn** using the **IBM Telco Customer Churn dataset**. The project focuses on handling **class imbalance**, evaluating models beyond accuracy, and improving **churn recall**, which is critical for business decisions.

---

## 📊 Dataset

**Source:** IBM / Kaggle – Telco Customer Churn

**Target Variable:** `Churn`

* `0` → No Churn
* `1` → Churn

**Class Distribution:**

* ~73% Non-Churn
* ~27% Churn (imbalanced)

---

## 🎯 Problem Statement

Customer churn directly impacts revenue. The goal is to build a predictive model that:

* Accurately identifies customers likely to churn
* Prioritizes **recall for churners** over raw accuracy
* Handles imbalanced data effectively

---

## 🧠 Approach

1. **Data Cleaning & Preprocessing**

   * Handled missing values (`TotalCharges`)
   * Encoded categorical features
   * Scaled numerical features

2. **Exploratory Data Analysis (EDA)**

   * Churn rate by contract type
   * Monthly charges vs churn
   * Tenure distribution

3. **Modeling**

   * Baseline models (Logistic Regression)
   * Tree-based models (Random Forest / Gradient Boosting)
   * Class imbalance handling using:

     * `class_weight`
     * SMOTE (optional)

4. **Evaluation Metrics**

   * Precision
   * Recall (Churn = 1)
   * F1-score
   * Accuracy

---

## 📈 Model Performance

Models trained and evaluated:

* Logistic Regression (baseline)
* Random Forest
* XGBoost (gradient boosting)



**Key Insight:**
- Achieved **80% accuracy**
- Captured **78% of churners**, with scope to improve recall via threshold tuning

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Imbalanced-learn (SMOTE)

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py
````

or open the notebook:

```bash
jupyter notebook telco_churn.ipynb
```

---



---

## 📌 Key Takeaway

> In imbalanced classification problems like churn prediction, **recall and F1-score are more important than accuracy alone**.

---

## 📄 License

This project is for educational and portfolio purposes.
