# 🏏 IPL Match Winner Prediction using Ensemble Machine Learning

## 📌 Problem Statement

Predicting the outcome of an IPL match before it is played is a challenging task due to multiple influencing factors such as team strength, venue advantage, toss decisions, and historical performance.

This project builds a machine learning-based classification model that predicts:

> **Which team is most likely to win an IPL match before the match begins.**

---

## 🎯 Objective

The primary goals of this project are to:

* Predict the match winner before the game starts.
* Capture team strength dynamically over time.
* Incorporate venue and home-ground advantage.
* Build a robust ensemble learning model to improve prediction accuracy.

---

## 📂 Dataset

The dataset is not included in this repository due to GitHub's file size limitations.

You can download it directly from Kaggle using the Kaggle API:

```bash
#!/bin/bash
kaggle datasets download chaitu20/ipl-dataset2008-2025
```

After downloading, place the dataset in the `data/` directory:

```text
ipl-match-winner/
│
├── data/
│   └── IPL.csv
│
├── notebooks/
├── src/
├── README.md
└── requirements.txt
```

---

## 🧠 Methodology

### 1. Data Preprocessing

* Loaded IPL match-level data derived from ball-by-ball records.
* Aggregated match information using `match_id`.
* Removed records with missing or unknown winners.
* Filtered out teams with very few matches to reduce noise and improve model stability.

---

### 2. Feature Engineering

Several domain-specific features were created to improve predictive performance.

#### 🔹 Team Encoding

* Standardized team names using a mapping dictionary.
* Ensured consistency across different IPL seasons.

#### 🔹 Dynamic Team Strength

Calculated historical win ratios before each match to avoid data leakage.

Features created:

* `batting_strength`
* `bowling_strength`

These metrics represent each team's performance up to that point in time.

#### 🔹 Home Advantage

Created binary indicators to capture home-ground advantage:

* `batting_home`
* `bowling_home`

These were determined using predefined home venues for each team.

#### 🔹 Match Context Features

Included additional match-level information:

* Toss winner
* Toss decision
* Venue
* City
* Match date (year, month, day)

---

### 3. Dataset Preparation

#### Target Variable

```text
match_won_by
```

#### Feature Types

**Categorical Features**

* Teams
* Venue
* City
* Toss winner
* Toss decision

**Numerical Features**

* Team strength metrics
* Home advantage indicators
* Date-related features

Categorical variables were encoded using **One-Hot Encoding**, while numerical features were passed directly to the model.

---

### 4. Ensemble Model

#### 🤖 Voting Classifier

The final prediction model combines:

* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

**Voting Strategy:** Soft Voting

Soft voting averages predicted probabilities from all base models and selects the class with the highest combined probability.

##### Why an Ensemble?

* Reduces variance and bias.
* Combines strengths of multiple algorithms.
* Produces more stable and accurate predictions.
* Improves generalization on unseen matches.

---

### 5. Training Strategy

To maintain the chronological nature of sports data, the model was trained using:

```python
TimeSeriesSplit(n_splits=5)
```

Benefits:

* Prevents future information leakage.
* Preserves match chronology.
* Provides a realistic evaluation framework.

---

### 6. Machine Learning Pipeline

A complete end-to-end pipeline was implemented using Scikit-learn.

#### Components

* **ColumnTransformer**

  * Handles preprocessing of categorical and numerical features.

* **Pipeline**

  * Integrates preprocessing and model training into a single workflow.

This ensures reproducibility and simplifies deployment.

---

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report

> **Note:** Performance metrics were computed using time-series cross-validation to simulate real-world prediction scenarios.

---

## 📈 Key Insights

* Teams with stronger historical win rates tend to have a higher probability of winning future matches.
* Home-ground advantage provides a measurable performance boost.
* Toss outcomes and decisions can significantly influence match results.
* Ensemble models outperform individual models in predictive accuracy and stability.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM

---

## 📌 Project Goal

To build a reliable, data-driven IPL match prediction system that leverages historical performance, team strength, venue effects, and match conditions to predict winners with high accuracy and practical relevance.
