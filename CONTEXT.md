# ai-from-scratch — Project Context

> This file exists for one purpose: paste it at the start of every new Claude 
> thread so Claude has full context without needing memory of previous sessions.

---

## 📁 Repo
- **GitHub:** https://github.com/Ansul-S/ai-from-scratch
- **Local:** ~/Desktop/ai-from-scratch
- **Stack:** Python, NumPy (implementations), pandas, matplotlib, 
  seaborn (EDA), scikit-learn (benchmarking + reuse), pytest (tests)

---

## 🎯 Goal
Build a library of AI/ML mini-projects from scratch for:
1. Applied AI Engineer job portfolio
2. Interview preparation
3. Community learning resource

---

## 📐 Standards (apply to every project)

### Code
- OOP class-based implementations
- New algorithms built with NumPy only
- Previously built algorithms → use sklearn (see rule below)
- solution.py contains all classes
- train.py is a CLI script runnable from terminal

### Notebooks
- `01_eda.ipynb` — data exploration and insights only
- `02_implementation.ipynb` — building and experimenting

### Tests
- Minimum 5 pytest unit tests per project
- All tests must pass before project is marked complete

### README
- Intuition first, math second, code third
- Math-to-code table (equation → line of code)
- 10 interview questions with full answers
- Common mistakes & gotchas section
- Exercises (beginner / intermediate / advanced)

---

## ⚡ The From-Scratch Rule

> Only build NEW algorithms from scratch.
> If already built in a previous project → use sklearn instead.

| Already Built | Use Going Forward |
|--------------|-------------------|
| LinearRegression | `sklearn.linear_model.LinearRegression` |
| Ridge regularization | `sklearn.linear_model.Ridge` |
| Train/test split | `sklearn.model_selection.train_test_split` |
| StandardScaler | `sklearn.preprocessing.StandardScaler` |
| PolynomialFeatures (basic) | Build full version in Project 02, then sklearn after |

---

## 📁 File Structure (every project)

project-name/
├── data/
│   ├── raw/                 # untouched data
│   ├── processed/           # cleaned / feature engineered
│   └── external/            # optional (APIs, downloads)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── model.py           # core model (e.g., LinearRegression)
│   │   └── train.py           # training logic
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── utils/
│       └── helpers.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_train.py
│
├── configs/                  # optional but powerful
│   └── config.yaml
│
├── results/
│   ├── plots/
│   └── models/               # saved weights
│
├── requirements.txt
├── .gitignore
├── README.md
└── run.py                    # main entry point

---

## ✅ Completed Projects

### Project 01 — Linear Regression
- **Dataset:** Medical Insurance Charges (1338 rows, 7 features)
- **Target:** charges (annual insurance cost)
- **Built from scratch:**
  - `LinearRegression` — fit (gradient descent + Ridge l2), predict, 
     score, mse, fit_normal_equation
  - `PolynomialFeatures` — single feature, single column
  - Manual StandardScaler (raw numpy, no class)
- **Key results:**
  - R² = 0.75 with all features
  - Adding smoker alone jumps R² from 0.12 → 0.75
  - Gradient descent and Normal Equation produce identical results
  - Ridge with λ>1 causes significant underfitting on this dataset
- **Tests:** 5/5 passing
- **Files:** solution.py, notebook.ipynb, tests/, results/ (8 plots)

---

## 🔄 Current Project

### Project 02 — Polynomial Regression
- **Dataset:** Bike Sharing Demand — hour.csv (UCI)
- **Target:** cnt (hourly bike rentals)
- **Features:** temp, hum, windspeed, hr, season, workingday, weathersit
- **Build from scratch:**
  - `PolynomialFeatures` — full version, all features + interaction terms
  - `PolynomialRegression` — wrapper class
- **Use sklearn for:**
  - LinearRegression, Ridge, StandardScaler, train_test_split
- **New additions vs Project 01:**
  - Split notebooks (01_eda + 02_implementation)
  - train.py CLI with argparse
  - sklearn benchmark test
- **Key concepts:**
  - Why linear regression fails on curved data
  - Overfitting vs underfitting visualised
  - Bias-variance tradeoff
  - Degree selection via train vs test R² curve
  - Interaction terms
- **Plots to save in results/:**
  - temp_vs_rentals_scatter.png
  - hourly_patterns.png
  - degree_experiment.png ← most important
  - fitted_curves_comparison.png
  - ridge_on_polynomial.png
- **Tests:** 5 minimum
  - PolynomialFeatures output shape correct
  - Degree 1 == linear regression
  - Higher degree fits train better
  - Ridge shrinks weights
  - sklearn benchmark within tolerance
- **Status:** 🔄 In progress — setup complete, starting EDA

---

## ⏳ Upcoming Projects

| # | Project | Phase | Status |
|---|---------|-------|--------|
| 03 | Logistic Regression | Phase 1 | ⏳ |
| 04 | K-Nearest Neighbours | Phase 1 | ⏳ |
| 05 | Naive Bayes | Phase 1 | ⏳ |
| 06 | Decision Tree | Phase 1 | ⏳ |
| 07 | K-Means Clustering | Phase 1 | ⏳ |
| 08 | PCA | Phase 1 | ⏳ |
| 09 | Random Forest | Phase 2 | ⏳ |
| 10 | Gradient Boosting | Phase 2 | ⏳ |
| 11 | XGBoost from Scratch | Phase 2 | ⏳ |
| 12 | SVM | Phase 2 | ⏳ |

---

## 🗣️ How I Learn Best
- Guide me step by step
- Don't give direct solutions — ask questions first
- Let me write the code, you review it
- Ask me to explain concepts in my own words
- Correct my understanding when wrong, but explain why

---

