# 02 — Polynomial Regression from Scratch

> **Difficulty:** 🟢 Easy  
> **Time to complete:** ~8–10 hours  
> **Phase:** Phase 1 — Classical ML Foundations  
> **Tags:** `regression` `polynomial-features` `bias-variance-tradeoff` `overfitting` `numpy` `supervised-learning`

---

## 📌 What You'll Learn

By the end of this project you will be able to:

- [ ] Explain why linear regression fails on non-linear data — with a real plot to prove it
- [ ] Build a `PolynomialFeatures` class from scratch that generates all power and interaction terms
- [ ] Build a `StandardScaler` class from scratch with fit, transform, and inverse_transform
- [ ] Wrap everything into a clean `PolynomialRegression` pipeline class
- [ ] Explain the bias-variance tradeoff and diagnose underfitting vs overfitting from a plot
- [ ] Use Ridge regularization to prevent coefficient explosion in high-degree models
- [ ] Select the optimal polynomial degree using cross-validation (RidgeCV)
- [ ] Build and run a CLI training script using `argparse`
- [ ] Write 5 meaningful pytest unit tests for your implementation
- [ ] Explain every design decision in a technical interview

---

## 🆕 New Concepts vs Project 01

This project builds directly on Project 01 (Linear Regression). Here's what's genuinely new:

| Concept | Project 01 | Project 02 |
|---------|-----------|-----------|
| Feature engineering | Raw features only | Polynomial expansion + interactions |
| Model complexity | Fixed (linear) | Variable (degree 1 to 10) |
| Bias-variance tradeoff | Mentioned briefly | Core experiment — plotted and analysed |
| Overfitting diagnosis | Train vs test R² | Full degree experiment curve |
| Regularization | Ridge as an option | Essential — prevents coefficient explosion |
| Degree selection | N/A | Cross-validation with RidgeCV |
| Code structure | Single notebook | Split notebooks + CLI train.py |
| Testing | 5 basic tests | 5 tests including benchmark comparison |
| StandardScaler | Raw numpy (no class) | Full class with fit/transform/inverse |

---

## 📦 Dataset — Bike Sharing Demand

**File:** `data/hour.csv`  
**Source:** [UCI Machine Learning Repository — Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)  
**Size:** 17,379 rows × 17 columns

### Features Used

| Feature | Type | Description |
|---------|------|-------------|
| `atemp` | float | Normalised "feels-like" temperature (0–1) |
| `hum` | float | Normalised humidity (0–1) |
| `windspeed` | float | Normalised wind speed (0–1) |
| `hr` | int | Hour of day (0–23) |
| `season` | int | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| `workingday` | int | 1 if working day, 0 otherwise |
| `cnt` | int | **Target** — Total bike rentals per hour |

### Columns Dropped and Why

| Column | Reason |
|--------|--------|
| `casual` + `registered` | **Target leakage** — `cnt = casual + registered`. Using them is cheating. |
| `instant` | Just a row number — carries zero predictive information |
| `dteday` | Date string — already encoded as `hr`, `mnth`, `yr`, `season` |
| `temp` | Kept `atemp` instead — people decide to bike based on how it *feels*, not the thermometer reading |

> ⚠️ **Target leakage is the most dangerous mistake in ML.** Your model scores perfectly in training and fails completely in production. Always ask: "Would this feature be available at prediction time, and does it directly encode the target?"

### Why this dataset?

This dataset is perfect for polynomial regression because `atemp` vs `cnt` forms a clear **inverted-U curve** — rentals rise as temperature becomes comfortable, then fall as it gets too hot. A straight line cannot capture this. The moment you plot it, polynomial regression justifies itself visually.

---

## 💡 Intuition First

### Why Does Linear Regression Fail Here?

In Project 01, we drew the best straight line through data. That works when the relationship is roughly linear — like age vs insurance charges.

But imagine plotting temperature vs bike rentals:

- At 0°C (freezing) → very few people rent bikes
- At 20°C (pleasant) → maximum rentals
- At 38°C (sweltering) → people stay home again

That's an **inverted-U shape**. A straight line through this data will always be wrong — it either misses the peak, or gets the direction wrong entirely. This is called **underfitting** — when your model's assumptions are too simple for the real pattern.

### The Key Insight

Polynomial regression doesn't change the algorithm. It changes the **features**.

Instead of feeding in `[atemp]`, you feed in `[atemp, atemp²]`. Now the model can learn:

```
rentals = w₁×atemp + w₂×atemp² + bias
```

With a negative `w₂`, this equation produces a parabola — exactly the inverted-U you need. You're still doing linear regression internally. You just gave it richer inputs.

> **Interview-ready line:** *"Polynomial regression doesn't change the algorithm — it changes the feature space. You're still doing linear regression, just on transformed features."*

### The Real-World Analogy

Think of a baker adjusting oven temperature. Too cold → bread doesn't bake. Too hot → bread burns. There's a sweet spot. A model that only knows "higher temperature = better" will always recommend maximum heat. A polynomial model can learn the sweet spot.

### What is the Bias-Variance Tradeoff?

Every model makes two kinds of errors:

**Bias** — error from wrong assumptions. A straight line through an inverted-U curve has high bias. The model is systematically wrong regardless of how much data you give it.

**Variance** — error from sensitivity to noise. A degree-10 polynomial wiggles to fit every quirk in the training data. Change the training set slightly and the curve changes dramatically.

The tradeoff:
- **High bias (underfit):** model too simple → misses real patterns
- **High variance (overfit):** model too complex → memorises noise, fails on new data
- **Sweet spot:** just complex enough to capture real patterns, not noise

The total error decomposes as:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

You can reduce bias by adding complexity. But complexity increases variance. The goal is to minimise their sum.

---

## 🔢 How It Works — The Math

### Polynomial Feature Expansion

For a single feature $x$ and degree $d$, we generate:

$$[x, x^2, x^3, \ldots, x^d]$$

For multiple features, we also generate **interaction terms** — products of pairs:

For features $[a, b]$ and degree 2:

$$[a,\ b,\ a^2,\ b^2,\ a \cdot b]$$

The general count formula for $n$ features at degree 2:

$$\text{Total columns} = n + n + \frac{n(n-1)}{2} = 2n + \frac{n(n-1)}{2}$$

| Features (n) | Original | Squared | Interactions | Total |
|---|---|---|---|---|
| 2 | 2 | 2 | 1 | 5 |
| 3 | 3 | 3 | 3 | 9 |
| 6 | 6 | 6 | 15 | 27 |

> ⚠️ **Feature explosion is real.** Our 6 features at degree=2 become 27. At degree=3, it grows further. This is why regularization is essential.

### Why Interaction Terms?

`atemp × hum` captures something neither feature captures alone. High temperature AND high humidity together make biking miserable — more than either factor separately. The interaction term lets the model learn this joint effect.

```
cnt = w₁×atemp + w₂×hum + w₃×(atemp×hum) + ...
```

The coefficient `w₃` captures: "when both temperature and humidity are high, reduce the prediction further."

### Coefficient Explosion at High Degrees

At degree=10, polynomial features become astronomically large. Feature values like `atemp^10` for `atemp=0.8` give `0.8^10 ≈ 0.107` — fine here because `atemp` is normalised. But in general, high-degree terms cause:

1. Extreme multicollinearity — `atemp^9` and `atemp^10` are nearly identical
2. Huge coefficients — the model compensates by making weights enormous
3. Numerical instability — matrix operations overflow or lose precision

**The fix: Ridge regularization.** Adding $\lambda \sum w_j^2$ to the loss makes large weights expensive, forcing them to stay small regardless of degree.

### Ridge Regression Recap

$$J(\theta) = \frac{1}{m} \sum(\hat{y} - y)^2 + \lambda \sum w_j^2$$

The gradient update becomes:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T(\hat{y} - y) + \lambda w$$

- $\lambda = 0$ → standard regression, no penalty
- $\lambda$ too small → coefficients still explode at high degrees
- $\lambda$ too large → all weights shrink to zero → underfitting
- $\lambda = 0.1$ → our sweet spot, confirmed by 5-fold cross-validation

### StandardScaler — The Math

Scaling transforms each feature to have mean=0 and std=1:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

Where $\mu$ and $\sigma$ are computed from **training data only**.

To reverse (inverse transform):

$$x_{original} = x_{scaled} \times \sigma + \mu$$

**Critical:** always fit the scaler on training data, then apply the same transformation to test data. Fitting on test data leaks future information and makes your model look artificially better.

**Why scale after polynomial expansion?**

If `atemp` is in `[0, 1]`, `atemp²` is also in `[0, 1]` — fine. But if a feature is in `[0, 100]`, squaring gives `[0, 10,000]`. The gradient landscape becomes wildly uneven — gradient descent zigzags instead of converging. Always expand first, scale second.

### R² and RMSE

R² from Project 01:

$$R^2 = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$$

New metric this project — **RMSE** (Root Mean Squared Error):

$$RMSE = \sqrt{\frac{1}{m}\sum(\hat{y} - y)^2}$$

RMSE is in the same units as the target (bike rentals). An RMSE of 122 means your predictions are off by ~122 rentals on average. Interpretable. Useful.

### From Math to Code

| Math | Code |
|------|------|
| $x^d$ for each feature | `cols.append(X ** d)` |
| $x_i \cdot x_j$ interaction | `X[:, i] * X[:, j]` |
| $x_{scaled} = \frac{x - \mu}{\sigma}$ | `(X - self.mean_) / self.std_` |
| $x_{original} = x_{scaled} \cdot \sigma + \mu$ | `(X * self.std_) + self.mean_` |
| $RMSE = \sqrt{\frac{1}{m}\sum error^2}$ | `np.sqrt(np.mean((y_pred - y)**2))` |

---

## 🏗️ Build From Scratch

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pytest
```

### Project Structure

```
02-polynomial-regression/
├── data/
│   └── hour.csv
├── results/
│   ├── cnt_distribution.png
│   ├── temp_vs_rentals_scatter.png
│   ├── hourly_patterns.png
│   ├── seasonal_breakdown.png
│   ├── correlation_heatmap.png
│   ├── degree_experiment.png
│   ├── fitted_curves_comparison.png
│   └── ridge_on_polynomial.png
├── tests/
│   ├── __init__.py
│   └── test_solution.py
├── 01_eda.ipynb
├── 02_implementation.ipynb
├── solution.py
├── train.py
└── README.md
```

---

### Step 1 — Exploratory Data Analysis (`01_eda.ipynb`)

#### Load and Inspect

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/hour.csv')
print(df.shape)        # (17379, 17)
print(df.dtypes)       # mix of int64 and float64
print(df.isnull().sum()) # All zeros — clean dataset
```

#### Feature Selection

```python
drop_cols = ['instant', 'dteday', 'casual', 'registered', 'temp']
features  = ['atemp', 'hum', 'windspeed', 'hr', 'season', 'workingday']
target    = 'cnt'

df = df.drop(columns=drop_cols)
df[features + [target]].describe()
```

**Key findings from describe():**
- `cnt` mean=189, max=977 — massive right skew
- `atemp` is normalised: min=0.06, max=1.0
- All features are already numeric — no encoding needed

#### Target Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['cnt'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('cnt — Raw Distribution')
axes[1].hist(np.log1p(df['cnt']), bins=50, color='coral', edgecolor='white')
axes[1].set_title('cnt — Log Transformed')
plt.tight_layout()
plt.savefig('results/cnt_distribution.png', dpi=150)
plt.show()
```

**Why `log1p` and not `log`?**
`log(0)` = −∞ which crashes everything. `log1p(x)` computes `log(1 + x)` — safe when x=0. Inverse operation is `np.expm1()`. Use `log1p` as a habit on any count data.

#### The Most Important Plot — atemp vs cnt

```python
plt.figure(figsize=(8, 5))
plt.scatter(df['atemp'], df['cnt'], alpha=0.1, color='steelblue', s=10)
plt.title('Feels-Like Temperature vs Bike Rentals')
plt.xlabel('atemp (normalized)')
plt.ylabel('cnt (rentals)')
plt.savefig('results/temp_vs_rentals_scatter.png', dpi=150)
plt.show()
```

**What you'll see:** A clear inverted-U cloud. Rentals peak around atemp=0.6 and drop at both extremes. This single plot justifies the entire project — a straight line cannot fit this.

#### Hourly Patterns

```python
hourly = df.groupby('hr')['cnt'].mean()
plt.figure(figsize=(10, 4))
plt.plot(hourly.index, hourly.values, marker='o', color='steelblue')
plt.title('Average Rentals by Hour of Day')
plt.xlabel('Hour')
plt.xticks(range(0, 24))
plt.savefig('results/hourly_patterns.png', dpi=150)
plt.show()
```

**Two clear peaks:** 8am (morning commute) and 5-6pm (evening commute). `hr` is one of the strongest features — but note we keep it as integer (0–23) rather than one-hot encoding, which would create 24 columns and massively inflate the feature space after polynomial expansion.

---

### Step 2 — The `PolynomialFeatures` Class (`solution.py`)

The upgraded version handles ALL features and generates interaction terms — unlike Project 01's single-feature version.

```python
import numpy as np
from itertools import combinations

class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X):
        n_samples, n_features = X.shape
        cols = [X]  # start with original features

        for d in range(2, self.degree + 1):
            # Power terms: each feature raised to degree d
            cols.append(X ** d)

            # Interaction terms: only at degree 2
            if d == 2:
                for i, j in combinations(range(n_features), 2):
                    interaction = X[:, i] * X[:, j]
                    cols.append(interaction.reshape(-1, 1))

        return np.hstack(cols)
```

**Key design decisions:**

1. **No bias column** — sklearn's `LinearRegression` adds its own intercept via `fit_intercept=True`. Adding a column of ones here would create a duplicate bias term — messy and wrong.

2. **`combinations(range(n_features), 2)`** — generates every unique pair `(i, j)` where `i < j`. For 3 features: `(0,1), (0,2), (1,2)`. No duplicates, no self-pairs.

3. **`reshape(-1, 1)`** — interaction `X[:, i] * X[:, j]` gives shape `(n,)`. `np.hstack` requires `(n, 1)`. The `-1` tells numpy to infer the dimension automatically.

**Verify the output shape:**

```python
X_test = np.random.randn(100, 2)
pf = PolynomialFeatures(degree=2)
print(pf.fit_transform(X_test).shape)  # (100, 5) — 2 + 2 + 1
```

---

### Step 3 — The `StandardScaler` Class (`solution.py`)

```python
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X):
        X = np.array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_  = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1, self.std_)
        return self

    def transform(self, X):
        X = np.array(X)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        X = np.array(X)
        return (X * self.std_) + self.mean_
```

**Notable additions over a naive implementation:**

- `np.where(self.std_ == 0, 1, self.std_)` — if a feature is constant (std=0), dividing by zero gives NaN. Replacing with 1 leaves the feature unchanged instead of crashing.
- `ValueError` in `inverse_transform` — fails loudly if you try to reverse-transform before fitting. Silent failures are worse than loud ones.

---

### Step 4 — The `PolynomialRegression` Wrapper (`solution.py`)

```python
from sklearn.linear_model import Ridge as SklearnRidge

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly   = PolynomialFeatures(degree=degree)
        self.scaler = StandardScaler()
        self.model  = SklearnRidge(alpha=0.1)

    def fit(self, X, y):
        X_poly   = self.poly.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_poly   = self.poly.fit_transform(X)
        X_scaled = self.scaler.transform(X_poly)   # transform only — NOT fit
        return self.model.predict(X_scaled)

    def score(self, X, y):
        X_poly   = self.poly.fit_transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.score(X_scaled, y)
```

**Why `transform` in predict, not `fit_transform`?**

The scaler learned mean and std from training data in `fit()`. In `predict()` you apply those exact same parameters to new data. If you called `fit_transform()` in predict, you'd refit on test data — leaking future information and getting artificially good metrics.

**Why Ridge instead of plain LinearRegression?**

At high degrees, polynomial features are highly correlated — `atemp^9` and `atemp^10` are nearly identical. Plain `LinearRegression` assigns enormous weights to compensate, causing numerical instability (predictions of `1e14`). Ridge penalises large weights and keeps everything stable.

---

### Step 5 — The Core Experiment: Degree 1 to 10

```python
from solution import PolynomialRegression
from sklearn.model_selection import train_test_split

X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = range(1, 11)
train_scores, test_scores = [], []

for degree in degrees:
    model = PolynomialRegression(degree=degree)
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))
```

**Results:**

| Degree | Train R² | Test R² | Verdict |
|--------|----------|---------|---------|
| 1 | 0.3365 | 0.3456 | Underfit — wrong assumption |
| 2 | 0.4807 | 0.4735 | Better, still underfit |
| **3** | **0.5246** | **0.5275** | **Sweet spot** |
| 4 | 0.5252 | 0.5289 | Negligible gain |
| 10 | 0.5617 | 0.5685 | Diminishing returns |

**Key observation:** Test R² slightly exceeds Train R² across all degrees. This is a sign of a healthy, stable model — 17k rows is large enough that the test set isn't harder than training. No classical overfitting occurred.

**Optimal degree = 3** because:
- Largest single jump: degree 2→3 (+0.054 test R²)
- Degree 3→4: only +0.001 — not worth the added complexity
- The model has learned what it can from these 6 features at degree 3

---

### Step 6 — Ridge Alpha Selection

```python
from sklearn.linear_model import RidgeCV

poly   = PolynomialFeatures(degree=3)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(poly.fit_transform(X_train))
X_test_scaled  = scaler.transform(poly.fit_transform(X_test))

ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train_scaled, y_train)

print(f"Best alpha (5-fold CV): {ridge_cv.alpha_}")   # 0.1
print(f"Test R²: {ridge_cv.score(X_test_scaled, y_test):.4f}")
```

**Alpha experiment results:**

| Alpha | Train R² | Test R² |
|-------|----------|---------|
| 0.001 | 0.5246 | 0.5276 |
| 0.1 | 0.5246 | 0.5275 |
| 1 | 0.5244 | 0.5267 |
| 10 | 0.5188 | 0.5182 |
| 100 | 0.5010 | 0.4981 |
| 1000 | 0.4634 | 0.4660 |

The flat region (alpha 0.001 → 1) confirms degree=3 doesn't overfit badly — Ridge adds stability without hurting performance. Once alpha > 10, over-penalisation causes underfitting.

---

### Step 7 — CLI Training Script (`train.py`)

```python
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from solution import PolynomialFeatures, StandardScaler, PolynomialRegression

def main():
    parser = argparse.ArgumentParser(description='Train Polynomial Regression')
    parser.add_argument('--degree',    type=int,   default=3)
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    df       = pd.read_csv('data/hour.csv')
    features = ['atemp', 'hum', 'windspeed', 'hr', 'season', 'workingday']
    target   = 'cnt'

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)

    model = PolynomialRegression(degree=args.degree)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    train_r2    = model.score(X_train, y_train)
    test_r2     = model.score(X_test, y_test)
    rmse        = np.sqrt(np.mean((y_test_pred - y_test) ** 2))

    # Sklearn benchmark — Ridge on raw features
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train, y_train)
    benchmark_r2 = ridge.score(X_test, y_test)

    print(f"\nTraining Polynomial Regression (degree={args.degree})...")
    print(f"Features: {', '.join(features)}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test  R²: {test_r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print(f"Ridge benchmark R²: {benchmark_r2:.4f}")

if __name__ == '__main__':
    main()
```

**Run from terminal:**

```bash
cd projects/phase-1-classical-ml/02-polynomial-regression
python train.py --degree 3
python train.py --degree 1   # matches benchmark exactly
python train.py --degree 10  # shows diminishing returns
```

**Sample output (degree=3):**
```
Training Polynomial Regression (degree=3)...
Features: atemp, hum, windspeed, hr, season, workingday
Train R²: 0.5246
Test  R²: 0.5275
RMSE:     122.3172
Ridge benchmark R²: 0.3456
```

> Your polynomial model beats the linear baseline by **+0.18 R²** — that's the entire justification for polynomial regression in one number.

---

### Step 8 — Unit Tests (`tests/test_solution.py`)

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution import PolynomialFeatures, StandardScaler, PolynomialRegression

def test_polynomial_features_shape():
    """Output shape: 2 features, degree=2 → 5 columns"""
    X = np.random.randn(100, 2)
    pf = PolynomialFeatures(degree=2)
    assert pf.fit_transform(X).shape == (100, 5)

def test_degree1_equals_linear():
    """Degree=1 polynomial regression == Ridge linear baseline"""
    np.random.seed(42)
    X = np.random.randn(200, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + np.random.randn(200) * 0.1

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler as SkScaler

    poly_model = PolynomialRegression(degree=1)
    poly_model.fit(X, y)

    scaler = SkScaler()
    ridge  = Ridge(alpha=0.1)
    ridge.fit(scaler.fit_transform(X), y)

    assert abs(poly_model.score(X, y) - ridge.score(scaler.transform(X), y)) < 0.01

def test_higher_degree_higher_train_r2():
    """More complexity → better training fit"""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = X[:, 0]**2 + np.random.randn(200) * 0.1

    m1, m3 = PolynomialRegression(degree=1), PolynomialRegression(degree=3)
    m1.fit(X, y); m3.fit(X, y)
    assert m3.score(X, y) > m1.score(X, y)

def test_standard_scaler():
    """Scaled output has mean≈0 and std≈1"""
    X = np.random.randn(100, 3) * 50 + 20
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_scaled.std(axis=0),  1, atol=1e-7)

def test_polynomial_beats_linear():
    """Polynomial model outperforms linear on non-linear data"""
    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = X[:, 0]**2 + X[:, 1]**2 + np.random.randn(300) * 0.1

    m1, m2 = PolynomialRegression(degree=1), PolynomialRegression(degree=2)
    m1.fit(X, y); m2.fit(X, y)
    assert m2.score(X, y) > m1.score(X, y)
```

**Run:**
```bash
pytest tests/test_solution.py -v
# 5 passed in 0.61s ✅
```

---

## ⚠️ Common Mistakes & Gotchas

1. **Fitting the scaler in `predict()`** — Always the most common bug. `scaler.fit_transform()` in `fit()`, `scaler.transform()` in `predict()` and `score()`. Never fit on test data.

2. **Scaling before polynomial expansion** — Wrong order. Expand first, then scale the whole expanded matrix. Squaring an unscaled feature like `age` (range 20–80) gives `age²` with range 400–6400 — gradient descent breaks.

3. **Using plain `LinearRegression` for high degrees** — Coefficient explosion. Predictions of `1e14` are not a model, they're a numerical crash. Always use Ridge with polynomial features.

4. **Duplicate bias column** — If `PolynomialFeatures` adds a column of ones AND sklearn's `LinearRegression` has `fit_intercept=True`, you have two bias terms. Perfect multicollinearity. Never add the bias column in `PolynomialFeatures`.

5. **Confusing Test > Train R² as overfitting** — It's the opposite. Test slightly > Train means the model is stable and the test set happened to be slightly easier. Overfitting is Train >> Test.

6. **Choosing degree by training R² alone** — Training R² always increases with degree. Always use test R² or cross-validation to select degree. The training curve tells you nothing about generalisation.

7. **Target leakage** — `casual + registered = cnt`. Using either as a feature makes your model perfect and completely useless. Always ask: "would this feature be available before the target is known?"

8. **Dropping features with low linear correlation** — `windspeed` has near-zero correlation with `cnt` linearly. But strong winds non-linearly suppress rentals. Correlation measures linear relationships only. Low correlation ≠ useless feature.

---

## 🎯 Interview Questions

<details>
<summary><strong>Q1: What is the bias-variance tradeoff? How did you observe it in this project?</strong></summary>

**Answer:** The bias-variance tradeoff describes the fundamental tension between two sources of model error. Bias is error from wrong assumptions — a linear model on curved data has high bias regardless of how much data you have. Variance is error from sensitivity to training data — a degree-10 polynomial wiggles to fit noise and changes dramatically with different training splits. Total error = Bias² + Variance + Irreducible Noise.

In this project, degree=1 showed high bias — the straight line systematically underpredicted peak temperatures. Degree=10 showed diminishing returns and slight instability. Degree=3 was the sweet spot: test R² was highest relative to added complexity, confirmed by the degree experiment plot where test R² stopped improving meaningfully after degree=3.

</details>

<details>
<summary><strong>Q2: How does polynomial regression relate to linear regression mathematically?</strong></summary>

**Answer:** Polynomial regression IS linear regression — on a transformed feature space. Given a feature `x`, polynomial regression creates new features `x², x³, ...` and feeds them all into a standard linear regression. The model is still linear in its parameters (weights). The equation `y = w₁x + w₂x² + w₃x³ + b` is non-linear in `x` but linear in `w₁, w₂, w₃`. This is why we can use the same gradient descent algorithm — we haven't changed the learning mechanism, only the inputs.

</details>

<details>
<summary><strong>Q3: How do you choose polynomial degree? What's wrong with using training R²?</strong></summary>

**Answer:** Training R² always increases monotonically with degree — more parameters always fit training data better. This makes it useless for degree selection. You need a measure of generalisation. The correct approach is to plot train vs test R² across degrees and find where test R² stops improving meaningfully — the point of diminishing returns. More rigorously, use k-fold cross-validation (RidgeCV in sklearn) which averages performance across multiple validation sets, reducing the effect of any single lucky or unlucky split. In our project, both approaches agreed: degree=3 with alpha=0.1.

</details>

<details>
<summary><strong>Q4: Why is Ridge regularization essential for polynomial regression?</strong></summary>

**Answer:** Polynomial features of high degree are inherently multicollinear — `x^9` and `x^10` are nearly identical. To distinguish their effects, plain linear regression assigns enormous, opposing coefficients that cancel out. This causes numerical instability (coefficients of `1e14`) and terrible predictions on new data. Ridge adds a penalty `λΣw²` to the loss, making large weights expensive. This forces coefficients to stay small and stable regardless of degree. The result: smooth curves instead of explosive wiggles. In our project, plain LinearRegression produced predictions of `1e14` at degree=10. Ridge fixed it immediately with alpha=0.1.

</details>

<details>
<summary><strong>Q5: What are interaction terms and why do they matter?</strong></summary>

**Answer:** Interaction terms are products of two features — for example `atemp × hum`. They let the model capture effects that depend on two features simultaneously. High temperature alone might increase rentals. High humidity alone might slightly decrease them. But high temperature AND high humidity together make biking miserable — far worse than either effect alone. An interaction term `atemp × hum` lets the model assign a separate weight to this combined condition. Without interaction terms, the model assumes all features act independently — which is rarely true in real data.

</details>

<details>
<summary><strong>Q6: What is target leakage and why is it dangerous?</strong></summary>

**Answer:** Target leakage occurs when a feature in your training data directly encodes the target — information that wouldn't be available at prediction time. In our dataset, `casual` and `registered` sum to `cnt` (our target). Including them gives R²=1.0 in training and complete failure in production — because you'd never know `casual` and `registered` before knowing `cnt`. Leakage is dangerous because it's invisible from metrics — your model looks perfect. The only defence is domain knowledge: always ask "would this feature be available before the target is known?"

</details>

<details>
<summary><strong>Q7: Why must you scale after polynomial expansion, not before?</strong></summary>

**Answer:** Scaling before expansion changes what the polynomial terms represent. But the real problem is scale explosion: if a feature is in range `[0, 100]`, squaring gives `[0, 10,000]`. If you scaled first to `[-1, 1]` and then squared, you'd get `[0, 1]` — which is fine. But if other polynomial terms have different ranges, the gradient landscape becomes uneven and gradient descent struggles. The safest practice is always: expand features first to generate all polynomial and interaction terms, then scale the entire expanded matrix together. This ensures every column entering the model has comparable scale.

</details>

<details>
<summary><strong>Q8: Your test R² is slightly higher than train R² across all degrees. Is the model overfitting or underfitting?</strong></summary>

**Answer:** Neither — this is a sign of a stable, healthy model. The classical overfitting pattern is Train R² >> Test R², meaning the model memorised training data and fails on new data. Test slightly > Train is the opposite — it means the test set happened to have slightly easier patterns to predict, or the model's regularization worked well. With 17k training rows and only 6 features at degree=3, the model has far more data than complexity. There's no reason to overfit. The concerning signal would be a large gap in the wrong direction, not a tiny gap in the "right" direction.

</details>

<details>
<summary><strong>Q9: When would you use polynomial regression vs a tree-based model like Random Forest?</strong></summary>

**Answer:** Polynomial regression is appropriate when: (1) you have a clear, smooth non-linear relationship you can visualise — like the inverted-U in temperature vs rentals; (2) interpretability matters — polynomial coefficients have clear mathematical meaning; (3) the dataset is small and you want to avoid overfitting that tree models can exhibit. Tree-based models are better when: (1) relationships are complex, non-smooth, and multi-way; (2) you have many features with unknown interaction structure; (3) outliers and non-Gaussian distributions are present; (4) you need state-of-the-art accuracy and interpretability is secondary. In practice, start with polynomial regression for intuition and simplicity, then escalate to trees if performance is insufficient.

</details>

<details>
<summary><strong>Q10: Explain what `combinations(range(n_features), 2)` does and why it's used for interaction terms.</strong></summary>

**Answer:** `combinations(range(n_features), 2)` generates every unique pair of feature indices `(i, j)` where `i < j`. For 3 features it gives `(0,1), (0,2), (1,2)` — the 3 unique pairs. We use it to generate interaction terms `X[:, i] * X[:, j]` for each pair. The key word is "unique" — `combinations` never produces `(1, 0)` if it already produced `(0, 1)`, and never produces `(0, 0)` (self-pairs). This avoids duplicate features (which cause multicollinearity) and avoids squared terms (which are already handled separately by `X ** d`). It's the correct mathematical tool for generating all pairwise interaction terms without repetition.

</details>

---

## 🏋️ Exercises & Challenges

**🟢 Beginner**
- [ ] Plot degree=1, 3, 10 fitted curves on the `atemp` vs `cnt` scatter — what shape does each curve take?
- [ ] Run `train.py --degree 1` and observe that Test R² matches the Ridge benchmark exactly. Why?
- [ ] Try `alpha` values `[0.001, 1, 1000]` and plot how the fitted curve changes shape.
- [ ] Add `RMSE` to the degree experiment table. Does RMSE decrease as degree increases?

**🟡 Intermediate**
- [ ] Add an `--alpha` argument to `train.py` so users can control regularization strength from the CLI.
- [ ] Implement `fit_transform` for degree=3 interaction terms (`a²b`, `ab²`) — extend `PolynomialFeatures` to handle this.
- [ ] Apply `np.log1p` transform to `cnt` before training. Does RMSE (on original scale, after `expm1`) improve?
- [ ] Add `mnth` and `weathersit` to your feature list. Does Test R² improve meaningfully?

**🔴 Advanced**
- [ ] Implement k-fold cross-validation from scratch (without sklearn) to select degree — compare with RidgeCV's answer.
- [ ] Try `hr` as one-hot encoded (24 columns) instead of integer. Does performance improve? What happens to feature count after expansion?
- [ ] Add Lasso regularization option to `PolynomialRegression`. Which features does Lasso zero out at high alpha?
- [ ] Compare `PolynomialRegression(degree=3)` against `sklearn.ensemble.RandomForestRegressor` on the same split. Which wins?

---

## 🔗 What's Next

- **[03 — Logistic Regression →](../03-logistic-regression/)** — Adapts gradient descent to classification using the sigmoid function. You'll reuse the same pipeline thinking from this project.
- **[06 — Decision Tree →](../06-decision-tree/)** — The tree-based alternative to polynomial regression for non-linear data. Compare when each approach wins.

---

## 📚 Further Reading

- [CS229 Lecture Notes — Regularization](https://cs229.stanford.edu/main_notes.pdf) — Stanford's derivation of Ridge and Lasso from first principles
- [StatQuest — Bias-Variance Tradeoff](https://www.youtube.com/@statquest) — Best visual explanation of the tradeoff with simulated data
- [Elements of Statistical Learning — Chapter 3](https://hastie.su.domains/ElemStatLearn/) — Rigorous treatment of linear methods and regularization
- [Python `itertools` docs](https://docs.python.org/3/library/itertools.html) — `combinations`, `permutations`, and other combinatorial tools used in feature engineering

---

<p align="center">
  <strong>ai-from-scratch</strong> · Project 02 · Polynomial Regression<br>
  Built with curiosity · Shared with the community
</p>