# Project 03 — Logistic Regression
### Binary Classification: Breast Cancer Detection

> **Difficulty:** 🟡 Medium  
> **Time to complete:** ~8–10 hours  
> **Phase:** Phase 1 — Classical ML Foundations  
> **Tags:** `classification` `sigmoid` `binary-cross-entropy` `gradient-descent` `confusion-matrix` `precision-recall` `numpy` `supervised-learning`

---

## 📌 What You'll Learn

By the end of this project you will be able to:

- [ ] Explain precisely why linear regression fails for classification — not just "outputs aren't bounded"
- [ ] Derive the sigmoid function and explain what it represents geometrically
- [ ] Implement binary cross-entropy loss from scratch and explain why MSE is mathematically wrong for classification
- [ ] Build a full `LogisticRegression` class with gradient descent, loss tracking, and validation loss
- [ ] Interpret a confusion matrix and explain the cost of each error type in context
- [ ] Navigate the precision-recall tradeoff and tune the decision threshold for a specific use case
- [ ] Explain why the loss always starts at exactly 0.693 — and what that tells you
- [ ] Write 5 meaningful pytest unit tests including a sklearn benchmark comparison
- [ ] Explain every design decision in a technical interview

---

## 🆕 New Concepts vs Project 02

This project makes the jump from regression to classification. Here's what's genuinely new:

| Concept | Project 02 | Project 03 |
|---------|-----------|-----------|
| Output type | Continuous value | Probability → binary class |
| Loss function | MSE | Binary Cross-Entropy |
| Activation | None (linear) | Sigmoid — squashes to (0, 1) |
| Evaluation metric | R², RMSE | Accuracy, Precision, Recall, F1 |
| Decision mechanism | Predict a number | Apply threshold to probability |
| New tool | Bias-variance plot | Confusion matrix, PR tradeoff curve |
| Error types | Residuals | TP, TN, FP, FN — with different costs |
| Optimization goal | Minimise squared error | Minimise classification error |
| Threshold | N/A | Tunable — default 0.5, context-dependent |

---

## 📦 Dataset — Breast Cancer Wisconsin

**Source:** `sklearn.datasets.load_breast_cancer()` — no file download needed.  
**Size:** 569 rows × 30 features

### Class Distribution

| Class | Label | Count | Percentage |
|-------|-------|-------|-----------|
| Benign | 1 | 357 | 62.7% |
| Malignant | 0 | 212 | 37.3% |

> ⚠️ **The dataset is slightly imbalanced.** A dummy classifier that always predicts "benign" would score 62.7% accuracy without learning anything. Always check recall and precision alongside accuracy.

### Feature Groups

All 30 features are computed from digitized images of fine needle aspirates of breast masses. They describe characteristics of cell nuclei present in the image:

| Group | Description | Versions |
|-------|-------------|---------|
| Radius | Mean distance from center to perimeter | mean, SE, worst |
| Texture | Standard deviation of grayscale values | mean, SE, worst |
| Perimeter | Perimeter of the nucleus | mean, SE, worst |
| Area | Area of the nucleus | mean, SE, worst |
| Smoothness | Local variation in radius lengths | mean, SE, worst |
| Compactness | Perimeter² / area − 1 | mean, SE, worst |
| Concavity | Severity of concave portions | mean, SE, worst |
| Concave points | Number of concave portions | mean, SE, worst |
| Symmetry | Symmetry of nucleus | mean, SE, worst |
| Fractal dimension | Coastline approximation − 1 | mean, SE, worst |

### Why No Feature Selection?

Logistic regression handles high dimensions well — it is not a distance-based algorithm, so the curse of dimensionality does not apply the way it does for KNN. 30 features is manageable. However, many features are highly correlated (radius, perimeter, and area all measure tumor size). This introduces **multicollinearity** — predictions remain reliable but individual coefficients become unstable and harder to interpret. We note this in gotchas and move on.

### Why This Dataset?

- Zero friction — `sklearn` built-in, anyone can reproduce with one line
- Clinical stakes make threshold tuning meaningful — missing a malignant tumor is a different cost than a false alarm
- Strong signal — 30 features produce ~98% accuracy, confirming the algorithm works
- Slight class imbalance introduces real-world evaluation nuance

---

## 💡 Intuition First

### Why Linear Regression Fails for Classification

Imagine you have a dataset: tumors with small radius are benign (label 1), tumors with large radius are malignant (label 0). You fit a linear regression line.

Three things go wrong immediately:

**1. Unbounded outputs.** For extreme feature values — a very large tumor radius — linear regression predicts 1.7, 2.3, or -0.4. These are meaningless as probabilities. You can't say "I'm 170% confident this is malignant."

**2. Threshold is arbitrary.** Applying a 0.5 cutoff to a linear model's output has no probabilistic grounding. The model never learned a probability — it learned a number. The decision boundary is based on a fiction.

**3. MSE punishes the wrong things.** MSE minimizes numeric distance from the label. A model that predicts 0.49 for a malignant tumor (label 0) incurs loss `(0.49−0)²=0.24`. A model that predicts 0.01 incurs `(0.01−0)²=0.0001`. MSE massively prefers the 0.01 prediction — correctly. But it rewards this *because the number is small*, not *because we confidently predicted the right class*. The gradient signal is wrong.

### The Key Insight — Wrap It in a Sigmoid

The fix is elegant. Take the exact same linear combination you used in linear regression:

```
z = w₁×radius + w₂×texture + ... + b
```

Then wrap it in the sigmoid function:

```
probability = σ(z) = 1 / (1 + e^(-z))
```

This single change:
- Squashes any real number into (0, 1) → interpretable as probability
- Creates a smooth, differentiable decision boundary at z = 0
- Enables binary cross-entropy loss — which punishes confident wrong predictions correctly

You're still doing gradient descent. You still have weights and a bias. The only architectural addition is the sigmoid and the loss function.

### The Real-World Analogy

Think of a radiologist. They look at a scan and don't say "this tumor scores 7.3." They say "I'm 95% confident this is malignant." Logistic regression does the same thing — it outputs a confidence score, then applies a threshold to convert it into a decision.

The threshold is the key: by default 0.5, but adjustable. For cancer detection, a radiologist might say "if I'm more than 30% confident, flag it for biopsy." The model's threshold works the same way.

> **Interview-ready line:** *"Logistic regression isn't really regression — it's a linear model with a sigmoid activation and a probabilistic loss function. It outputs calibrated probabilities, not just labels."*

---

## 🔢 How It Works — The Math

### 1. The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Three reference points:
- $z = 0$: $\sigma(0) = 0.5$ — the decision boundary
- $z \gg 0$: $\sigma(z) \to 1$ — predict benign
- $z \ll 0$: $\sigma(z) \to 0$ — predict malignant

The derivative has a beautiful closed form:

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

This is not a coincidence — it's why sigmoid and BCE work so cleanly together (the derivative cancels with the BCE gradient, giving a simple update rule).

### 2. Binary Cross-Entropy Loss

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Unpack it for a single sample:

- If $y=1$ (benign): loss = $-\log(\hat{y})$. When $\hat{y} \to 1$: loss → 0. When $\hat{y} \to 0$: loss → ∞.
- If $y=0$ (malignant): loss = $-\log(1-\hat{y})$. When $\hat{y} \to 0$: loss → 0. When $\hat{y} \to 1$: loss → ∞.

**The key property:** BCE penalises confident wrong predictions exponentially. Predicting 0.99 benign for a malignant tumor: $-\log(1-0.99) = -\log(0.01) \approx 4.6$. Predicting 0.51 for the same tumor: $-\log(0.49) \approx 0.71$. The gradient for the confident mistake is 6× larger — it gets corrected much more aggressively.

### 3. Why BCE and Not MSE?

With MSE, the loss surface for logistic regression is **non-convex** — gradient descent can get stuck in local minima. With BCE, the loss surface is **convex** — guaranteed single global minimum.

BCE is derived from **maximum likelihood estimation** of a Bernoulli distribution. Given labels $y \in \{0, 1\}$, the likelihood of observing the data is:

$$P(\text{data} | w) = \prod_{i=1}^{n} \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i}$$

Taking the negative log-likelihood gives exactly the BCE formula. This isn't an arbitrary choice — BCE is the mathematically correct loss for a binary classification problem under the probabilistic model that logistic regression assumes.

### 4. Gradient Derivation

Starting from BCE and applying the chain rule through sigmoid:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

- $\frac{\partial L}{\partial \hat{y}} = \frac{\hat{y} - y}{\hat{y}(1-\hat{y})}$
- $\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$ — the sigmoid derivative cancels with the numerator above
- $\frac{\partial z}{\partial w} = X$

The beautiful result:

$$\frac{\partial L}{\partial w} = \frac{1}{n} X^T (\hat{y} - y) \qquad \frac{\partial L}{\partial b} = \frac{1}{n} \sum (\hat{y} - y)$$

**These are identical in form to linear regression's gradients.** The sigmoid derivative cancels with the BCE derivative — leaving a clean, simple update rule. This is why logistic regression is so stable to train.

### 5. The Math-to-Code Table

| Equation | Code |
|----------|------|
| $z = Xw + b$ | `z = X @ self.weights + self.bias` |
| $\hat{y} = \sigma(z) = \frac{1}{1+e^{-z}}$ | `1 / (1 + np.exp(-z))` |
| clip to prevent $\log(0)$ | `np.clip(y_pred, 1e-15, 1 - 1e-15)` |
| $L = -\frac{1}{n}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | `-np.mean(y * np.log(yp) + (1-y) * np.log(1-yp))` |
| $\frac{\partial L}{\partial w} = \frac{1}{n}X^T(\hat{y}-y)$ | `(1/n_samples) * X.T @ (y_pred - y)` |
| $\frac{\partial L}{\partial b} = \frac{1}{n}\sum(\hat{y}-y)$ | `(1/n_samples) * np.sum(y_pred - y)` |
| $w \leftarrow w - \alpha \nabla w$ | `self.weights -= self.learning_rate * dw` |
| $\hat{y} \geq \text{threshold} \to 1$ | `(probs >= self.threshold).astype(int)` |
| Recall $= \frac{TP}{TP+FN}$ | `recall_score(y, preds, pos_label=0)` |
| Precision $= \frac{TP}{TP+FP}$ | `precision_score(y, preds, pos_label=0)` |

---

## 🔄 Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                               │
│                                                                     │
│  load_breast_cancer()  ──►  train_test_split  ──►  StandardScaler  │
│  569 × 30 features          80% train               fit_transform() │
│  target: 0=malignant        20% test                mean=0, std=1  │
│          1=benign                                   train only  ▼  │
│                                                                     │
│                          LogisticRegression.fit(X_train, y_train)  │
│                         ┌──────────────────────────────────────┐   │
│                         │  For each of n_iterations:           │   │
│                         │  1. z = X @ w + b                    │   │
│                         │  2. ŷ = sigmoid(z)                   │   │
│                         │  3. loss = BCE(y, ŷ)  → loss_history │   │
│                         │  4. dw = (1/n) X.T @ (ŷ - y)        │   │
│                         │  5. db = (1/n) sum(ŷ - y)            │   │
│                         │  6. w -= lr × dw                     │   │
│                         │  7. b -= lr × db                     │   │
│                         │  8. [optional] val_loss → history     │   │
│                         └──────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    PREDICTION PIPELINE                              │
│                                                                     │
│  New Data ──► Scaler.transform() ──► predict_proba()               │
│              (NOT fit_transform!)    sigmoid(X @ w + b)            │
│                                      probabilities ∈ (0, 1)    ▼  │
│                                                                     │
│                           apply threshold (default 0.5)            │
│                           ──► {0, 1} class predictions             │
│                                                                     │
│  Evaluate: Accuracy · Precision · Recall · F1 · Confusion Matrix   │
└─────────────────────────────────────────────────────────────────────┘
```

**The two rules visible in this diagram:**
1. `StandardScaler` uses `fit_transform()` in training, `transform()` only in prediction.
2. The threshold is applied after `predict_proba()` — decoupling probability estimation from the classification decision. This lets you tune the threshold without retraining.

---

## 🛠️ Prerequisites & Structure

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pytest
```

### Project Directory

```
03-logistic-regression/
├── data/                        ← empty — dataset loads from sklearn
├── results/
│   ├── class_distribution.png
│   ├── feature_correlations.png
│   ├── sigmoid_function.png
│   ├── loss_curve.png
│   ├── decision_boundary.png
│   ├── confusion_matrix.png
│   └── precision_recall_tradeoff.png
├── tests/
│   ├── __init__.py
│   └── test_solution.py
├── 01_eda.ipynb                 ← data exploration only
├── 02_implementation.ipynb     ← building and experimenting
├── solution.py                  ← all classes live here
├── train.py                     ← CLI entry point
└── README.md
```

> ⚠️ **No `data/` CSV this time.** `load_breast_cancer()` is a sklearn built-in. The `data/` folder exists for structural consistency with other projects but stays empty.

---

## 🏗️ Build From Scratch

### Step 1 — The `LogisticRegression` Class (`solution.py`)

#### Part A — `__init__`: Store Hyperparameters

Start with the constructor. All it does is store settings and initialize placeholders.

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=3000, threshold=0.5):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.threshold     = threshold
        self.weights       = None   # set during fit()
        self.bias          = None   # set during fit()
```

**Why `n_iterations=3000`?** At 1000 iterations, both train and validation loss are still declining — not converged. At 3000, both curves flatten to near-zero slope. Empirically verified: adding more iterations beyond 3000 reduces loss by <0.01. Engineering judgment: diminishing returns.

**Why `threshold=0.5` as default?** It treats FP and FN as equally costly. For cancer detection you may want to change this — see Step 5 for threshold tuning.

---

#### Part B — `_sigmoid`: The Core Nonlinearity

This is the function that makes logistic regression a classifier, not a regressor. It squashes any real number to (0, 1).

```python
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
```

**Reference points:**
- `z = 0` → 0.5 (decision boundary — model is maximally uncertain)
- `z = +5` → 0.993 (very confident: predict benign)
- `z = -5` → 0.007 (very confident: predict malignant)

**Why one line works for any shape of input:** NumPy broadcasting applies this element-wise to scalars, vectors, and matrices without any loops.

---

#### Part C — `_compute_loss`: Binary Cross-Entropy

Measuring "how wrong" the model is. Extracted as a helper to keep the training loop readable and to centralize the clip operation in one place.

```python
    def _compute_loss(self, y, y_pred):
        # Clip to prevent log(0) = -inf — see Failure Analysis for why this matters
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
```

**What clipping does:** `1e-15` is approximately `10^-15` — effectively zero but safe for `log`. Without this, a single floating-point underflow in sigmoid can produce `log(0) = -inf`, which corrupts the entire loss history.

**Why a helper method and not inline?** Two reasons: (1) the training loop calls it for both train and val loss — no code duplication; (2) any change to the loss formula (e.g., adding a regularization term) only needs updating in one place.

---

#### Part D — `fit`: The Training Loop

This is where learning happens. Every line corresponds directly to the math in the previous section.

```python
    def fit(self, X, y, X_val=None, y_val=None):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # start at zero → loss starts at 0.693
        self.bias    = 0

        self.loss_history     = []
        self.val_loss_history = []

        for _ in range(self.n_iterations):
            # 1. Forward pass — compute predictions from current weights
            y_pred = self._sigmoid(X @ self.weights + self.bias)

            # 2. Compute and store training loss
            self.loss_history.append(self._compute_loss(y, y_pred))

            # 3. Backward pass — compute gradients (math: dL/dw = (1/n) X^T (ŷ - y))
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 4. Update weights in the direction that reduces loss
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            # 5. Optional: measure validation loss AFTER update (weights don't change from val)
            if X_val is not None and y_val is not None:
                y_pred_val = self._sigmoid(X_val @ self.weights + self.bias)
                self.val_loss_history.append(self._compute_loss(y_val, y_pred_val))

        return self
```

**Why `X_val=None` as optional?** Backward compatible — `model.fit(X_train, y_train)` still works. Validation loss tracking is opt-in.

**Why validation loss after the weight update?** You want to measure how the updated weights perform on held-out data — computing it before the update would measure the previous iteration's weights.

---

#### Part E — `predict_proba`, `predict`, `score`

Three methods that use the learned weights for inference.

```python
    def predict_proba(self, X):
        # Returns raw probabilities in (0, 1) — same as forward pass
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        # Converts probabilities to class labels using threshold
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score(self, X, y):
        # Returns proportion of correct predictions (accuracy)
        return np.mean(self.predict(X) == y)
```

**The architecture decision:** `predict_proba` is separate from `predict` by design. Threshold is a **policy decision** — it can be changed without retraining. See Step 5 for how this is used in practice.

---

### Step 2 — Exploratory Data Analysis (`01_eda.ipynb`)

```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = malignant, 1 = benign

print(df.shape)                  # (569, 31)
print(df.isnull().sum().sum())   # 0 — perfectly clean
print(df['target'].value_counts())
# 1    357   ← benign
# 0    212   ← malignant
```

> ⚠️ **Data Reality Check:** This sklearn dataset is perfectly clean — zero nulls, no duplicates, pre-encoded features, balanced enough to train directly. In a real medical dataset, you would spend 60–80% of your time handling missing values caused by failed imaging sensors, standardizing feature extraction across different hospital equipment, removing duplicate patient records, and hunting for target leakage in derived columns. Do not expect zero null values in the wild. This project teaches the algorithm — real projects teach data cleaning.

**Key EDA findings:**

1. **No missing values.** Clean dataset. In production, missing features are common and require explicit handling.

2. **Slight class imbalance (37%/63%).** A dummy classifier scores 62.7% accuracy. Always report recall per class.

3. **`worst concave points` has the strongest negative correlation with target.** Malignant tumors (label 0) exhibit irregular, invasive growth — more concave points. The "worst" version (most extreme value across cell nuclei) is more discriminative than the mean.

4. **High feature-to-feature correlation.** Radius, perimeter, and area measure tumor size in different units — essentially the same signal. Predictions stay reliable, but coefficients are unstable for interpretation.

---

### Step 3 — CLI Training Script (`train.py`)

```python
import argparse
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression as SklearnLR
from solution import LogisticRegression

def main():
    parser = argparse.ArgumentParser(description='Train Logistic Regression')
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--iterations',    type=int,   default=3000)
    parser.add_argument('--test-size',     type=float, default=0.2)
    parser.add_argument('--threshold',     type=float, default=0.5)
    args = parser.parse_args()

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(
        learning_rate=args.learning_rate,
        n_iterations=args.iterations,
        threshold=args.threshold
    )
    model.fit(X_train, y_train)

    sk = SklearnLR(max_iter=3000)
    sk.fit(X_train, y_train)

    print(f"\nTraining Logistic Regression ({args.iterations} iterations)...")
    print(f"Train Accuracy:    {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"Test  Accuracy:    {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"sklearn benchmark: {sk.score(X_test, y_test):.4f}")
    print(f"\nClassification Report (Test):")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=['Malignant', 'Benign']))

if __name__ == '__main__':
    main()
```

**Run from terminal:**
```bash
cd projects/phase-1-classical-ml/03-logistic-regression
python train.py
python train.py --learning-rate 0.1 --iterations 2000 --threshold 0.4
python train.py --threshold 0.85    # maximize malignant recall
```

---

### Step 4 — Unit Tests (`tests/test_solution.py`)

```python
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from solution import LogisticRegression

def test_predictions_are_binary():
    """predict() must return only 0 or 1 — no floats, no other values."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})

def test_predict_proba_range():
    """predict_proba() must return values strictly in (0, 1)."""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert np.all(probs > 0) and np.all(probs < 1)

def test_loss_decreases():
    """Loss at end of training must be lower than at the start."""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    model = LogisticRegression(n_iterations=200, learning_rate=0.1)
    model.fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]

def test_accuracy_beats_random():
    """On linearly separable data, model must exceed 50% accuracy."""
    np.random.seed(42)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)
    assert model.score(X, y) > 0.5

def test_against_sklearn():
    """Custom implementation must be within 2% of sklearn's accuracy."""
    np.random.seed(42)
    X = np.random.randn(300, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    from sklearn.linear_model import LogisticRegression as SklearnLR
    model    = LogisticRegression(n_iterations=1000, learning_rate=0.1)
    sk_model = SklearnLR()
    model.fit(X, y)
    sk_model.fit(X, y)
    assert abs(model.score(X, y) - sk_model.score(X, y)) < 0.02
```

**Run:**
```bash
pytest tests/ -v
# 5 passed in 1.42s ✅
```

---

### Step 5 — Threshold Tuning (`02_implementation.ipynb`)

This is the most important applied step in a cancer detection project. The default threshold of 0.5 is not a clinical decision — it's a mathematical default that assumes FP and FN are equally costly. In cancer screening, they are not.

**The experimentally verified tradeoff on this dataset:**

| Metric | Threshold = 0.5 | Threshold = 0.85 |
|--------|----------------|-----------------|
| Accuracy | **98.25%** | 94.74% |
| Malignant Recall | 97.67% | **100%** |
| Malignant Precision | **97.67%** | 87.76% |
| F1 (Malignant) | **97.67%** | 93.48% |
| Cancers missed (FP*) | 1 | **0** |
| Unnecessary biopsies (FN*) | **1** | 6 |

*\*FP here = actual malignant predicted as benign. FN here = actual benign predicted as malignant. See Confusion Matrix section for the full explanation of sklearn's label convention.*

**Neither threshold is universally better.** The right choice depends entirely on the cost of each error:
- Optimize for **overall accuracy and F1** → use threshold **0.5**
- Optimize for **zero missed cancers** → use threshold **0.85** (accept 6 unnecessary biopsies instead of 1)

Run this sweep yourself to find any threshold:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

probs = model.predict_proba(X_test)

print(f"{'Threshold':<12} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1':<10} {'Missed':<8} {'FalseAlarm'}")
print("-" * 80)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    acc  = accuracy_score(y_test, preds)
    rec  = recall_score(y_test, preds, pos_label=0, zero_division=0)
    prec = precision_score(y_test, preds, pos_label=0, zero_division=0)
    f1   = f1_score(y_test, preds, pos_label=0, zero_division=0)
    print(f"{threshold:<12.2f} {acc:<12.4f} {rec:<12.4f} {prec:<12.4f} {f1:<10.4f} {fp:<8} {fn}")
```

This is equivalent to the degree experiment in Project 02 — a systematic sweep across a hyperparameter to understand the tradeoff, then a deliberate choice based on the use case.

**How to pick your threshold in practice:**

```python
# Find the lowest threshold that achieves a target recall
target_recall = 0.999

for threshold in np.linspace(0.99, 0.01, 1000):
    preds = (probs >= threshold).astype(int)
    recall = recall_score(y_test, preds, pos_label=0, zero_division=0)
    if recall >= target_recall:
        prec = precision_score(y_test, preds, pos_label=0, zero_division=0)
        print(f"Threshold {threshold:.3f} → Recall: {recall:.4f}, Precision: {prec:.4f}")
        break
```

> ⚠️ **Threshold selection is a policy decision, not a modeling one.** It can be changed at any time without retraining. This is why `predict_proba` is architecturally separate from `predict` — the model produces probabilities, and humans decide where the line is drawn.

### `class_distribution.png`

The bar chart shows 212 malignant (37.3%) and 357 benign (62.7%). This plot belongs first in any classification notebook — it sets the baseline. A model predicting "benign" for everyone scores **62.7% accuracy without learning anything.** If your model scores below this, it's doing active harm. This is why recall matters more than accuracy in imbalanced medical settings.

### `feature_correlations.png`

A horizontal bar chart of Pearson correlation between each feature and the target. The most negative bars (correlating with malignancy) are `worst concave points`, `worst perimeter`, `worst area`. The pattern: features measuring **size** and **shape irregularity** predict malignancy most strongly. Biologically: malignant cells grow uncontrollably (size) and invade tissue irregularly (shape irregularity). The "worst" group (most extreme nucleus measurement) is more discriminative than the "mean" group — outlier cell behavior reveals malignancy better than average behavior.

### `sigmoid_function.png`

The S-curve from z=−10 to z=+10. Three things to read from this plot:

1. **The flat tails at z=±8.** Sigmoid output is essentially 0 or 1. Gradient is near-zero here — the model is already very confident and needs little correction.
2. **The steep center near z=0.** This is where gradient updates are largest — the model is uncertain here and most responsive to correction. This region is the "active learning zone."
3. **The decision boundary at σ=0.5.** The vertical dashed line at z=0 maps exactly to σ=0.5. This is the geometric meaning of the decision boundary: the hyperplane where $Xw + b = 0$.

```python
z = np.linspace(-10, 10, 300)
sigmoid = 1 / (1 + np.exp(-z))
plt.plot(z, sigmoid, color='#2ecc71', linewidth=2)
plt.axhline(0.5, color='gray', linestyle='--', label='threshold = 0.5')
plt.fill_between(z, sigmoid, 0.5, where=(sigmoid > 0.5),
                 alpha=0.1, color='#2ecc71', label='Predict Benign')
plt.fill_between(z, sigmoid, 0.5, where=(sigmoid < 0.5),
                 alpha=0.1, color='#e74c3c', label='Predict Malignant')
```

### `loss_curve.png`

Train and validation loss plotted together across 3000 iterations.

```
Loss
0.693 ─ ●                  ← always starts here (see math section)
        │\
        │  \               ← steep drop: correcting large errors
0.300 ─ │    \
        │      \──
0.100 ─ │          ──────── ← convergence: both curves plateau
        └──────────────────── iterations
        0    300   1000   3000
```

**What to look for:**
- **Two curves tracking closely** → no overfitting. If validation loss rose while training loss fell, that's the signal to add regularization or stop early.
- **Elbow at ~300 iterations** → fast learning phase ends. Remaining iterations are fine-tuning. Do NOT stop here — loss at the elbow is ~0.25, final loss is ~0.09. That's real accuracy gain.
- **Still declining slightly at 3000** → diminishing returns, not full convergence. Adding 3000 more iterations reduces loss by <0.01. The engineering judgment: stop here.

> ⚠️ **A training loss curve alone cannot tell you when to stop.** You need validation loss alongside it. The training curve always decreases — it tells you gradient descent is working, not that generalization is improving.

### `decision_boundary.png`

Plotted on the first two features (mean radius, mean texture) for 2D visualization. The decision boundary is a **straight line** — logistic regression's fundamental geometric signature. It is always a hyperplane regardless of the number of features.

What the 2D plot also reveals: with only 2 of 30 features, there is significant class overlap. The full 30-feature model achieves 98.2% accuracy because the other 28 features provide additional separating signal invisible in this 2D projection. This is a useful demonstration of the curse of low dimensionality in visualization — 2D projections lie.

### `confusion_matrix.png`

**Understanding sklearn's label convention first.** When you call `confusion_matrix(y_test, preds).ravel()` → `tn, fp, fn, tp`, sklearn treats class 1 (benign) as the positive class by default. So:

| Variable | sklearn meaning | In plain English |
|----------|----------------|-----------------|
| `tn` | actual 0, predicted 0 | Malignant correctly caught ✅ |
| `fp` | actual 0, predicted 1 | **Malignant missed** — sent home as benign ❌ |
| `fn` | actual 1, predicted 0 | Benign falsely flagged as malignant ⚠️ |
| `tp` | actual 1, predicted 1 | Benign correctly cleared ✅ |

**At threshold = 0.5 (experimentally verified):**

```
                  Predicted
                  Malignant   Benign
Actual Malignant |    42    |    1   |  ← 1 cancer sent home (FP in sklearn = missed malignant)
       Benign    |     1    |   70   |  ← 1 unnecessary biopsy (FN in sklearn = false alarm)
```

- **42 TN** → malignant correctly caught, patients get treatment ✅
- **1 FP** → malignant predicted as benign — patient sent home untreated ❌ **dangerous**
- **1 FN** → benign predicted as malignant — unnecessary biopsy ⚠️ costly but safe
- **70 TP** → benign correctly cleared ✅

One missed malignant tumor in 43. At threshold = 0.85, this becomes zero missed cancers — at the cost of 6 unnecessary biopsies. See Step 5 for the full tradeoff.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Malignant', 'Benign'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix — Threshold = 0.5')
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"Malignant correctly caught (TN): {tn}")
print(f"Malignant missed — sent home (FP): {fp}  ← DANGEROUS")
print(f"Benign falsely flagged (FN):       {fn}  ← unnecessary biopsy")
print(f"Benign correctly cleared (TP):     {tp}")
```

### `precision_recall_tradeoff.png`

Plotted for the malignant class (`pos_label=0`) across thresholds 0.1–0.9. This is the plot that makes threshold selection a data-driven decision rather than a guess.

**What the plot actually shows (verified on this model):**

```
Score
1.00 ─ ────────────────────────────── Recall (malignant) at threshold=0.85 → 1.0
       │
0.977 ─ ─── Recall at threshold=0.5
       │
0.878 ─ ─── Precision at threshold=0.85
       │
       │     Precision at 0.5 → 0.977
       └─────────────────────────────────── threshold
       0.1       0.5            0.85    0.9
```

**Reading the plot:**
- At threshold = 0.5: recall = 0.9767, precision = 0.9767, F1 = 0.9767 — balanced and high
- At threshold = 0.85: recall = 1.0000, precision = 0.8776, F1 = 0.9348 — perfect recall, more false alarms

**The tradeoff is real and has been verified:**

| Threshold | Cancers missed | Unnecessary biopsies | Accuracy |
|-----------|---------------|---------------------|---------|
| 0.5 | 1 | 1 | **98.25%** |
| 0.85 | **0** | 6 | 94.74% |

No universal winner. Choose based on the clinical requirement.

```python
from sklearn.metrics import precision_score, recall_score

thresholds = np.linspace(0.1, 0.9, 100)
probs      = model.predict_proba(X_test)

precisions = [precision_score(y_test, (probs >= t).astype(int),
              pos_label=0, zero_division=0) for t in thresholds]
recalls    = [recall_score(y_test, (probs >= t).astype(int),
              pos_label=0, zero_division=0) for t in thresholds]

plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions, label='Precision (Malignant)', color='#3498db', linewidth=2)
plt.plot(thresholds, recalls,    label='Recall (Malignant)',    color='#e74c3c', linewidth=2)
plt.axvline(0.5,  color='gray', linestyle='--', linewidth=1, label='Default threshold (0.5)')
plt.axvline(0.85, color='orange', linestyle=':', linewidth=1.5, label='High-recall threshold (0.85)')
plt.title('Precision-Recall Tradeoff vs Threshold (Malignant class)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.tight_layout()
plt.savefig('results/precision_recall_tradeoff.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 🔬 BCE Loss — The Math Behind Why It Works

### Why MSE Fails Geometrically

Consider a single malignant sample (y=0). MSE loss at $\hat{y} = 0.99$ (confidently wrong):

$$L_{MSE} = (0.99)^2 = 0.98, \quad \nabla = 2 \times 0.99 = 1.98$$

But the effective gradient flowing back through sigmoid: $\sigma'(z) = 0.99 \times 0.01 = 0.0099$. Total: $1.98 \times 0.0099 \approx 0.02$. Almost nothing. The model barely corrects its worst mistake.

BCE fixes this because the $\hat{y}(1-\hat{y})$ in BCE's derivative cancels exactly with $\sigma'(z)$:

$$\frac{\partial L_{BCE}}{\partial z} = \hat{y} - y$$

No vanishing gradient. The more wrong the prediction, the larger the update. This is why BCE + sigmoid is the natural pairing.

### Why Loss Always Starts at 0.693

With zero-weight initialization: $z = 0$ for every sample, sigmoid outputs 0.5 everywhere.

$$L = -[y \log(0.5) + (1-y)\log(0.5)] = -\log(0.5) = \log(2) \approx 0.693$$

This is **mathematically guaranteed** regardless of dataset size, number of features, or class balance. A quick implementation sanity check: assert `loss_history[0] ≈ 0.693`. Any other value means your weight initialization or sigmoid has a bug.

### Convexity — Why Gradient Descent Always Converges

The BCE loss surface is **strictly convex** — one global minimum, no local minima, no saddle points. Proven by the Hessian:

$$H = \frac{1}{n} X^T S X \quad \text{where } S = \text{diag}(\hat{y}_i(1-\hat{y}_i))$$

Since $\hat{y}_i \in (0,1)$, all diagonal entries of $S$ are positive. $H$ is positive semi-definite — gradient descent is guaranteed to find the optimal weights. This convexity guarantee is what logistic regression offers that neural networks do not.

---

## 💥 Failure Analysis — What Broke and What We Learned

### Failure 1: Loss Explodes to `nan` — The `log(0)` Problem

**What happened:**

```python
loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
# RuntimeWarning: divide by zero encountered in log
# loss_history: [0.693, 0.612, 0.544, ..., nan, nan, nan]
```

After several hundred iterations, loss became `nan` and never recovered. All subsequent predictions were garbage.

**Why it happened:**

When $z$ is very large negative (e.g., $z = -800$), `np.exp(800)` overflows to `inf`, making sigmoid output exactly `0.0`. Then `np.log(0.0) = -inf`. The `nan` propagates through all subsequent arithmetic.

**The fix:**

```python
y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
```

`log(1e-15) ≈ -34.5` — a very large penalty, but finite. The model gets strongly corrected rather than crashing.

**The lesson:** Always clip probabilities before any log. Make this a reflex. Centralize the clip in `_compute_loss()` — not sprinkled across the training loop.

---

### Failure 2: Name Collision — sklearn Overwrites Custom Class

**What happened:**

```python
from solution import LogisticRegression       # custom class

def test_against_sklearn():
    from sklearn.linear_model import LogisticRegression  # ← OVERWRITES
    model = LogisticRegression(n_iterations=1000)
    # TypeError: __init__() got unexpected keyword argument 'n_iterations'
```

**Why it happened:**

The second import inside the function body overwrote the module-level name `LogisticRegression`. All subsequent code used sklearn's class instead of the custom one. No syntax error — just a wrong class silently substituted.

**The fix:**

```python
from sklearn.linear_model import LogisticRegression as SklearnLR
sk_model = SklearnLR()
```

**The lesson:** Never import two classes with the same name without aliasing one. The bug is invisible until runtime.

---

### Failure 3: `predict` Nested Inside `predict_proba` — Indentation Bug

**What happened:**

```python
def predict_proba(self, X):
    ŷ = self._sigmoid(X @ self.weights + self.bias)
    return ŷ
        def predict(self, X):          # ← nested — NOT a class method
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
# AttributeError: 'LogisticRegression' object has no attribute 'predict'
```

**Why it happened:** Python silently accepts nested function definitions. `predict` was defined inside `predict_proba`'s scope and inaccessible as a class method.

**The fix:** Align `def predict` at the same indentation level as `def predict_proba`.

**The lesson:** Verify method accessibility after writing a new class: `[m for m in dir(model) if not m.startswith('_')]`. The bug only surfaces at call time.

---

### Failure 4: `fit_transform` on Test Data — Silent Leakage

**What happened:**

```python
X_test_scaled = scaler.fit_transform(X_test)  # ← BUG
```

Test accuracy looked identical to the correct version. The bug was invisible.

**Why it matters:** The Breast Cancer dataset is large enough that refitting on test gives nearly identical mean/std. But on a biased test set (different hospital, different demographics), refitting on test data would produce materially wrong predictions — and you'd never know because your reported metrics still look fine.

**The fix:** `scaler.fit_transform(X_train)`, then `scaler.transform(X_test)`. Rule is absolute: **fit on train, transform both.**

---

### Failure 5: `sum()` Instead of `np.sum()`

**What happened:**

```python
db = (1/n_samples) * sum(ŷ - y)  # Python built-in — works but wrong
```

No error. Correct result. But Python's built-in `sum` iterates through the array in a Python loop — much slower than `np.sum`'s vectorized C operation. Over 3000 iterations this overhead is measurable.

**The fix:** `np.sum(ŷ - y)`. Always use NumPy functions in NumPy codebases.

---

## 🏭 Production Thinking

### Saving and Loading the Model

```python
import joblib

joblib.dump({
    'model':         model,
    'scaler':        scaler,
    'features':      list(data.feature_names),
    'n_features':    30,
    'threshold':     0.5,
    'trained_at':    '2026-05-01',
    'test_accuracy': 0.9825,
    'params': {'learning_rate': 0.01, 'n_iterations': 3000}
}, 'results/logistic_regression.joblib')

# Load at prediction time
saved     = joblib.load('results/logistic_regression.joblib')
model     = saved['model']
scaler    = saved['scaler']
threshold = saved['threshold']

def predict_patient(raw_features_df):
    X_scaled = scaler.transform(raw_features_df[saved['features']].values)
    probs    = model.predict_proba(X_scaled)
    return (probs >= threshold).astype(int), probs
```

> ⚠️ **Always save metadata alongside the model.** Which features it expects, in what order, which threshold was used, when it was trained. A model file without metadata is a black box that silently fails when the data pipeline changes.

---

### What Happens When Things Go Wrong in Production

#### Problem 1: Input Distribution Shift

**Scenario:** Trained on Hospital A's imaging equipment. Hospital B joins the network — their equipment calibrates `mean radius` 15% differently.

**What happens:** Features are shifted relative to the scaler's learned mean. After normalization they land in a different part of feature space. Accuracy drops 98% → 85% with no error raised.

**Defensive detection:**

```python
def check_distribution(X_new, saved_stats, threshold_stds=2.0):
    for i, feature in enumerate(saved_stats['feature_names']):
        drift = abs(X_new[:, i].mean() - saved_stats['train_means'][i])
        drift /= saved_stats['train_stds'][i]
        if drift > threshold_stds:
            print(f"⚠️  Drift in '{feature}': {drift:.2f} standard deviations")
```

**The fix:** Retrain periodically. Flag out-of-distribution inputs for human review.

---

#### Problem 2: Missing Features at Inference

**Scenario:** Imaging artifact corrupts concavity measurements. Three features arrive as `NaN`.

**What happens:** `NaN` propagates through scaler → matrix multiply → sigmoid → `NaN >= 0.5` evaluates to `False` → **predicts class 0 (malignant) for everyone.** Silent wrong predictions, no error.

**Defensive prediction:**

```python
def predict_safe(model, scaler, X_raw, feature_names):
    X_df = pd.DataFrame(X_raw, columns=feature_names)
    missing = X_df.columns[X_df.isnull().any()].tolist()
    if missing:
        # In cancer detection: always reject, never impute silently
        raise ValueError(f"Cannot score patient with missing features: {missing}")
    X_scaled = scaler.transform(X_df.values)
    probs = model.predict_proba(X_scaled)
    return (probs >= model.threshold).astype(int), probs
```

---

#### Problem 3: Threshold Policy Change

**Scenario:** Your hospital's policy changes — legal requires catching 100% of malignant cases.

**Verified result — no retraining required.**

On this model, threshold = 0.85 achieves:
- Malignant recall: **1.0000** — zero missed cancers
- Malignant precision: 0.8776 — 6 unnecessary biopsies out of 49 flagged
- Overall accuracy: 94.74% (down from 98.25% at threshold 0.5)

This tradeoff was verified empirically — not inferred from a plot. See Step 5 for the full comparison table.

```python
probs = model.predict_proba(X_val)
target_recall = 1.0

for threshold in np.linspace(0.99, 0.01, 1000):
    preds  = (probs >= threshold).astype(int)
    recall = recall_score(y_val, preds, pos_label=0, zero_division=0)
    if recall >= target_recall:
        prec = precision_score(y_val, preds, pos_label=0, zero_division=0)
        print(f"Threshold {threshold:.3f} → Recall: {recall:.4f}, Precision: {prec:.4f}")
        model.threshold = threshold
        break
```

Threshold is a **policy decision**, not a model parameter. Change it without touching the weights. This is why separating `predict_proba` from `predict` matters architecturally.

---

### Production Checklist

- [ ] Model saved with metadata: features, order, threshold, training date, metrics
- [ ] Input validation: check feature names, count, types before scoring
- [ ] Missing value strategy defined and documented
- [ ] Distribution drift monitoring: feature means and stds tracked at inference
- [ ] Threshold documented and linked to clinical requirement
- [ ] Threshold review process: who approves changes?
- [ ] Prediction logging for audit trail (required for medical devices)
- [ ] Retraining trigger defined: time-based or drift-based?
- [ ] Fallback defined: what happens when model errors?

---

## 🚫 When NOT to Use Logistic Regression

| Scenario | Why LR Fails | Use Instead |
|----------|-------------|-------------|
| Non-linear boundary (donut, XOR, spiral) | Always draws a hyperplane — cannot curve | SVM (RBF), Random Forest, GBM |
| Many irrelevant features (500+, most are noise) | Includes all features, no auto-selection | Lasso (L1), Random Forest |
| Extreme class imbalance (1:100+) | Majority class gradient drowns minority | XGBoost + `scale_pos_weight`, SMOTE |
| Multi-class (>2 classes), no extension | Binary output only | Softmax regression, tree models |
| Need max accuracy, interpretability unimportant | Trees almost always outperform | GBM, XGBoost, Random Forest |
| Sparse data (NLP, one-hot hundreds of levels) | Dense polynomial space, memory issues | Sparse LR with L1, Naive Bayes |

### When to Still Use It

- Linear boundary is known or suspected ✅
- Need calibrated probabilities (well-calibrated by default) ✅
- Need interpretable coefficients (business stakeholder) ✅
- Small dataset (<1k rows) where trees overfit ✅
- Fast training and inference required ✅

---

## ⚠️ Common Mistakes & Gotchas

1. **Accuracy is not enough for imbalanced data.** Predicting "benign" always scores 62.7% here. Always evaluate precision and recall per class.

2. **The default threshold of 0.5 is almost never optimal.** It assumes equal cost for FP and FN. In cancer detection, they're not equal. Plot a precision-recall curve and choose based on the actual cost of each error type.

3. **Fitting the scaler on test data.** `scaler.fit_transform(X_test)` leaks test statistics. The bug is invisible when test distribution matches train — which it usually does in benchmarks. It surfaces in production. Rule: `fit_transform` on train, `transform` on test, always.

4. **Not clipping probabilities before log.** `log(0) = -inf`. Even if sigmoid theoretically never outputs exactly 0, floating point underflow makes it happen. Always `np.clip(y_pred, 1e-15, 1 - 1e-15)` before any log operation.

5. **Stopping at the loss curve elbow.** The elbow is where fast learning ends, not where learning ends. At the elbow (~300 iterations), loss is ~0.25. At convergence (~3000), loss is ~0.09. That gap is real accuracy gain.

6. **Using `sum()` instead of `np.sum()`.** Python's built-in works but is a Python loop over a NumPy array. Over 3000 training iterations this accumulates. Use `np.sum` everywhere in numerical code.

7. **Importing two classes with the same name.** `from sklearn.linear_model import LogisticRegression` overwrites your custom class. Alias: `from sklearn.linear_model import LogisticRegression as SklearnLR`.

8. **Confusing threshold direction with malignant recall.** In this dataset 1=benign. `predict` outputs 1 when `prob >= threshold`. Raising the threshold makes it harder to predict benign → more cases flagged malignant → **raising** threshold increases malignant recall. Direction depends on which class is encoded as 1.

---

## 🎯 10 Interview Questions

<details>
<summary><strong>Q1: Why can't you use linear regression for binary classification?</strong></summary>

**Answer:** Three reasons. First, linear regression produces unbounded outputs — predictions can be −2.4 or 1.7, which are meaningless as probabilities. Second, applying a 0.5 threshold to these outputs has no probabilistic grounding — the model never learned a probability, just a number. Third, MSE minimizes numeric distance from labels, not classification accuracy. The gradient signal is wrong: at extreme probabilities, the MSE gradient is nearly zero when composed with sigmoid — the model barely learns from its most confident mistakes. BCE fixes all three: it's derived from maximum likelihood, its gradient doesn't vanish at extremes, and the output is a calibrated probability.

</details>

<details>
<summary><strong>Q2: Explain the sigmoid function and what it represents geometrically.</strong></summary>

**Answer:** The sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$ squashes any real number to (0, 1). Reference points: z=0 → 0.5 (decision boundary), large positive z → 1, large negative z → 0. Geometrically, z is the signed distance from the decision boundary hyperplane, scaled by the weights. Points far from the boundary get probabilities close to 0 or 1. Points near the boundary get probabilities close to 0.5. The sigmoid converts "distance from boundary" into "confidence of prediction." Its derivative is $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which cancels perfectly with the BCE gradient to give a clean update rule.

</details>

<details>
<summary><strong>Q3: Why is binary cross-entropy the right loss for logistic regression?</strong></summary>

**Answer:** Two reasons — practical and theoretical. Practical: BCE penalises confident wrong predictions exponentially. Predicting 0.99 benign for a malignant tumor costs `−log(0.01) ≈ 4.6`. Predicting 0.51 costs `−log(0.49) ≈ 0.71`. The gradient for the confident mistake is 6× larger. MSE's gradient nearly vanishes at extreme probabilities due to sigmoid's flat tails — the model barely corrects its worst mistakes. Theoretical: BCE is the negative log-likelihood of a Bernoulli distribution — the mathematically correct loss for binary classification. It produces a convex loss surface, guaranteeing a global minimum for gradient descent.

</details>

<details>
<summary><strong>Q4: Why does the loss always start at exactly 0.693?</strong></summary>

**Answer:** With zero-weight initialization, z=0 for every sample, so sigmoid outputs 0.5 everywhere. BCE loss for any label with $\hat{y}=0.5$: $-[y\log(0.5)+(1-y)\log(0.5)] = -\log(0.5) = \log(2) \approx 0.693$. This is mathematically guaranteed regardless of dataset size, number of features, or class balance. It's a quick implementation sanity check: assert `loss_history[0] ≈ 0.693`. Any other starting value indicates a bug in weight initialization or sigmoid.

</details>

<details>
<summary><strong>Q5: What is the decision boundary and what shape does it take?</strong></summary>

**Answer:** The decision boundary is the set of points where the model predicts exactly 0.5 probability — where $Xw + b = 0$. For logistic regression this is always a **hyperplane** — a straight line in 2D, a flat plane in 3D. Logistic regression is a linear classifier. It cannot represent curved boundaries without feature engineering. The perpendicular distance from any point to the hyperplane determines confidence: farther away means higher probability. This geometric interpretation explains why logistic regression fails on non-linearly separable data — no single hyperplane can separate a donut pattern, for example.

</details>

<details>
<summary><strong>Q6: Explain precision, recall, and the F1 score. When does each matter?</strong></summary>

**Answer:** Precision = TP/(TP+FP): of all positive predictions, how many were actually positive? Recall = TP/(TP+FN): of all actual positives, how many did we catch? F1 = 2×(P×R)/(P+R): harmonic mean — penalizes extremes more than arithmetic mean. Precision matters when FP is costly: spam detection (don't delete real emails). Recall matters when FN is costly: cancer screening (don't miss tumors). F1 matters when both must be balanced. The harmonic mean is appropriate because a model with precision=1, recall=0.01 is useless — the arithmetic mean flatters it at 0.505, the harmonic mean correctly scores it at 0.02.

</details>

<details>
<summary><strong>Q7: What is the confusion matrix and how do you decide which error type matters more?</strong></summary>

**Answer:** A 2×2 table mapping predicted vs actual classes. TN=correctly predicted negatives, TP=correctly predicted positives, FP=wrong positive predictions (Type I error), FN=missed positives (Type II error). The cost of FP vs FN is entirely context-dependent and requires domain knowledge: cancer screening — FN means untreated cancer (life-threatening), FP means unnecessary biopsy (costly but safe) → optimize for recall. Spam detection — FN means spam in inbox (minor), FP means deleted real email (severe) → optimize for precision. The confusion matrix shows you the counts. Deciding which error is worse is a business and ethical decision the model alone cannot make.

</details>

<details>
<summary><strong>Q8: How does the decision threshold work and how do you choose it in production?</strong></summary>

**Answer:** `predict_proba` returns a probability in (0, 1). Threshold converts this to a label: prob ≥ threshold → class 1, else → class 0. Default 0.5 assumes equal cost for FP and FN — almost never true. To choose: plot precision and recall for the class of interest across all thresholds from 0 to 1. Find the threshold that satisfies your constraint. For cancer detection requiring ≥99% malignant recall: find the minimum threshold achieving this recall on the validation set, accept the precision at that point. Importantly — threshold selection is a policy decision, not a modeling one. It can be changed without retraining, which is why separating `predict_proba` from `predict` is architecturally important.

</details>

<details>
<summary><strong>Q9: What is class imbalance and how does it affect logistic regression?</strong></summary>

**Answer:** Class imbalance occurs when one class significantly outnumbers the other. Effects: (1) A dummy classifier predicting majority achieves high accuracy — accuracy becomes misleading. (2) BCE loss receives more gradient signal from the majority class, causing the model to bias toward predicting majority. (3) Minority class predictions become unreliable. Fixes for severe imbalance (1:100+): `class_weight='balanced'` (scales minority gradient contributions proportionally), SMOTE oversampling (creates synthetic minority samples), threshold adjustment (lower threshold to flag more positives), or using precision-recall AUC instead of ROC AUC (ROC AUC is misleadingly optimistic under severe imbalance — it doesn't reflect the low base rate of positives).

</details>

<details>
<summary><strong>Q10: Your train and validation loss curves are very close together. What does that tell you, and what would overfitting and underfitting look like?</strong></summary>

**Answer:** Two curves tracking closely means the model generalizes well — it's learning the real signal, not noise, so it performs similarly on held-out data. For underfitting: both train AND validation loss would be high and flat — near-random accuracy, model hasn't learned the pattern. Fix: more iterations, higher learning rate, or more expressive features. For overfitting: train loss low and still decreasing while validation loss plateaus or rises. Model memorized training noise. Fix: regularization (L1/L2), more data, or fewer parameters. In this project with 455 training samples and only 31 parameters (30 weights + bias), overfitting is geometrically impossible — the model is massively underdetermined. The close train/val curves confirm this theoretical expectation empirically.

</details>

---

## 🏋️ Exercises & Challenges

**🟢 Beginner**
- [ ] Run `train.py --threshold 0.3` and `train.py --threshold 0.8`. Print the classification report for both. How does malignant recall change in each direction? Explain the mechanism.
- [ ] Remove `StandardScaler` from the pipeline. What happens to convergence speed and final accuracy? Why?
- [ ] Change `n_iterations` to 500, 1000, 2000, 3000. Plot all four loss curves on the same figure. At what iteration count do they visually converge?
- [ ] Initialize weights with `np.random.randn(n_features) * 0.01` instead of zeros. Does the starting loss change? Why or why not?

**🟡 Intermediate**
- [ ] Add L2 regularization to the gradient: $\nabla w = \frac{1}{n}X^T(\hat{y}-y) + \frac{\lambda}{n}w$. Add `lambda_` as a hyperparameter. Test values `[0, 0.01, 0.1, 1.0]`. Does accuracy improve?
- [ ] Implement a `cross_validate` method: split training data into k folds, train on k−1, validate on 1, report mean ± std accuracy across folds.
- [ ] Plot the ROC curve (TPR vs FPR across all thresholds) using `predict_proba` output. Compute AUC. Compare with the precision-recall curve — when is each more informative?
- [ ] Add a `batch_size` parameter to `fit` for mini-batch gradient descent. Compare convergence speed vs full-batch on this dataset.

**🔴 Advanced**
- [ ] Implement early stopping: monitor validation loss and stop when it hasn't improved for `patience` consecutive iterations. Add `patience=50` as a hyperparameter.
- [ ] Extend `LogisticRegression` to multi-class using One-vs-Rest: train k binary classifiers, predict the class with highest probability. Test on Iris (3 classes) and compare to sklearn.
- [ ] Implement Platt scaling to calibrate probabilities: fit a second logistic regression on the raw model outputs. Plot reliability diagrams before and after calibration.
- [ ] Implement SGD (batch_size=1). Compare convergence trajectories with full-batch. Visualize the noisy vs smooth loss curves side by side.

---

## 🔗 What's Next

- **[04 — K-Nearest Neighbours →](../04-knn/)** — A distance-based, non-parametric classifier. No training — just memorize the data. Compare how the decision boundary looks vs logistic regression's hyperplane, and discover when distance metrics fail.
- **[05 — Naive Bayes →](../05-naive-bayes/)** — A probabilistic classifier based on Bayes' theorem. Extremely fast, great for text. Compare probabilistic frameworks.

---

## 📚 Further Reading

1. [Stanford CS229 — Logistic Regression Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Full MLE derivation + gradient derivation from first principles
2. [Deep Learning Book — Chapter 6](https://www.deeplearningbook.org/) — Why cross-entropy is the natural loss for classifiers with sigmoid/softmax outputs
3. [Precision-Recall vs ROC Curves — Davis & Goadrich (2006)](https://dl.acm.org/doi/10.1145/1143844.1143874) — When PR curves are more informative than ROC under class imbalance
4. [sklearn LogisticRegression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — Production implementation: solvers, regularization options, multi-class
5. [Breast Cancer Wisconsin Dataset — UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) — Original paper and feature descriptions

---

<p align="center">
  <strong>ai-from-scratch</strong> · Project 03 · Logistic Regression<br>
  Built with curiosity · Shared with the community
</p>