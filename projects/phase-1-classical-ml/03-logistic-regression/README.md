# Project 03 — Logistic Regression
### Binary Classification: Breast Cancer Detection

---

## 🎯 What You'll Learn

- Why linear regression fails for classification tasks
- How the sigmoid function squashes unbounded outputs into probabilities
- Binary cross-entropy loss — and why MSE is the wrong choice for classification
- Gradient descent for logistic regression — derived and implemented from scratch
- Decision boundaries — what they mean geometrically
- Confusion matrix — TP, TN, FP, FN and what each costs in the real world
- Precision vs recall tradeoff — and why the right metric depends on the problem
- Threshold tuning — moving beyond the default 0.5
- Class imbalance — how it distorts accuracy and what to do about it

---

## 🆕 New Concepts vs Previous Projects

| Concept | Project 01–02 | Project 03 |
|--------|--------------|------------|
| Output type | Continuous value | Probability → class |
| Loss function | MSE | Binary Cross-Entropy |
| Activation | None (linear) | Sigmoid |
| Evaluation | R², RMSE | Accuracy, Precision, Recall, F1 |
| Goal | Minimize residuals | Maximize correct classifications |
| Decision | Predict a number | Predict a class |
| New tool | — | Confusion matrix, threshold tuning |

---

## 📊 Dataset

**Breast Cancer Wisconsin** — `sklearn.datasets.load_breast_cancer()`

| Property | Value |
|----------|-------|
| Rows | 569 |
| Features | 30 |
| Target | 0 = Malignant, 1 = Benign |
| Class distribution | 212 malignant (37.3%), 357 benign (62.7%) |
| Missing values | None |

### Feature Groups
All 30 features are computed from digitized images of fine needle aspirates of breast masses. They describe characteristics of cell nuclei:

| Group | Features |
|-------|---------|
| Radius | mean, standard error, worst |
| Texture | mean, standard error, worst |
| Perimeter | mean, standard error, worst |
| Area | mean, standard error, worst |
| Smoothness | mean, standard error, worst |
| Compactness | mean, standard error, worst |
| Concavity | mean, standard error, worst |
| Concave points | mean, standard error, worst |
| Symmetry | mean, standard error, worst |
| Fractal dimension | mean, standard error, worst |

### Why No Feature Selection?
Logistic regression handles high dimensions well — it is not a distance-based algorithm so the curse of dimensionality does not apply. 30 features is fine. However, many features are highly correlated (e.g., radius, perimeter, and area all measure tumor size). This introduces multicollinearity — predictions remain reliable but individual coefficients become unstable and harder to interpret.

---

## 💡 Intuition First

Imagine you are a doctor looking at a scan. You do not output a number like "this tumor is 7.3 dangerous." You output a decision: malignant or benign.

Logistic regression learns to do the same thing — but instead of jumping straight to a class, it first learns a **probability**. "I am 94% confident this is malignant." Then it applies a threshold (default: 0.5) to convert that probability into a class.

The key ingredient is the **sigmoid function** — a mathematical S-curve that squashes any real number into (0, 1). Feed it the same linear combination of features you used in linear regression, and out comes a probability instead of an unbounded number.

The model learns by asking: "How wrong were my probability predictions?" That wrongness is measured by **binary cross-entropy loss** — a function that punishes confident wrong predictions exponentially. If you say 99% benign for a malignant tumor, you get a massive penalty. If you say 51% benign, the penalty is small.

---

## 📐 Math Section

### 1. Forward Pass

Linear combination (same as linear regression):
$$z = \mathbf{X}\mathbf{w} + b$$

Sigmoid activation:
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Binary Cross-Entropy Loss

$$L = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 3. Gradients (derived from BCE loss)

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{n} \mathbf{X}^T (\hat{y} - y)$$

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum (\hat{y} - y)$$

### 4. Weight Update

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial L}{\partial \mathbf{w}}$$

$$b \leftarrow b - \alpha \frac{\partial L}{\partial b}$$

### Math-to-Code Table

| Equation | Code |
|----------|------|
| $z = \mathbf{X}\mathbf{w} + b$ | `z = X @ self.weights + self.bias` |
| $\sigma(z) = \frac{1}{1+e^{-z}}$ | `1 / (1 + np.exp(-z))` |
| $L = -\frac{1}{n}\sum[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | `-np.mean(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))` |
| $\frac{\partial L}{\partial w} = \frac{1}{n}X^T(\hat{y}-y)$ | `(1/n_samples) * X.T @ (y_pred - y)` |
| $\frac{\partial L}{\partial b} = \frac{1}{n}\sum(\hat{y}-y)$ | `(1/n_samples) * np.sum(y_pred - y)` |
| $w \leftarrow w - \alpha \nabla w$ | `self.weights -= self.learning_rate * dw` |
| $\hat{y} \geq \text{threshold} \rightarrow 1$ | `(probs >= self.threshold).astype(int)` |

---

## 🔄 Pipeline Diagram

```
Raw Data (569 × 30)
        │
        ▼
train_test_split (80/20, random_state=42)
        │
        ▼
StandardScaler (fit on train, transform both)
        │
        ▼
LogisticRegression.fit(X_train, y_train)
        │
    ┌───┴────────────────────┐
    │   For each iteration:  │
    │  1. z = Xw + b         │
    │  2. ŷ = sigmoid(z)     │
    │  3. loss = BCE(y, ŷ)   │
    │  4. dw, db = gradients │
    │  5. update w, b        │
    └───────────────────────┘
        │
        ▼
predict_proba(X_test) → probabilities ∈ (0, 1)
        │
        ▼
predict(X_test) → apply threshold → {0, 1}
        │
        ▼
Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
```

---

## 🔨 Build From Scratch

### Step 1 — Sigmoid
```python
def _sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
```

### Step 2 — Loss Helper
```python
def _compute_loss(self, y, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # prevent log(0)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
```

### Step 3 — Training Loop
```python
def fit(self, X, y, X_val=None, y_val=None):
    n_samples, n_features = X.shape
    self.weights = np.zeros(n_features)
    self.bias = 0
    self.loss_history = []
    self.val_loss_history = []

    for _ in range(self.n_iterations):
        y_pred = self._sigmoid(X @ self.weights + self.bias)
        self.loss_history.append(self._compute_loss(y, y_pred))

        dw = (1/n_samples) * X.T @ (y_pred - y)
        db = (1/n_samples) * np.sum(y_pred - y)

        self.weights -= self.learning_rate * dw
        self.bias    -= self.learning_rate * db

        if X_val is not None and y_val is not None:
            y_pred_val = self._sigmoid(X_val @ self.weights + self.bias)
            self.val_loss_history.append(self._compute_loss(y_val, y_pred_val))

    return self
```

### Step 4 — Predict
```python
def predict_proba(self, X):
    return self._sigmoid(X @ self.weights + self.bias)

def predict(self, X):
    return (self.predict_proba(X) >= self.threshold).astype(int)
```

---

## 📊 Visualization Deep Dive

### `class_distribution.png`
Malignant: 212 (37.3%), Benign: 357 (62.7%). The dataset is slightly imbalanced. Not severe enough to require SMOTE or class weights, but enough to make accuracy a misleading metric in isolation. A dummy classifier that always predicts benign would score 62.7% — always check recall alongside accuracy.

### `feature_correlations.png`
Features like `worst concave points`, `worst perimeter`, and `worst area` show the strongest negative correlation with the target (remember: 0 = malignant, 1 = benign). These capture the irregular, invasive growth patterns characteristic of malignant tumors. Strongly correlated features span multiple measurement groups — size, shape, and texture all contribute signal.

### `sigmoid_function.png`
The S-curve that makes logistic regression work. Any value of z maps to (0, 1). z = 0 → 0.5 (decision boundary). Large positive z → benign. Large negative z → malignant. The steepness at z = 0 is what makes the gradient informative near the boundary.

### `loss_curve.png`
Train and validation loss move in parallel with no divergence — healthy generalization. The steep drop in the first ~300 iterations shows gradient descent making large corrections early. The curve flattens after ~1500 iterations, entering diminishing returns. At 3000 iterations both curves are nearly flat — convergence achieved. Validation loss slightly below training loss is normal for this dataset size.

### `decision_boundary.png`
Plotted on the first two features (mean radius, mean texture) for 2D visualization. The boundary is a straight line — logistic regression is a linear classifier. Malignant tumors (red) cluster at higher mean radius values; benign (green) cluster lower. Misclassified points near the boundary are expected — two features out of 30 capture limited signal.

### `confusion_matrix.png`
- True Negatives (malignant correctly caught): 42
- False Positives (benign flagged as malignant): 1
- False Negatives (malignant missed): 1
- True Positives (benign correctly identified): 70

One missed malignant case in 43. In a medical context, that one FN is a patient whose cancer goes untreated. FP costs an unnecessary biopsy. FN costs a life. Always weight FN more heavily in healthcare applications.

### `precision_recall_tradeoff.png`
At threshold = 0.5, recall for malignant ≈ 0.977. Recall reaches ≥ 0.99 at threshold ≈ 0.85–0.90 for the 3000-iteration model. More training iterations shift the entire recall curve upward — better class separation at every threshold. The default threshold of 0.5 is a reasonable starting point but should always be tuned for the specific cost of each error type.

---

## 📐 Mathematical Derivation

### Why BCE Loss, not MSE?

With MSE, the loss surface for logistic regression is **non-convex** — gradient descent can get stuck in local minima. With BCE, the loss surface is **convex** (guaranteed single global minimum). This is because BCE is derived from maximum likelihood estimation of a Bernoulli distribution, which produces a log-probability objective that is convex when composed with the sigmoid.

### Why Does Loss Start at ln(2) ≈ 0.693?

When weights are initialized to zero, every prediction is exactly 0.5. For a balanced dataset:
$$L = -[y \log(0.5) + (1-y)\log(0.5)] = -\log(0.5) = \log(2) \approx 0.693$$

This is why every logistic regression trained with zero-weight initialization starts at exactly 0.693 — it is mathematically guaranteed, not a coincidence.

### Gradient Derivation

Starting from BCE loss and applying chain rule through the sigmoid:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$$

The beautiful result: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ cancels with the BCE derivative, leaving the clean gradient $\frac{1}{n}X^T(\hat{y} - y)$ — identical in form to linear regression's gradient.

---

## 🐛 Failure Analysis

### Bug 1 — `log(0)` Explosion
**What broke:** Loss became `-inf` on the first iteration.
**Root cause:** Sigmoid outputs values in (0, 1) but never exactly 0 or 1 — except when floating point underflow pushes extremely negative z values to exactly 0.0. `log(0) = -inf`.
**Fix:** `np.clip(y_pred, 1e-15, 1 - 1e-15)` before computing loss.
**Lesson:** Always clip probabilities before any log operation. Make this a reflex.

### Bug 2 — Name Collision in Tests
**What broke:** `test_against_sklearn` threw `TypeError: unexpected keyword argument 'n_iterations'`.
**Root cause:** `from sklearn.linear_model import LogisticRegression` overwrote the custom class imported at the top of the test file. All subsequent instantiations called sklearn's class instead.
**Fix:** Alias the sklearn import: `from sklearn.linear_model import LogisticRegression as SklearnLR`.
**Lesson:** Never import two classes with the same name in the same file without aliasing.

### Bug 3 — Indentation Error Hiding `predict` Inside `predict_proba`
**What broke:** `model.predict()` threw `AttributeError: 'LogisticRegression' object has no attribute 'predict'`.
**Root cause:** `predict` was indented one level too deep — defined as a nested function inside `predict_proba` rather than as a class method.
**Fix:** Align `def predict` at the same indentation level as `def predict_proba`.
**Lesson:** Python silently accepts nested function definitions. The bug only surfaces at call time. Always verify method accessibility after writing a new class.

### Bug 4 — Wrong Variable Name in Gradients
**What broke:** `NameError: name 'n' is not defined`.
**Root cause:** Gradient formula uses `(1/n)` but the variable defined in `fit` is `n_samples`.
**Fix:** Replace `n` with `n_samples` in gradient computation.
**Lesson:** Keep variable names consistent between math notation and code. When in doubt, spell it out: `n_samples` beats `n`.

---

## 🏭 Production Thinking

### Save and Load
```python
import joblib

joblib.dump({
    'model':      model,
    'scaler':     scaler,
    'features':   list(data.feature_names),
    'trained_at': '2026-05-01',
    'test_accuracy': 0.982,
    'threshold':  0.5,
    'params':     {'learning_rate': 0.01, 'n_iterations': 3000}
}, 'results/model.joblib')

# Load and predict
saved = joblib.load('results/model.joblib')
X_scaled = saved['scaler'].transform(X_new)
probs = saved['model'].predict_proba(X_scaled)
preds = (probs >= saved['threshold']).astype(int)
```

### Distribution Shift
The model was trained on 569 samples from a specific hospital imaging protocol. If deployed at a different institution using different imaging equipment or staining techniques, feature distributions will shift. Retraining or fine-tuning on local data is essential before clinical deployment.

### Missing Values
This dataset has no missing values. In production, if any of the 30 features are missing (e.g., an imaging artifact), the model will produce unreliable probabilities. Implement a validation layer that flags incomplete inputs before scoring.

### Feature Mismatch
```python
def validate_and_predict(model, input_df, expected_features):
    if list(input_df.columns) != expected_features:
        raise ValueError(f"Feature mismatch. Expected: {expected_features}")
    return model.predict(input_df[expected_features].values)
```

### Production Checklist
- [ ] Scale features with the same scaler fitted on training data
- [ ] Validate all 30 features are present before scoring
- [ ] Choose threshold based on clinical cost of FN vs FP
- [ ] Log all predictions with confidence scores for audit trail
- [ ] Monitor feature distributions over time for drift
- [ ] Never retrain on production labels without clinical validation

---

## 🚫 When NOT to Use Logistic Regression

| Scenario | Better Alternative | Why |
|----------|-------------------|-----|
| Non-linear decision boundary | SVM (RBF kernel), Random Forest | Logistic regression draws a straight line. Non-linearly separable data needs a curved boundary. |
| Many irrelevant features | Lasso (L1), Random Forest | Logistic regression includes all features. L1 regularization zeroes out irrelevant ones. |
| Multi-class classification (>2 classes) | Softmax regression, Random Forest | Binary logistic regression only outputs one probability. Multi-class needs extension. |
| Extreme class imbalance (1:100+) | XGBoost with `scale_pos_weight`, SMOTE + LR | Severely imbalanced data overwhelms the gradient signal from the minority class. |
| Small dataset with many features | Ridge/Lasso regularized LR, SVM | High-dimensional small datasets cause overfit without explicit regularization. |

---

## ⚠️ Common Mistakes & Gotchas

1. **Accuracy is not enough for imbalanced data.** A model predicting "benign" for every patient scores 62.7% on this dataset. Always check recall and precision separately per class.

2. **The default threshold of 0.5 is rarely optimal.** It assumes equal cost for FP and FN. In cancer detection, FN cost >> FP cost. Always tune the threshold using a precision-recall tradeoff plot.

3. **Fitting the scaler on test data.** `scaler.fit_transform(X_test)` leaks test statistics into the model. Always `fit` on training data only, `transform` on both.

4. **Forgetting to clip probabilities before log.** `log(0) = -inf` and will silently corrupt your loss history. Always use `np.clip(y_pred, 1e-15, 1 - 1e-15)`.

5. **Stopping at the elbow of the loss curve.** The elbow is where learning slows, not where it stops. Stopping there causes underfitting. Plot validation loss alongside training loss to make early stopping decisions correctly.

6. **Using `sum()` instead of `np.sum()`.** Python's built-in `sum` works on NumPy arrays but is slower and inconsistent with the rest of NumPy code. Always use `np.sum()`.

7. **Importing two classes with the same name.** Importing sklearn's `LogisticRegression` after your own silently overwrites your class. Always alias one: `from sklearn.linear_model import LogisticRegression as SklearnLR`.

8. **Assuming sigmoid output direction matches class label.** In this dataset, high probability = benign (class 1). Lowering the threshold makes it harder to predict benign, so more cases are flagged malignant. The direction of threshold tuning depends entirely on which class is encoded as 1.

---

## ❓ 10 Interview Questions

**Q1: Why can't linear regression be used for classification?**

Linear regression produces unbounded outputs — predictions can be negative or greater than 1, which are meaningless as probabilities. The MSE loss treats {0, 1} labels as numeric measurements, so it minimizes numeric error rather than classification error. It can prefer numerically close but wrong-class predictions. For extreme feature values, predictions drift arbitrarily far from [0, 1], making threshold-based decisions unreliable.

**Q2: What is the sigmoid function and why is it used?**

The sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$ squashes any real number to (0, 1). At z = 0, output is 0.5 — the decision boundary. Large positive z → approaches 1. Large negative z → approaches 0. It provides a natural probabilistic interpretation and has a mathematically convenient derivative: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.

**Q3: Why use binary cross-entropy instead of MSE?**

Two reasons. First, BCE produces a convex loss surface for logistic regression — gradient descent is guaranteed to find the global minimum. MSE produces a non-convex surface with potential local minima. Second, BCE penalizes confident wrong predictions exponentially (e.g., predicting 0.99 benign for a malignant tumor incurs massive loss), which aligns with how we want the model to behave.

**Q4: Why does the loss always start at ~0.693?**

With zero-weight initialization, every prediction is exactly 0.5. BCE loss of a 0.5 prediction is $-\log(0.5) = \log(2) \approx 0.693$. This is mathematically guaranteed regardless of dataset size or feature count.

**Q5: What is a decision boundary?**

The set of points where the model predicts probability exactly 0.5 — i.e., where $z = Xw + b = 0$. For logistic regression this is always a hyperplane (a straight line in 2D). Points on one side are classified as class 1, points on the other as class 0. Logistic regression is a linear classifier — it cannot learn curved boundaries without feature engineering.

**Q6: What is precision and recall? When does each matter?**

Precision = TP / (TP + FP): of all positive predictions, how many are correct? Recall = TP / (TP + FN): of all actual positives, how many did we catch? Precision matters when false positives are costly (spam detection — you do not want to delete real emails). Recall matters when false negatives are costly (cancer detection — you do not want to miss tumors). They trade off against each other as threshold changes.

**Q7: What is the confusion matrix and what does each cell mean?**

A 2×2 table of prediction outcomes:
- True Positive (TP): predicted positive, actually positive
- True Negative (TN): predicted negative, actually negative
- False Positive (FP): predicted positive, actually negative — Type I error
- False Negative (FN): predicted negative, actually positive — Type II error

In cancer detection, FN is the most dangerous error — a missed malignant case goes untreated.

**Q8: How does class imbalance affect logistic regression?**

A majority-class dummy classifier can achieve high accuracy without learning anything. The gradient signal from the minority class is overwhelmed by the majority class during training, causing the model to be biased toward predicting the majority class. Fixes include adjusting class weights, oversampling the minority class (SMOTE), undersampling the majority, or choosing metrics like F1 and recall over raw accuracy.

**Q9: What happens if you don't scale features before logistic regression?**

Features with large ranges dominate the gradient updates — the weight for `area` (range ~1000) would update much faster than `smoothness` (range ~0.1). This causes slow, unstable convergence and potentially missing the optimal solution entirely. StandardScaler normalizes all features to zero mean and unit variance, ensuring balanced gradient contributions.

**Q10: How would you decide the threshold in production?**

By constructing a precision-recall tradeoff curve and choosing the threshold that minimizes the cost function for your specific use case. In cancer detection, the cost of a false negative (missed cancer) far exceeds the cost of a false positive (unnecessary biopsy). So you would optimize for high recall on the malignant class, accepting lower precision, and choose the threshold from the curve where recall meets your clinical requirement.

---

## 🏋️ Exercises

### 🟢 Beginner
1. Train the model with `learning_rate=0.001` and `n_iterations=5000`. How does the loss curve change compared to the default? Does accuracy improve?
2. Change the threshold to 0.3 and 0.7. How do precision and recall change for the malignant class? Explain the direction of each change.
3. Remove `StandardScaler` from the pipeline. What happens to training loss and accuracy? Why?

### 🟡 Intermediate
4. Add L2 regularization to the gradient: $\frac{\partial L}{\partial w} = \frac{1}{n}X^T(\hat{y} - y) + \frac{\lambda}{n}w$. Implement `lambda_` as a hyperparameter and test values `[0.01, 0.1, 1.0]`. What happens to accuracy and weights?
5. Implement a `cross_validate` method that performs k-fold cross-validation and returns mean and standard deviation of accuracy across folds.
6. Plot the ROC curve (TPR vs FPR across all thresholds) and compute AUC. Compare to sklearn's implementation.

### 🔴 Advanced
7. Implement mini-batch gradient descent as an option in `fit`. Add a `batch_size` parameter. Compare convergence speed to full-batch gradient descent.
8. Extend the class to handle multi-class classification using the One-vs-Rest (OvR) strategy. Test on the Iris dataset (3 classes).
9. Implement early stopping: monitor validation loss and stop training when it has not improved for `patience` consecutive iterations. Add `patience` as a hyperparameter.

---

## ➡️ What's Next

- **Project 04 — K-Nearest Neighbours:** A non-parametric classifier that makes no assumptions about data distribution. See how distance-based methods compare to logistic regression.
- **Project 05 — Naive Bayes:** A probabilistic classifier based on Bayes' theorem. Extremely fast and effective for text classification.

---

## 📚 Further Reading

1. [Logistic Regression — Stanford CS229 Notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Full mathematical derivation with MLE framing
2. [Cross-Entropy Loss — Deep Learning Book, Chapter 6](https://www.deeplearningbook.org/) — Why BCE is the natural loss for probabilistic classifiers
3. [Precision-Recall vs ROC Curves — Davis & Goadrich (2006)](https://dl.acm.org/doi/10.1145/1143844.1143874) — When PR curves are more informative than ROC
4. [Sklearn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — Production implementation with all solvers
5. [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) — Original UCI repository with feature descriptions