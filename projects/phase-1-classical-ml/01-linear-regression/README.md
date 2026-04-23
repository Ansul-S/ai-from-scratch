# 01 — Linear Regression from Scratch

> **Difficulty:** 🟢 Easy  
> **Time to complete:** ~6–8 hours  
> **Phase:** Phase 1 — Classical ML Foundations  
> **Tags:** `regression` `gradient-descent` `numpy` `statistics` `supervised-learning`

---

## 📌 What You'll Learn

By the end of this project you will be able to:

- [ ] Explain linear regression in plain English and to a non-technical audience
- [ ] Derive the cost function (MSE) and understand why we minimise it
- [ ] Implement gradient descent from scratch using only NumPy
- [ ] Build a complete `LinearRegression` class with fit, predict, score, and MSE methods
- [ ] Solve linear regression analytically using the Normal Equation
- [ ] Add polynomial features to capture non-linear relationships
- [ ] Apply Ridge (L2) regularization and understand the bias-variance tradeoff
- [ ] Diagnose your model using residual plots and Q-Q plots
- [ ] Explain every line of your code in a technical interview

---

## 📦 Dataset — Medical Insurance Charges

**File:** `data/insurance.csv`  
**Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) (CC0 Public Domain)  
**Size:** 1,338 rows × 7 columns

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Age of the primary beneficiary |
| `sex` | categorical | Gender — male / female |
| `bmi` | float | Body Mass Index (weight/height²) |
| `children` | int | Number of dependents |
| `smoker` | categorical | Smoking status — yes / no |
| `region` | categorical | US region — northeast, northwest, southeast, southwest |
| `charges` | float | **Target** — Annual medical insurance cost in USD |

### Why this dataset?

This dataset is perfect for learning linear regression because it has one dominant feature (`smoker`) that explains ~79% of the variance in charges on its own. Adding that one binary feature jumps R² from 0.12 to 0.75 — making the lesson of feature importance unforgettable.

---

## 💡 Intuition First

### What is Linear Regression?

Imagine you want to predict how much someone will pay for insurance based on their age. You collect data from 1,338 people and plot it — age on the x-axis, charges on the y-axis. You now have a scatter plot with 1,338 dots.

Linear regression draws the **single best straight line** through those dots. "Best" means the line is as close as possible to all the dots at once.

Once you have that line, predicting for a new person is simple — find their age on the x-axis, go straight up to the line, and read off the predicted charge on the y-axis.

### The Real-World Analogy

Think of it like a salary calculator. A company might say: *"For every year of experience, we pay $5,000 more, starting from a base of $40,000."*

That's linear regression:
```
salary = 5000 × experience + 40000
```

`5000` is the weight (slope), `40000` is the bias (intercept). Linear regression's job is to find these numbers from data.

### What "Learning" Actually Means

The model starts with random weights (or zeros). It makes terrible predictions at first. Then it looks at how wrong it was, figures out which direction to adjust the weights, and takes a small step in that direction. It does this thousands of times until the predictions stop improving.

This process of repeatedly adjusting weights to reduce error is called **gradient descent** — and it's the foundation of almost all machine learning, including deep learning.

---

## 🔢 How It Works — The Math

### Simple Linear Regression

For one feature:

$$\hat{y} = wx + b$$

Where:
- $\hat{y}$ = predicted value
- $x$ = input feature
- $w$ = weight (slope of the line)
- $b$ = bias (y-intercept)

### Multiple Linear Regression

For multiple features, we use matrix form:

$$\hat{y} = X\theta$$

Where:
- $X$ is the feature matrix of shape $(m \times n)$ — $m$ samples, $n$ features
- $\theta$ is the weight vector of shape $(n \times 1)$
- $\hat{y}$ is the prediction vector of shape $(m \times 1)$

In code: `y_pred = X @ weights + bias`

### The Cost Function — MSE

We need a way to measure how wrong our predictions are. We use **Mean Squared Error**:

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

**Why squared?**
1. Squaring makes all errors positive (a -$5000 error is just as bad as +$5000)
2. Squaring penalises large errors more than small ones
3. It's mathematically smooth — easy to differentiate

### Gradient Descent

To minimise $J(\theta)$, we compute the gradient (derivative) of the cost with respect to each weight, then move in the opposite direction:

**Gradient for weights:**
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$

**Gradient for bias:**
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)$$

**Update rule:**
$$w := w - \alpha \cdot \frac{\partial J}{\partial w}$$
$$b := b - \alpha \cdot \frac{\partial J}{\partial b}$$

Where $\alpha$ is the **learning rate** — how big a step we take each iteration.

### From Math to Code

| Equation | Code |
|----------|------|
| $\hat{y} = X\theta + b$ | `y_pred = X @ self.weights + self.bias` |
| $error = \hat{y} - y$ | `error = y_pred - y` |
| $\frac{\partial J}{\partial w} = \frac{1}{m} X^T \cdot error$ | `dw = (1/n_samples) * X.T @ error` |
| $\frac{\partial J}{\partial b} = \frac{1}{m} \sum error$ | `db = (1/n_samples) * np.sum(error)` |
| $w := w - \alpha \cdot dw$ | `self.weights = self.weights - lr * dw` |
| $J = \frac{1}{m}\sum(error^2)$ | `np.mean((y_pred - y) ** 2)` |

### The Normal Equation

Instead of iterating, we can solve for the optimal weights analytically in one shot:

$$\theta = (X^TX)^{-1}X^Ty$$

**Pros:** No learning rate to tune, exact solution in one calculation.  
**Cons:** Computing $(X^TX)^{-1}$ is $O(n^3)$ — extremely slow for large feature sets.

**Rule of thumb:** Use gradient descent when $n > 10,000$ features. Use Normal Equation for smaller datasets.

### Ridge Regularization (L2)

To prevent overfitting, we add a penalty term to the cost function:

$$J(\theta) = \frac{1}{m} \sum (\hat{y} - y)^2 + \lambda \sum w_j^2$$

This forces weights to stay small. The gradient update becomes:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T(\hat{y} - y) + \lambda w$$

Where $\lambda$ (lambda) controls the strength of regularization:
- $\lambda = 0$ → no regularization (standard regression)
- $\lambda$ large → weights shrink towards zero → model becomes simpler

### R² Score

R² measures how much of the variance in $y$ our model explains:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y - \hat{y})^2}{\sum(y - \bar{y})^2}$$

- $R^2 = 1.0$ → perfect predictions
- $R^2 = 0.0$ → model is no better than predicting the mean
- $R^2 < 0$ → model is worse than predicting the mean (possible!)

---

## 🏗️ Build From Scratch

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn pytest
```

### Project Structure

```
01-linear-regression/
├── data/
│   └── insurance.csv
├── results/
│   ├── charges_distribution.png
│   ├── charges_by_smoker.png
│   ├── age_vs_charges.png
│   ├── correlation_heatmap.png
│   ├── loss_curve.png
│   ├── ridge_regularization.png
│   ├── residuals_vs_fitted.png
│   └── residuals_log.png
├── tests/
│   ├── __init__.py
│   └── test_solution.py
├── notebook.ipynb
├── solution.py
└── README.md
```

### Step 1 — Exploratory Data Analysis

Before building any model, understand your data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/insurance.csv')

print(df.shape)        # (1338, 7)
print(df.dtypes)       # age: int, bmi: float, sex/smoker/region: object
print(df.isnull().sum()) # All zeros — clean dataset
print(df.describe())   # Note: charges mean=13270, max=63770 — huge spread
```

**Key insights from EDA:**
- `charges` is right-skewed — most people pay under $15k, but a tail extends to $63k
- The Age vs Charges scatter plot shows **3 distinct bands** — caused by smoking status
- `smoker` has a correlation of **0.79** with charges — far higher than any other feature
- Adding `smoker` as a feature jumps R² from 0.12 → 0.75 in one step

```python
# Verify smoker dominance
df['smoker_encoded'] = (df['smoker'] == 'yes').astype(int)
print(df[['age','bmi','children','smoker_encoded','charges']].corr()['charges'])

# age              0.299
# bmi              0.198
# children         0.068
# smoker_encoded   0.787  ← by far the highest
```

### Step 2 — Data Preparation

```python
# Reload clean
df = pd.read_csv('data/insurance.csv')

# Encode binary categoricals
df['smoker'] = (df['smoker'] == 'yes').astype(int)
df['sex']    = (df['sex'] == 'male').astype(int)

# One-hot encode region — drop_first=True avoids dummy variable trap
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Separate features and target
X = df.drop('charges', axis=1).values.astype(float)
y = df['charges'].values.astype(float)

# Manual train/test split (80/20)
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Standardise — fit on TRAIN only to prevent data leakage
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test  - mean) / std
```

> ⚠️ **Critical:** Always compute mean/std from training data only. Using the full dataset leaks test information into training and makes your model look artificially better.

### Step 3 — The LinearRegression Class

```python
import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.weights       = None
        self.bias          = None
        self.loss_history  = []

    def fit(self, X, y, l2=0.0):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias    = 0

        for _ in range(self.n_iterations):
            # Step 1 — make prediction
            y_pred = X @ self.weights + self.bias

            # Step 2 — compute error
            error = y_pred - y

            # Step 3 — compute gradients (+ L2 penalty on weights)
            dw = (1 / n_samples) * X.T @ error + (l2 * self.weights)
            db = (1 / n_samples) * np.sum(error)

            # Step 4 — update weights
            self.weights = self.weights - self.learning_rate * dw
            self.bias    = self.bias    - self.learning_rate * db

            # Track loss each iteration
            self.loss_history.append(self.mse(X, y))

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

    def mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)

    def fit_normal_equation(self, X, y):
        # θ = (XᵀX)⁻¹Xᵀy
        X_b   = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.bias    = theta[0]
        self.weights = theta[1:]
        return self
```

### Step 4 — Feature Importance Experiment

```python
feature_names = ['age','sex','bmi','children','smoker',
                 'region_nw','region_se','region_sw']

for i in range(1, len(feature_names) + 1):
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train_scaled[:, :i], y_train)
    r2 = model.score(X_test_scaled[:, :i], y_test).round(4)
    print(f"Features {feature_names[:i]} → R²: {r2}")
```

**Results:**

| Features Added | R² |
|---------------|-----|
| age | 0.094 |
| + sex | 0.087 |
| + bmi | 0.123 |
| + children | 0.128 |
| + **smoker** | **0.751** ← the jump |
| + region_nw | 0.751 |
| + region_se | 0.750 |
| + region_sw | 0.751 |

One binary feature (smoker) contributed more than all other features combined.

### Step 5 — Normal Equation Comparison

```python
model_ne = LinearRegression()
model_ne.fit_normal_equation(X_train_scaled, y_train)

model_gd = LinearRegression(learning_rate=0.01, n_iterations=1000)
model_gd.fit(X_train_scaled, y_train)

print("Normal Equation R²:", model_ne.score(X_test_scaled, y_test).round(4))
print("Gradient Descent R²:", model_gd.score(X_test_scaled, y_test).round(4))
# Both produce identical R² and near-identical weights
```

Both methods solve the same problem, just via different paths. Gradient descent scales to large datasets; Normal Equation is exact but expensive for many features.

### Step 6 — Polynomial Features

```python
class PolynomialFeatures:
    def __init__(self, degree=2):
        self.degree = degree

    def fit_transform(self, X, feature_idx=0):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            new_col = X[:, feature_idx] ** d
            X_poly  = np.c_[X_poly, new_col]
        return X_poly
```

Adding `bmi²` and scaling the new column:

```python
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled, feature_idx=2)
X_test_poly  = poly.fit_transform(X_test_scaled,  feature_idx=2)

# Scale the new polynomial column
poly_mean = X_train_poly[:, 8].mean()
poly_std  = X_train_poly[:, 8].std()
X_train_poly[:, 8] = (X_train_poly[:, 8] - poly_mean) / poly_std
X_test_poly[:, 8]  = (X_test_poly[:, 8]  - poly_mean) / poly_std
```

> ⚠️ **Always scale polynomial features.** Squaring amplifies the scale dramatically and causes gradient overflow if left unscaled.

### Step 7 — Ridge Regularization

```python
lambdas = [0, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]

for l2 in lambdas:
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train_scaled, y_train, l2=l2)
    r2 = model.score(X_test_scaled, y_test).round(4)
    print(f"λ={l2:6} → R²: {r2}")
```

**Results:**

| Lambda (λ) | R² |
|-----------|-----|
| 0 | 0.7513 |
| 0.1 | 0.7405 |
| 1 | 0.5484 |
| 10 | 0.1264 |
| 100 | 0.0141 |

As λ increases, weights shrink toward zero and the model becomes too simple (underfitting).

### Step 8 — Diagnostics

```python
# Residuals vs Fitted
y_pred_train = model_gd.predict(X_train_scaled)
residuals    = y_train - y_pred_train

plt.scatter(y_pred_train, residuals, alpha=0.4)
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.show()
```

**What to look for:**
- ✅ Random cloud around zero → assumptions satisfied
- ❌ Pattern/funnel shape → heteroscedasticity (error variance not constant)
- ❌ Curved pattern → non-linearity not captured

Our residual plot shows a funnel shape — heteroscedasticity caused by smokers. The fix is a log transform on the target:

```python
y_train_log = np.log(y_train)
y_test_log  = np.log(y_test)

model_log = LinearRegression(learning_rate=0.01, n_iterations=1000)
model_log.fit(X_train_scaled, y_train_log)

print("R² with log target:", model_log.score(X_test_scaled, y_test_log).round(4))
# 0.7663 — slightly better, and residuals are much better behaved
```

---

## ⚠️ Common Mistakes & Gotchas

1. **Fitting the scaler on test data** — Always compute mean/std from train set only. Using full data leaks future information and makes metrics misleadingly optimistic.

2. **Not scaling before gradient descent** — Features with large scales (e.g. `charges` in thousands) dominate gradients. Always standardise first.

3. **Forgetting to scale polynomial features** — Squaring a feature amplifies its scale dramatically. This causes gradient overflow (NaN loss). Always scale after adding polynomial terms.

4. **Confusing R² with accuracy** — R² of 0.75 doesn't mean the model is "75% accurate". It means the model explains 75% of the variance in the target. The remaining 25% may be genuinely unexplainable noise.

5. **High R² = good model (false!)** — A model can have R²=0.99 by overfitting. Always check train vs test R². If train >> test, you're overfitting.

6. **Using gradient descent without feature scaling** — Without scaling, some gradients will be enormous (large-scale features) and others tiny (small-scale features). The loss landscape becomes elongated and gradient descent zigzags instead of converging cleanly.

7. **Dummy variable trap** — One-hot encoding $k$ categories creates $k$ columns, but you only need $k-1$. Keeping all $k$ creates perfect multicollinearity — the model can't distinguish the effect of each category.

---

## 🎯 Interview Questions

<details>
<summary><strong>Q1: Explain linear regression to someone with no ML background.</strong></summary>

**Answer:** Linear regression finds the best straight line through a set of data points. Given a set of inputs (like age, BMI, smoking status), it learns a formula that multiplies each input by a weight, adds them up, and predicts an output (like insurance charges). "Learning" means adjusting those weights repeatedly until the predictions are as close as possible to the real values. Once trained, you can plug in any new inputs and get a prediction instantly.

</details>

<details>
<summary><strong>Q2: Why do we minimise MSE instead of MAE? When would you prefer MAE?</strong></summary>

**Answer:** MSE (Mean Squared Error) is preferred because squaring errors makes it smooth and differentiable everywhere — critical for gradient descent. It also penalises large errors more heavily, which is often desirable. However, this sensitivity to large errors also makes MSE sensitive to outliers. MAE (Mean Absolute Error) is more robust to outliers because it doesn't square errors. You'd prefer MAE when your dataset has significant outliers you don't want to dominate training — for example, predicting house prices where a few mansions shouldn't skew the model.

</details>

<details>
<summary><strong>Q3: What are the 4 key assumptions of linear regression?</strong></summary>

**Answer:**
1. **Linearity** — the relationship between features and target is linear. Check with scatter plots.
2. **Independence** — observations are independent of each other. Violated with time-series data.
3. **Homoscedasticity** — residuals have constant variance across all fitted values. Check with residuals vs fitted plot — should look like a random cloud.
4. **Normality of residuals** — residuals are normally distributed. Check with a Q-Q plot.

Violations don't make the model useless, but they affect the reliability of inference (p-values, confidence intervals). The insurance dataset violates homoscedasticity due to smokers — a log transform on the target helps.

</details>

<details>
<summary><strong>Q4: What is the difference between gradient descent and the Normal Equation?</strong></summary>

**Answer:** Both find the weights that minimise the cost function, but via different paths. Gradient descent is iterative — it takes small steps in the direction of steepest descent, requiring a learning rate and many iterations. The Normal Equation solves for the exact optimal weights analytically in one step: θ = (XᵀX)⁻¹Xᵀy. The tradeoff is computational cost — the Normal Equation requires computing a matrix inverse which is O(n³) in the number of features. For small feature sets (n < 10,000), Normal Equation is simpler. For large datasets or many features, gradient descent is far more efficient.

</details>

<details>
<summary><strong>Q5: What is multicollinearity and why is it a problem?</strong></summary>

**Answer:** Multicollinearity occurs when two or more features are highly correlated with each other — for example, having both `weight_kg` and `weight_lb` in the same model. The problem is that the model can't distinguish the individual effect of each correlated feature on the target, leading to unstable and unreliable weight estimates. A small change in data can cause large swings in coefficients. You can detect it with the Variance Inflation Factor (VIF) — a VIF > 10 signals a problem. Fixes include removing one of the correlated features, combining them, or using Ridge regularization which shrinks correlated weights together.

</details>

<details>
<summary><strong>Q6: Can R² be negative? Can a high R² mean your model is bad?</strong></summary>

**Answer:** Yes to both. R² can be negative when your model is worse than simply predicting the mean every time — this happens when the model is severely underfitted or when you evaluate on test data with a model trained on very different data. A high R² can still indicate a bad model in two scenarios: (1) **Overfitting** — training R²=0.99 but test R²=0.50 means the model memorised training data and generalises poorly. (2) **Spurious correlation** — R² measures fit, not causation. A model can have high R² by fitting noise rather than signal.

</details>

<details>
<summary><strong>Q7: What is regularization? Explain Ridge (L2) vs Lasso (L1).</strong></summary>

**Answer:** Regularization adds a penalty term to the cost function to prevent overfitting by discouraging large weights. Ridge (L2) adds the sum of squared weights: λΣw². Lasso (L1) adds the sum of absolute weights: λΣ|w|. The key difference is in how they shrink weights — Ridge shrinks all weights proportionally toward zero but never exactly to zero. Lasso can shrink weights exactly to zero, effectively performing feature selection. If you have many irrelevant features, Lasso is preferable because it removes them entirely. If features are all relevant but correlated, Ridge is better because it distributes the weight among correlated features rather than zeroing some arbitrarily.

</details>

<details>
<summary><strong>Q8: What happens when the learning rate is too high or too low?</strong></summary>

**Answer:** If the learning rate is **too high**, the weight updates overshoot the minimum — the loss oscillates or diverges to infinity instead of converging. You'll see the loss curve bouncing or exploding rather than decreasing smoothly. If the learning rate is **too low**, convergence is extremely slow — the model takes thousands of iterations to make meaningful progress. In practice, you try a few values (0.001, 0.01, 0.1) and plot the loss curve to find the sweet spot. Modern deep learning uses adaptive learning rate methods like Adam that adjust the rate automatically per parameter.

</details>

<details>
<summary><strong>Q9: Why do we use one-hot encoding for categorical variables? What is the dummy variable trap?</strong></summary>

**Answer:** Linear regression requires numeric inputs — it can't work with strings like "northeast" or "smoker". One-hot encoding converts a categorical column with k categories into k binary columns (0 or 1). The dummy variable trap occurs when you include all k columns — since the k columns always sum to 1, they are perfectly collinear, which makes the matrix XᵀX non-invertible (breaks the Normal Equation) and causes multicollinearity. The fix is to always drop one category (`drop_first=True` in pandas), using it as the baseline. The coefficients of the remaining columns represent the difference in outcome relative to that baseline.

</details>

<details>
<summary><strong>Q10: When should you NOT use linear regression?</strong></summary>

**Answer:** Linear regression is a poor choice when: (1) **The relationship is non-linear** — if the true relationship curves or has interactions, a straight line will systematically mis-predict. Use polynomial features, tree-based models, or neural networks instead. (2) **The target is categorical** — use logistic regression or classification models. (3) **There are severe outliers** — MSE amplifies their effect. Use robust regression or tree-based models. (4) **Features are highly correlated** — multicollinearity makes coefficients unstable. (5) **The data violates independence** — time series data requires specialised models like ARIMA or LSTMs. In our insurance dataset, the non-linear smoker×BMI interaction is the main limitation of linear regression.

</details>

---

## 🏋️ Exercises & Challenges

**🟢 Beginner**
- [ ] Train on `age` only. Plot the regression line over the scatter plot.
- [ ] Manually compute MSE and R² without using class methods — verify they match.
- [ ] Try learning rates `[0.001, 0.01, 0.1]` and plot all 3 loss curves on the same chart.
- [ ] Add features one at a time and build a table of R² values. Which single feature helps most?

**🟡 Intermediate**
- [ ] Compare gradient descent vs Normal Equation coefficients on the same dataset — do they match?
- [ ] Implement `StandardScaler` from scratch (without sklearn) — fit on train, transform both.
- [ ] Add the `smoker × bmi` interaction term manually and measure R² improvement.
- [ ] Apply log transform to `charges`. Does it improve residual normality?

**🔴 Advanced**
- [ ] Implement Lasso (L1) regularization. Why is it harder to optimise than Ridge?
- [ ] Implement k-fold cross-validation from scratch to find the optimal Ridge λ.
- [ ] Implement mini-batch gradient descent and compare convergence speed to full-batch.
- [ ] Add learning rate scheduling (decay over iterations). Does it reach a better minimum?

---

## 🔗 What's Next

- **[02 — Polynomial Regression →](../02-polynomial-regression/)** — Extends this project by systematically adding polynomial features and studying overfitting vs underfitting in depth.
- **[03 — Logistic Regression →](../03-logistic-regression/)** — Adapts the same gradient descent framework to classification problems using the sigmoid function.

---

## 📚 Further Reading

- [CS229 Lecture Notes — Linear Regression](https://cs229.stanford.edu/notes/) — Stanford's gold-standard derivation of the cost function and gradient descent
- [Mathematics for ML — Chapter 9](https://mml-book.github.io/) — Rigorous linear regression from a linear algebra perspective
- [StatQuest — Linear Regression](https://www.youtube.com/@statquest) — Best visual explanation of R², p-values, and assumptions
- [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus) — Visual intuition for derivatives and gradients

---

<p align="center">
  <strong>ai-from-scratch</strong> · Project 01 · Linear Regression<br>
  Built with curiosity · Shared with the community
</p>