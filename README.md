# 🧠 AI From Scratch

> A hands-on library of AI/ML mini-projects — built from the ground up, one concept at a time.

This repository is designed for **two kinds of people**:
- **Job seekers** preparing for Applied AI / ML Engineer interviews who want a strong GitHub portfolio.
- **Beginners** who learn best by building, not just reading.

Every project follows the same structure: intuition first, math second, code third.

---

## 🗺️ How to use this repo

**If you're a beginner** — start at Phase 1 and work through projects in order.  
**If you're interview-prepping** — jump to any project by concept.  
**If you want to contribute** — read CONTRIBUTING.md.

---

## 📚 Project Index

### Phase 1 — Classical ML Foundations
| # | Project | Concepts | Difficulty |
|---|---------|----------|------------|
| 01 | [Linear Regression](./projects/phase-1-classical-ml/01-linear-regression/) | Gradient descent, MSE, R² | 🟢 Easy |
| 02 | [Polynomial Regression](./projects/phase-1-classical-ml/02-polynomial-regression/) | Feature engineering, overfitting | 🟢 Easy |
| 03 | Logistic Regression | Sigmoid, log loss, decision boundary | 🟢 Easy |
| 04 | K-Nearest Neighbours | Distance metrics, hyperparameters | 🟢 Easy |
| 05 | Naive Bayes | Bayes theorem, conditional probability | 🟢 Easy |
| 06 | Decision Tree | Information gain, Gini impurity | 🟡 Medium |
| 07 | K-Means Clustering | Unsupervised learning, elbow method | 🟡 Medium |
| 08 | PCA | Dimensionality reduction, eigenvectors | 🟡 Medium |

### Phase 2 — Ensemble & Advanced Classical ML
| # | Project | Concepts | Difficulty |
|---|---------|----------|------------|
| 09 | Random Forest | Bagging, feature importance | 🟡 Medium |
| 10 | Gradient Boosting | Boosting, residuals, learning rate | 🟡 Medium |
| 11 | XGBoost from Scratch | Regularization, tree pruning | 🔴 Hard |
| 12 | Support Vector Machine | Hyperplane, kernel trick, margin | 🔴 Hard |

### Phase 3 — Deep Learning Basics
| # | Project | Concepts | Difficulty |
|---|---------|----------|------------|
| 15 | Perceptron from Scratch | Weights, bias, activation functions | 🟡 Medium |
| 16 | MLP with Backpropagation | Backprop, chain rule | 🔴 Hard |
| 17 | Image Classifier (CNN) | Convolution, pooling | 🔴 Hard |

### Phase 4 — Modern AI & Applied LLMs
| # | Project | Concepts | Difficulty |
|---|---------|----------|------------|
| 20 | Transformer Attention | Self-attention, Q/K/V | 🔴 Hard |
| 21 | Fine-tune a Small LLM | LoRA, instruction tuning | 🔴 Hard |
| 22 | RAG Pipeline from Scratch | Embeddings, vector search | 🔴 Hard |

### Phase 5 — MLOps & Production
| # | Project | Concepts | Difficulty |
|---|---------|----------|------------|
| 25 | Model Versioning with MLflow | Experiment tracking | 🟡 Medium |
| 26 | REST API for ML Model | FastAPI, model serving | 🟡 Medium |

---

## 🛠️ Setup

```bash
git clone https://github.com/Ansul-S/ai-from-scratch.git
cd ai-from-scratch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📦 Rule

Core algorithms are implemented from scratch using only NumPy. scikit-learn is used solely for loading datasets and benchmarking.

---

<p align="center">Built with curiosity · Shared with the community</p>
