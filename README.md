# Explainable ML for DTC Recurrence Prediction

Predicting recurrence in Differentiated Thyroid Cancer using DT, RF, SVM, and ANN 
with SHAP and LIME explainability.

## Dataset
UCI ML Repository — 383 patients, 16 clinicopathological features.

## Results
| Model | Accuracy | Recall | AUC |
|-------|----------|--------|-----|
| DT    | 0.974    | 0.9091 | 0.9835 |
| RF    | 0.974    | 0.9091 | 0.9872 |
| SVM   | 0.987    | 0.9545 | 0.9967 |
| ANN   | 0.883    | 0.9545 | 0.9802 |

## Usage
```bash
pip install -r requirements.txt
jupyter notebook DTC_Recurrence_Notebook.ipynb
```

## Author
Kounabé Paulin MIEN — Erciyes University, Computer Engineering
