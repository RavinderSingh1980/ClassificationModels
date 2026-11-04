# Dry Bean Classification using Machine Learning

## a. Problem statement

The objective of this project is to predict the variety of a dry bean using 16 numerical physical and geometric features. The application compares five supervised classification models and evaluates their performance on a held-out test set, while allowing the user to upload test CSV data for prediction and evaluation.

## b. Dataset description

The dataset is the Dry Bean Dataset from the UCI Machine Learning Repository. It contains **13,611 instances, 16 input features, and 1 multiclass target column** (`Class`). It has seven classes: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, and BOMBAY. The repository uses the dataset's Excel version, `Dry_Bean_Dataset.xlsx`. There are no missing values and 68 duplicate rows; no duplicates were removed because all five models must be evaluated consistently on the same dataset.

The project performs an 80/20 stratified train-test split with `random_state=42`. Feature scaling is used for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM. The trained artifacts and test CSV are stored in this repository.

## c. GitHub Repository Link

Repository: **Add the final GitHub repository URL here before submission.**

## d. Models used

The following weighted metrics were calculated on the held-out 20% test set. AUC is multiclass weighted one-vs-rest AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9934 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| K-Nearest Neighbors | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Gaussian Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

### Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high accuracy, AUC, and MCC. Scaling supports stable linear decision boundaries across the bean classes. |
| Decision Tree | Lowest accuracy and AUC among the five models, suggesting less generalization than the stronger linear, distance, and margin-based methods. |
| K-Nearest Neighbors | Competitive performance and a high AUC after feature scaling, although its predictions are generally slower and more memory-intensive at inference time. |
| Gaussian Naive Bayes | Good class separation and a high AUC, but the conditional-independence assumption likely limits it compared with Logistic Regression and SVM. |
| Support Vector Machine | **Overall winner:** achieved the highest accuracy, AUC, precision, recall, F1, and MCC by a small margin over Logistic Regression. |
| Overall winner | The Support Vector Machine is the best-performing model for this dataset, with Logistic Regression a close second. |

## Repository contents

- `app.py`: Streamlit interactive frontend
- `train_models.py`: Reproducible model training source code
- `requirements.txt`: Deployment dependencies
- `README.md`: Assignment documentation
- `test_data.csv`: Held-out test data containing features and the `Class` target
- `model/`: Saved encoder, five model pipelines, and training metrics
- `Dry_Bean_Dataset.xlsx`: Source dataset

## Streamlit features

The app lets users upload a CSV containing all required input features and the `Class` column, select one of the saved models, and view accuracy, weighted AUC, weighted precision, weighted recall, weighted F1, MCC, a confusion matrix, and a classification report. Models are loaded from disk and are not retrained using uploaded test data.

## Running locally

```bash
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```
