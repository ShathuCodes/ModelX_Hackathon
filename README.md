# ModelX_Hackathon

# Dementia Risk Prediction from Non-Medical Factors

This project builds a binary classification model to predict the risk of dementia using only non-medical data, such as lifestyle, education, and social context.

The model is built from scratch using **NumPy** for logistic regression and gradient descent.

---

## Project Goal & Scenario

As dementia becomes a growing global health issue, this project explores how well non-medical information alone can help predict dementia risk.

our goal : build a binary classification model that predicts whether a person is at risk of dementia or not, expressed as a probability (0–100%), using only non-medical variables from the dataset.

## Dataset

This model was trained on the `Dementia Prediction Dataset.csv`. The data was loaded from Google Drive for the analysis within the Colab notebook.

## Methodology

The entire process, from data cleaning to model training, is contained in the notebook.

### 1. Feature Selection & Preprocessing
A specific subset of non-medical features was selected for the model. The process included:
* **Feature Selection:** Isolating columns related to demographics, lifestyle (smoking, alcohol), and living situation.
* **Data Validation:** Applying strict validation rules to filter out rows with invalid or missing data codes (e.g., -4, 9).
* **Normalization:** Scaling numerical features (like `NACCMMSE`, `NACCAGE`, `EDUC`) to a common range (0 to 1) to help the model converge.

### 2. Model: Logistic Regression from Scratch
Instead of using a pre-built library like Scikit-learn, this model is a **Logistic Regression** classifier built from the ground up using NumPy.
* **Sigmoid Function:** Used to map predictions to a probability between 0 and 1.
* **Loss Function:** Binary Cross-Entropy is used to measure the model's error.
* **Gradient Descent:** The model's weights are optimized over 5,000 iterations using gradient descent to minimize the loss.

## Results

The model's performance was tracked during training by plotting the **Loss** and **Accuracy** over all iterations.



The final model achieves a training accuracy of **       **.

## How to Use the Model

The trained model is saved to a file named `dementia_model.p` using `pickle`. This file contains three key components needed to make new predictions:
1.  **`weights`**: The optimized model parameters.
2.  **`X_mean`**: The mean used for scaling the training data.
3.  **`X_std`**: The standard deviation used for scaling the training data.

To make a prediction on new data, you must load this `pickle` file and apply the *same* scaling (using `X_mean` and `X_std`) to the new data before feeding it to the model.
