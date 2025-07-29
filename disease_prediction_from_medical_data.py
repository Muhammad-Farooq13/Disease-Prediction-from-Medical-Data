
"""# Task
Analyze the heart disease dataset at "/content/heart.csv" to predict the possibility of heart disease using classification algorithms (SVM, Logistic Regression, Random Forest, XGBoost). Evaluate the performance of each model and select the best one.

## Load the data

### Subtask:
Load the heart disease dataset into a pandas DataFrame.

**Reasoning**:
Import pandas, load the data, and display the head and info of the dataframe to understand its structure.
"""

import pandas as pd

df = pd.read_csv('/content/heart.csv')
display(df.head())
display(df.info())

"""## Preprocess the data

### Subtask:
Handle missing values, encode categorical features, and scale numerical features as necessary.

**Reasoning**:
Identify categorical columns, apply one-hot encoding, separate features and target, identify numerical columns in the transformed data, scale numerical columns, and replace original numerical columns with scaled values.
"""

categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

display(X.head())

"""## Split the data

### Subtask:
Split the dataset into training and testing sets.

**Reasoning**:
I need to split the preprocessed data into training and testing sets. I will import the `train_test_split` function from `sklearn.model_selection` and then use it to split the `X` and `y` dataframes into training and testing sets, with a test size of 20% and a random state of 42 for reproducibility.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

"""## Train classification models

### Subtask:
Train various classification models (SVM, Logistic Regression, Random Forest, XGBoost) on the training data.

**Reasoning**:
Import the necessary classification models and train them on the training data.
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

svm_model = SVC()
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
xgb_model = XGBClassifier()

svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

print("Models trained successfully.")

"""## Evaluate the models

### Subtask:
Evaluate the performance of each model on the testing data using appropriate metrics like accuracy, precision, recall, and F1-score.

**Reasoning**:
Import the necessary evaluation metrics and create a dictionary to store the results. Then, evaluate each trained model on the test data and store the results in the dictionary. Finally, print the evaluation results.
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

evaluation_results = {}

models = {
    "SVM": svm_model,
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    evaluation_results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

for name, metrics in evaluation_results.items():
    print(f"--- {name} ---")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1-score: {metrics['F1-score']:.4f}")
    print("-" * (len(name) + 6))

"""## Select the best model

### Subtask:
Choose the best-performing model based on the evaluation metrics.

**Reasoning**:
Analyze the evaluation results to identify the best model based on the F1-score and store it.
"""

best_f1 = 0
best_model_name = None

for name, metrics in evaluation_results.items():
    if metrics['F1-score'] > best_f1:
        best_f1 = metrics['F1-score']
        best_model_name = name

best_model = models[best_model_name]

print(f"The best performing model based on F1-score is: {best_model_name}")

"""## Make predictions

### Subtask:
Use the best model to make predictions on new data.

**Reasoning**:
Use the best model to make predictions on the test data and display the first few predictions.
"""

y_pred_best_model = best_model.predict(X_test)
print("First few predictions from the best model:")
display(y_pred_best_model[:10])

"""## Summary:

### Data Analysis Key Findings

*   The dataset contains 1025 entries and 14 columns with no missing values.
*   The dataset was successfully preprocessed by encoding categorical features and scaling numerical features.
*   The dataset was split into training (80%) and testing (20%) sets.
*   Four classification models (SVM, Logistic Regression, Random Forest, and XGBoost) were trained on the training data.
*   Model evaluation on the test set showed varying performance:
    *   Random Forest achieved perfect scores (Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000, F1-score: 1.0000).
    *   XGBoost also performed very well (Accuracy: 0.9854, Precision: 1.0000, Recall: 0.9709, F1-score: 0.9852).
    *   SVM had an Accuracy of 0.8732 and F1-score of 0.8889.
    *   Logistic Regression had the lowest performance (Accuracy: 0.8195, F1-score: 0.8333).
*   Based on the F1-score, Random Forest was selected as the best-performing model.
*   Predictions were successfully made on the test data using the best model (Random Forest).

### Insights or Next Steps

*   Investigate the perfect scores achieved by the Random Forest model to ensure there is no data leakage or overfitting. Cross-validation could be used for a more robust evaluation.
*   Tune the hyperparameters of the top-performing models (Random Forest and XGBoost) to potentially improve performance and generalization.

"""
