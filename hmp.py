import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# 1. Load dataset
loan_dataset = pd.read_csv('Loan Data.csv')

print("First 5 rows of dataset:")
print(loan_dataset.head())

# 2. Basic info
print("\nShape (rows, columns):", loan_dataset.shape)

print("\nStatistical summary:")
print(loan_dataset.describe())

print("\nMissing values before dropping:")
print(loan_dataset.isnull().sum())

# 3. Drop rows with missing values
loan_dataset = loan_dataset.dropna()

print("\nMissing values after dropping:")
print(loan_dataset.isnull().sum())

# 4. Encode target column Loan_Status (Y/N -> 1/0)
loan_dataset.replace({'Loan_Status': {'N': 0, 'Y': 1}}, inplace=True)

print("\nAfter encoding Loan_Status:")
print(loan_dataset.head())

# 5. Check Dependents
print("\nDependents value counts BEFORE replace:")
print(loan_dataset['Dependents'].value_counts())

# Replace '3+' with 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

print("\nDependents value counts AFTER replace:")
print(loan_dataset['Dependents'].value_counts())

# 6. Convert categorical columns to numeric
loan_dataset.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

print("\nDataset after encoding categorical features:")
print(loan_dataset.head())

# 7. Split into features (X) and target (Y)
X = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
Y = loan_dataset['Loan_Status']

print("\nFeature matrix X (first 5 rows):")
print(X.head())

print("\nLabel vector Y (first 5 values):")
print(Y.head())

# 8. Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,
    stratify=Y,
    random_state=2
)

print("\nShapes:")
print("X:", X.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# 9. Create SVM classifier
classifier = svm.SVC(kernel='linear')

# Train model
classifier.fit(X_train, Y_train)

# 10. Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('\nAccuracy on training data : ', training_data_accuracy)

# 11. Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy on test data : ', test_data_accuracy)