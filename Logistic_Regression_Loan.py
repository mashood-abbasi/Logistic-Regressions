import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1️⃣ Load dataset
csv_path = os.path.join("DataSet", "loan_dataset.csv")
df = pd.read_csv(csv_path)
print("Dataset Preview:")
print(df.head())

# 2️⃣ Keep only relevant columns
columns_to_keep = [
    'applicant_name', 'gender', 'age', 'city', 'employment_type',
    'monthly_income_pkr', 'loan_amount_pkr', 'loan_tenure_months',
    'existing_loans', 'default_history', 'has_credit_card', 'approved'
]
df = df[columns_to_keep]

# 3️⃣ Split features and target
X = df.drop('approved', axis=1)
y = df['approved']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 4️⃣ Preprocessing pipelines
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

# 5️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Build complete pipeline with logistic regression
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 7️⃣ Train the model
model.fit(X_train, y_train)

# 8️⃣ Evaluate
y_pred = model.predict(X_test)
print("\nModel Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Optional: show heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 9️⃣ Take user input for prediction
print("\nPredict Loan Approval for a new applicant:")
applicant_name = input("Applicant Name: ")
gender = input("Gender (M/F): ")
age = int(input("Age: "))
city = input("City: ")
employment_type = input("Employment Type (Salaried/Self-Employed): ")
monthly_income_pkr = float(input("Monthly Income (PKR): "))
loan_amount_pkr = float(input("Loan Amount (PKR): "))
loan_tenure_months = int(input("Loan Tenure (Months): "))
existing_loans = int(input("Existing Loans: "))
default_history = int(input("Default History (0/1): "))
has_credit_card = int(input("Has Credit Card (0/1): "))

input_data = pd.DataFrame({
    'applicant_name': [applicant_name],
    'gender': [gender],
    'age': [age],
    'city': [city],
    'employment_type': [employment_type],
    'monthly_income_pkr': [monthly_income_pkr],
    'loan_amount_pkr': [loan_amount_pkr],
    'loan_tenure_months': [loan_tenure_months],
    'existing_loans': [existing_loans],
    'default_history': [default_history],
    'has_credit_card': [has_credit_card]
})

# 10️⃣ Predict
prediction = model.predict(input_data)[0]
result = "Approved ✅" if prediction == 1 else "Rejected ❌"
print(f"\nLoan Status for {applicant_name}: {result}")

model_filename = "loan_approval_model.pkl"
joblib.dump(model, model_filename)

print(f"\nModel saved successfully as '{model_filename}'")

