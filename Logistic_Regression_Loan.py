import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")
st.title("🏦 Loan Approval Prediction App")

# 1️⃣ Upload Dataset
st.subheader("Upload CSV Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # 2️⃣ Keep relevant columns
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
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # 5️⃣ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6️⃣ Build complete pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # 7️⃣ Train the model
    model.fit(X_train, y_train)

    # 8️⃣ Evaluate
    y_pred = model.predict(X_test)
    st.subheader("Model Performance")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # 9️⃣ User input for prediction
    st.subheader("Predict Loan Approval for a New Applicant")
    with st.form("loan_form"):
        applicant_name = st.text_input("Applicant Name")
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        city = st.text_input("City")
        employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
        monthly_income_pkr = st.number_input("Monthly Income (PKR)", min_value=0.0, value=50000.0)
        loan_amount_pkr = st.number_input("Loan Amount (PKR)", min_value=0.0, value=100000.0)
        loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, value=12)
        existing_loans = st.number_input("Existing Loans", min_value=0, value=0)
        default_history = st.selectbox("Default History", [0, 1])
        has_credit_card = st.selectbox("Has Credit Card", [0, 1])

        submitted = st.form_submit_button("Predict")
        if submitted:
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

            prediction = model.predict(input_data)[0]
            result = "Approved ✅" if prediction == 1 else "Rejected ❌"
            st.success(f"Loan Status for {applicant_name}: {result}")

    # 10️⃣ Save the trained model
    model_filename = "loan_approval_model.pkl"
    joblib.dump(model, model_filename)
    st.info(f"Trained model saved as '{model_filename}'")
