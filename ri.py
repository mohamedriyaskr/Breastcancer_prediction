import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Unnamed: 32'], errors='ignore')
    columns_to_use = [
        'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean'
    ]
    return data[columns_to_use]

def correlation_heatmap(data):
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig

def eda(data):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("### Summary Statistics")
    st.write(data.describe())
    st.write("### Correlation Heatmap")
    st.pyplot(correlation_heatmap(data))
    st.write("### Histograms")
    for column in data.columns:
        if column != 'diagnosis':
            st.write(f"#### Histogram of {column}")
            fig, ax = plt.subplots()
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)
    st.write("### Boxplots")
    for column in data.columns:
        if column != 'diagnosis':
            st.write(f"#### Boxplot of {column}")
            fig, ax = plt.subplots()
            sns.boxplot(x=data[column], ax=ax)
            st.pyplot(fig)

def train_models(data):
    X = data.drop(columns=['diagnosis'])
    y = LabelEncoder().fit_transform(data['diagnosis'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    models = {
        "SVM": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "Decision Tree": DecisionTreeClassifier(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    return results, scaler, best_model, best_model_name, X_train, y_train

st.title("🔬 Breast Cancer Diagnosis & Prediction System")
st.markdown("Upload your breast cancer report in CSV format or enter data manually to predict the diagnosis.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

data_loaded = False
if uploaded_file:
    data = load_data(uploaded_file)
    st.write("Uploaded Dataset Preview:")
    st.write(data.head())
    data_loaded = True

    if st.button("View EDA"):
        eda(data)

if data_loaded:
    if st.button("Train Models"):
        results, scaler, best_model, best_model_name, X_train, y_train = train_models(data)
        st.session_state.best_model = best_model
        st.session_state.scaler = scaler
        st.session_state.best_model_name = best_model_name
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.success(f"Best performing model: {best_model_name} with accuracy {results[best_model_name] * 100:.2f}%")
        st.bar_chart(results)

if 'best_model' in st.session_state:
    st.subheader("Batch Prediction on Full Dataset")
    predict_data = st.session_state.scaler.transform(data.drop(columns=['diagnosis']))
    batch_predictions = st.session_state.best_model.predict(predict_data)
    data['Prediction'] = ['Malignant' if p == 1 else 'Benign' for p in batch_predictions]
    st.write("### Predictions for Full Dataset")
    st.write(data)
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", data=csv, file_name="batch_predictions.csv", mime='text/csv')

    st.subheader("Single Entry Prediction")
    feature_columns = [
        'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'smoothness_mean', 'compactness_mean',
        'concavity_mean', 'concave points_mean'
    ]
    input_features = []
    with st.form("manual_input_form"):
        for feature in feature_columns:
            value = st.number_input(f"{feature}", value=0.0)
            input_features.append(value)
        predict_button = st.form_submit_button("Predict Diagnosis")
   
    if predict_button:
        scaled_input = st.session_state.scaler.transform([input_features])
        prediction = st.session_state.best_model.predict(scaled_input)
        diagnosis = "Malignant" if prediction[0] == 1 else "Benign"
        st.success(f"Predicted Diagnosis: {diagnosis}")
       
        report = pd.DataFrame({"Feature": feature_columns, "Value": input_features})
        report.loc[len(report)] = ["Prediction", diagnosis]
        csv = report.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report", data=csv, file_name="diagnosis_report.csv", mime='text/csv')