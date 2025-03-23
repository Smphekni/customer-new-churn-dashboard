# Final Version: newchurnnew_dash.py (Functional API Refactor with Full Dashboard + Lazy SHAP)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import shap
import joblib

@st.cache_data
def load_data():
    np.random.seed(42)
    num_samples = 1000
    customer_ids = np.arange(1, num_samples + 1)
    monthly_charges = np.round(np.random.uniform(20, 120, num_samples), 2)
    tenure = np.random.randint(1, 72, num_samples)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(0, 50, num_samples), 2)
    contract_types = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], num_samples)
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], num_samples)
    paperless_billing = np.random.choice([True, False], num_samples).astype(int)
    has_dependents = np.random.choice([True, False], num_samples).astype(int)
    churn = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'monthly_charges': monthly_charges,
        'tenure': tenure,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'paperless_billing': paperless_billing,
        'has_dependents': has_dependents,
        'churn': churn
    })

    df.drop_duplicates(inplace=True)
    Q1 = df['monthly_charges'].quantile(0.25)
    Q3 = df['monthly_charges'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['monthly_charges'] >= Q1 - 1.5 * IQR) & (df['monthly_charges'] <= Q3 + 1.5 * IQR)]
    expected_total = df['monthly_charges'] * df['tenure']
    df['total_charges'] = np.where(abs(df['total_charges'] - expected_total) > 50, expected_total, df['total_charges'])
    return df

def preprocess_data(df):
    num = ['monthly_charges', 'tenure', 'total_charges']
    cat = ['contract_type', 'payment_method']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([('num', num_pipe, num), ('cat', cat_pipe, cat)], remainder='passthrough')

    X = df.drop(['customer_id', 'churn'], axis=1)
    y = df['churn']
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y.values, preprocessor, num, cat

def build_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache_resource

def get_shap_summary_plot(model, X_sample, feature_names):
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        return fig
    except Exception as e:
        st.warning(f"SHAP failed: {e}")
        return None

# --- App Execution Starts Here ---

data = load_data()
X, y, preprocessor, num_features, cat_features = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Build and train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop], class_weight=class_weights, verbose=0)

# Save model and preprocessor
model.save("churn_model.keras")
joblib.dump(preprocessor, "preprocessor.pkl")

# --- Streamlit Dashboard ---
st.title("Customer Churn Prediction Dashboard")
tabs = st.tabs(["🔍 Predict Churn", "📊 Data Explorer", "📈 Evaluation Report"])

with tabs[0]:
    st.header("Predict Individual Churn")
    tenure = st.number_input("Tenure (months)", 1, 72, 24)
    monthly_charges = st.number_input("Monthly Charges", 20.0, 120.0, 60.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1200.0)
    contract = st.selectbox("Contract Type", ['Month-to-Month', 'One Year', 'Two Year'])
    payment = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    paperless = st.checkbox("Paperless Billing", value=True)
    dependents = st.checkbox("Has Dependents", value=False)

    if st.button("Predict Now"):
        new_data = pd.DataFrame([{ 'monthly_charges': monthly_charges, 'tenure': tenure, 'total_charges': total_charges,
                                    'contract_type': contract, 'payment_method': payment,
                                    'paperless_billing': int(paperless), 'has_dependents': int(dependents) }])
        loaded_model = build_model(X_train.shape[1])
        loaded_model.load_weights("churn_model.keras")
        transformed = joblib.load("preprocessor.pkl").transform(new_data)
        prediction = loaded_model.predict(transformed)
        st.success(f"Predicted Churn Probability: {prediction[0][0]:.2%}")

with tabs[1]:
    st.header("Explore the Data")
    st.dataframe(data.head())
    st.subheader("Missing Data Overview")
    st.dataframe(data.isnull().mean() * 100)

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data['churn'], ax=ax)
    st.pyplot(fig)

    col = st.selectbox("Histogram Column", data.columns)
    fig2, ax2 = plt.subplots()
    sns.histplot(data[col], kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    fig3, ax3 = plt.subplots()
    sns.heatmap(data[num_features].corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

with tabs[2]:
    st.header("Final Model Evaluation")
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, zero_division=0))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))
    st.write("ROC-AUC:", roc_auc_score(y_test, y_pred))

    st.subheader("Feature Importance with SHAP")
    try:
        ohe_cols = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_features))
        final_features = num_features + ohe_cols + ['paperless_billing', 'has_dependents']
        X_sample_df = pd.DataFrame(X_test, columns=final_features).sample(100, random_state=42)
        fig = get_shap_summary_plot(model, X_sample_df, final_features)
        if fig:
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP Visualization Failed: {e}")

    st.subheader("Baseline Models")
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)

    models = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
    for name, clf in models.items():
        preds = clf.predict(X_test)
        st.write(f"**{name} Accuracy:**", accuracy_score(y_test, preds))
        st.write(f"**{name} ROC-AUC:**", roc_auc_score(y_test, preds))



