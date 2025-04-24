import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("german_credit_dataset.csv")

# Mapping dictionaries for friendly labels
feature_mappings = {
    'purpose': {
        'A40': 'Car (new)', 'A41': 'Car (used)', 'A42': 'Furniture/Equipment', 'A43': 'Radio/TV',
        'A44': 'Domestic Appliances', 'A45': 'Repairs', 'A46': 'Education',
        'A47': 'Vacation', 'A48': 'Retraining', 'A49': 'Business', 'A410': 'Others'
    },
    'housing': {
        'A151': 'Own', 'A152': 'Rent', 'A153': 'Free'
    },
    'savings_account': {
        'A61': '< 100', 'A62': '100 - 500', 'A63': '500 - 1000',
        'A64': '>= 1000', 'A65': 'Unknown'
    },
    'checking_account': {
        'A11': '< 0', 'A12': '0 <= x < 200', 'A13': '>= 200', 'A14': 'None'
    },
    'employment': {
        'A71': 'Unemployed', 'A72': '< 1 year', 'A73': '1 <= x < 4 years',
        'A74': '4 <= x < 7 years', 'A75': '>= 7 years'
    },
    'job': {
        'A171': 'Unskilled (non-resident)', 'A172': 'Unskilled (resident)',
        'A173': 'Skilled', 'A174': 'Highly Skilled'
    }
}

# Reverse mappings
reverse_mappings = {feature: {v: k for k, v in mapping.items()} for feature, mapping in feature_mappings.items()}

# Preprocessing
st.title("Credit Risk Prediction App")
st.subheader("Dataset Preview")
st.write(df.head())

X = df.drop(columns=['class'])
y = df['class'].map({'good': 1, 'bad': 0})

# One-hot encode original X
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.json(classification_report(y_test, y_pred, output_dict=True))

# Input form
st.subheader("Make a Prediction")
input_data = {}

for col in X.columns:
    if X[col].dtype == 'object':
        if col in feature_mappings:
            options = list(feature_mappings[col].values())
            selected = st.selectbox(f"{col.replace('_', ' ').capitalize()}", options)
            input_data[col] = reverse_mappings[col][selected]
        else:
            options = sorted(X[col].unique())
            input_data[col] = st.selectbox(f"{col.replace('_', ' ').capitalize()}", options)
    else:
        default_val = float(X[col].median())
        input_data[col] = st.number_input(f"{col.replace('_', ' ').capitalize()}", value=default_val)

# Prediction
if st.button("Predict Credit Risk"):
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    prediction = model.predict(input_df_encoded)[0]
    st.success("Prediction: Good Credit Risk" if prediction == 1 else "Prediction: Bad Credit Risk")

    # Optional debug
    st.write("Input DataFrame:")
    st.write(input_df)
    st.write("Encoded DataFrame:")
    st.write(input_df_encoded)
