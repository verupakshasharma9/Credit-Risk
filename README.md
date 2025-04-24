Credit Risk Prediction App

This is a Streamlit web application that predicts credit risk (Good or Bad) using the German Credit dataset. The model is trained with a Random Forest classifier and allows users to interactively input customer information to receive a prediction.

Features

- Clean and simple web interface using Streamlit
- Categorical input values shown with user-friendly labels
- One-hot encoding behind the scenes for compatibility with the trained model
- Displays model accuracy and performance metrics
- Handles both numeric and categorical input types properly

Dataset

This app uses the [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) from the UCI Machine Learning Repository. Ensure that the CSV file `german_credit_dataset.csv` is placed in the root directory of the app.

 ML Model

- Model: Random Forest Classifier
- Library: Scikit-learn
- Preprocessing: One-hot encoding for categorical variables
- Train/Test Split: 80% Train, 20% Test
