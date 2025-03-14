import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../dataset/Bangkok Land Data.csv")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    return df  

df = load_data()

# Rename columns for easier handling
df.rename(columns={
    'หมายเลขระหว่างภูมิประเทศ': 'UTMMAP1',
    'หมายเลขแผ่นระวางภูมิประเทศ': 'UTMMAP2',
    'หมายเลขระวาง UTM': 'UTMMAP3',
    'หมายเลขแผ่น': 'UTMMAP4',
    'มาตราส่วนระวางแผนที่': 'UTMSCALE',
    'เลขที่ดิน': 'LAND_NO',
    'ราคาประเมิน (บาท/ตร.ว.)': 'EVAPRICE'
}, inplace=True)

st.title('🤖 Machine Learning')
st.info('This is a machine learning model to estimate land prices in Bangkok.')

# Show Preview Dataset
with st.expander("Show Preview Dataset"):
    st.write(df.head())

# Data Visualisation
with st.expander("Data Visualisation"):
    st.scatter_chart(data=df, x='UTMMAP1', y='EVAPRICE', color='#1f77b4')

# Train Model with error tracking and hyperparameter tuning
@st.cache_resource
def train_model():
    # Feature selection and scaling
    X = df[['UTMMAP1', 'UTMMAP3', 'UTMMAP4']]  # Removed UTMMAP2 and UTMSCALE
    y = df['EVAPRICE']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning with GridSearchCV
    param_grid = {
        'n_estimators': [100, 500, 1000],  # Increased n_estimators to 1000
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # Prediction and error calculation
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model, scaler, X_test, y_test, y_pred, mae, mse, r2, grid_search.best_params_

# Automatically train the model when the page is loaded
with st.expander("Train Model"):
    model, scaler, X_test, y_test, y_pred, mae, mse, r2, best_params = train_model()

    # Display Model Performance
    st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f} Baht")
    st.write(f"**Mean Squared Error (MSE):** {mse:,.2f} Baht²")
    st.write(f"**R² Score:** {r2:,.2f}")

    # Plotting Feature Importance
    feature_importance = model.feature_importances_
    feature_names = ['UTMMAP1', 'UTMMAP3', 'UTMMAP4']

    fig, ax = plt.subplots()
    ax.barh(feature_names, feature_importance)
    ax.set_title("Feature Importance")
    
    # Save the figure and display it in Streamlit
    plot_path = 'feature_importance_plot.png'
    fig.savefig(plot_path)
    st.pyplot(fig)
    
    # Also print the plot file path to the terminal (simulating plt.show() in terminal)
    print(f"Feature Importance plot saved to {plot_path}")
    
    # Plotting Residual Plot
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.hlines(0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='dashed')
    ax.set_title("Residual Plot")
    ax.set_xlabel("Predicted Prices")
    ax.set_ylabel("Residuals")

    # Save the figure and display it in Streamlit
    residual_plot_path = 'residual_plot.png'
    fig.savefig(residual_plot_path)
    st.pyplot(fig)
    
    # Also print the plot file path to the terminal
    print(f"Residual plot saved to {residual_plot_path}")

# Input Features
with st.expander("Input Features"):
    numeric_cols = ['UTMMAP1', 'UTMMAP3', 'UTMMAP4']  # Removed UTMMAP2 and UTMSCALE
    min_values = df[numeric_cols].min().astype(int)
    max_values = df[numeric_cols].max().astype(int)
    median_values = df[numeric_cols].median().astype(int)

    UTMMAP1 = st.slider("หมายเลขระหว่างภูมิประเทศ", min_values['UTMMAP1'], max_values['UTMMAP1'], median_values['UTMMAP1'])
    UTMMAP3 = st.slider("หมายเลขระวาง UTM", min_values['UTMMAP3'], max_values['UTMMAP3'], median_values['UTMMAP3'])
    UTMMAP4 = st.slider("หมายเลขแผ่น", min_values['UTMMAP4'], max_values['UTMMAP4'], median_values['UTMMAP4'])

with st.expander("Predict Price"):
    # Automatically predict the price when input features are changed
    input_data = np.array([[UTMMAP1, UTMMAP3, UTMMAP4]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)[0]

    # Bar Chart for Predicted vs Median Price
    fig, ax = plt.subplots()
    ax.bar(['Predicted Price', 'Median Price'], [prediction, median_values['UTMMAP1']], color=['red', 'blue'])
    ax.set_title("Predicted vs Median Land Price")
    st.pyplot(fig)

    st.success(f"**Estimated Land Price:** {prediction:,.2f} Baht per sq. wah")