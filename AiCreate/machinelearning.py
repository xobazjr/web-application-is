import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Input Features
with st.expander("Input Features"):
    numeric_cols = ['UTMMAP1', 'UTMMAP3', 'UTMMAP4']  # เอา UTMMAP2 และ UTMSCALE ออก
    
    min_values = df[numeric_cols].min().astype(int)
    max_values = df[numeric_cols].max().astype(int)
    median_values = df[numeric_cols].median().astype(int)

    UTMMAP1 = st.slider("หมายเลขระหว่างภูมิประเทศ", min_values['UTMMAP1'], max_values['UTMMAP1'], median_values['UTMMAP1'])
    UTMMAP3 = st.slider("หมายเลขระวาง UTM", min_values['UTMMAP3'], max_values['UTMMAP3'], median_values['UTMMAP3'])
    UTMMAP4 = st.slider("หมายเลขแผ่น", min_values['UTMMAP4'], max_values['UTMMAP4'], median_values['UTMMAP4'])

# Train Model
@st.cache_resource
def train_model():
    X = df[['UTMMAP1', 'UTMMAP3', 'UTMMAP4']]  # เอา UTMMAP2 และ UTMSCALE ออก
    y = df['EVAPRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model()

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Model Performance
with st.expander("Model Performance"):
    st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f} Baht")
    st.write(f"**Mean Squared Error (MSE):** {mse:,.2f} Baht²")

# Predict Price

with st.expander("Predict Price"):
    if st.button("Predict"):
        input_data = np.array([[UTMMAP1, UTMMAP3, UTMMAP4]])
        prediction = model.predict(input_data)[0]
    
        # Show as Pie Chart
        fig, ax = plt.subplots()
        ax.pie([prediction, median_values['UTMMAP1']], labels=['Predicted Price', 'Median Price'], autopct='%1.1f%%', colors=['red', 'blue'])
        ax.set_title("Predicted vs Median Land Price")
        st.pyplot(fig)
    
        st.success(f"**Estimated Land Price:** {prediction:,.2f} Baht per sq. wah")