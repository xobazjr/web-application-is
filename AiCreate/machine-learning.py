import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
@st.cache_data
def load_data():
    file_path = "/Users/xobazjr/Documents/GitHub/web-application-is/dataset/Bangkok Land Data.csv"
    df = pd.read_csv(file_path)
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

st.info('This is bangkok land data machine learing')

# Show Preview Dataset
with st.expander("Show Preview Dataset"):
    st.write(df.head())

# Data Visualisation
with st.expander("Data Visualisation"):
    st.scatter_chart(data=df, x='UTMMAP1', y='EVAPRICE', color='#1f77b4')

# Input Features
with st.expander("Input Features"):
    UTMMAP1 = st.slider("หมายเลขระหว่างภูมิประเทศ", min_value=int(df['UTMMAP1'].min()), max_value=int(df['UTMMAP1'].max()), value=int(df['UTMMAP1'].median()))
    UTMMAP2 = st.slider("หมายเลขแผ่นระวางภูมิประเทศ", min_value=int(df['UTMMAP2'].min()), max_value=int(df['UTMMAP2'].max()), value=int(df['UTMMAP2'].median()))
    UTMMAP3 = st.slider("หมายเลขระวาง UTM", min_value=int(df['UTMMAP3'].min()), max_value=int(df['UTMMAP3'].max()), value=int(df['UTMMAP3'].median()))
    UTMMAP4 = st.slider("หมายเลขแผ่น", min_value=int(df['UTMMAP4'].min()), max_value=int(df['UTMMAP4'].max()), value=int(df['UTMMAP4'].median()))
    UTMSCALE = st.slider("มาตราส่วนระวางแผนที่", min_value=int(df['UTMSCALE'].min()), max_value=int(df['UTMSCALE'].max()), value=int(df['UTMSCALE'].median()))

# Data Preparation
with st.expander("Data Preparation"):
    @st.cache_resource
    def train_model():
        X = df[['UTMMAP1', 'UTMMAP2', 'UTMMAP3', 'UTMMAP4', 'UTMSCALE']]
        y = df['EVAPRICE']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    model = train_model()

    if st.button("Predict Price"):
        input_data = np.array([[UTMMAP1, UTMMAP2, UTMMAP3, UTMMAP4, UTMSCALE]])
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated Land Price: {prediction:,.2f} Baht per sq. wah")