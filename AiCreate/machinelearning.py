import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
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

# Train Model with error tracking
@st.cache_resource
def train_model():
    X = df[['UTMMAP1', 'UTMMAP3', 'UTMMAP4']]  # เอา UTMMAP2 และ UTMSCALE ออก
    y = df['EVAPRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)  # เปลี่ยนจาก 100 เป็น 200
    
    # Error tracking lists
    mae_list = []
    mse_list = []
    
    # Train the model and track errors
    for i in range(1, 201):  # Train with 200 iterations
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        mae_list.append(mae)
        mse_list.append(mse)
    
    return model, X_test, y_test, mae_list, mse_list

model, X_test, y_test, mae_list, mse_list = train_model()

# Model Performance (Show final MAE and MSE)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

with st.expander("Model Performance"):
    st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f} Baht")
    st.write(f"**Mean Squared Error (MSE):** {mse:,.2f} Baht²")

# Train Model Button (ฝึกโมเดลใหม่)
with st.expander("Train Model"):
    if st.button("Train Model"):
        # Train the model with the selected features
        model, X_test, y_test, mae_list, mse_list = train_model()
        
        # Show model performance after training
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        st.write(f"**Model has been retrained!**")
        st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f} Baht")
        st.write(f"**Mean Squared Error (MSE):** {mse:,.2f} Baht²")
        
        # Plot MAE and MSE over the iterations
        fig, ax = plt.subplots()
        ax.plot(range(1, 201), mae_list, label='MAE', color='blue')
        ax.plot(range(1, 201), mse_list, label='MSE', color='red')
        ax.set_title("Error vs Number of Trees (Iterations)")
        ax.set_xlabel("Number of Trees (n_estimators)")
        ax.set_ylabel("Error")
        ax.legend()
        
        # Show the plot in terminal as well
        plt.show()  # This will display the graph in a GUI window (terminal)
        
        # Show the plot in Streamlit
        st.pyplot(fig)

# Input Features
with st.expander("Input Features"):
    numeric_cols = ['UTMMAP1', 'UTMMAP3', 'UTMMAP4']  # เอา UTMMAP2 และ UTMSCALE ออก
    
    min_values = df[numeric_cols].min().astype(int)
    max_values = df[numeric_cols].max().astype(int)
    median_values = df[numeric_cols].median().astype(int)

    UTMMAP1 = st.slider("หมายเลขระหว่างภูมิประเทศ", min_values['UTMMAP1'], max_values['UTMMAP1'], median_values['UTMMAP1'])
    UTMMAP3 = st.slider("หมายเลขระวาง UTM", min_values['UTMMAP3'], max_values['UTMMAP3'], median_values['UTMMAP3'])
    UTMMAP4 = st.slider("หมายเลขแผ่น", min_values['UTMMAP4'], max_values['UTMMAP4'], median_values['UTMMAP4'])

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