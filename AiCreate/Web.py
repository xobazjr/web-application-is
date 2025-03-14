import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.rc("font", family="Tahoma") #ภาษาไทยไม่รองรับจึงต้องใส่

# 🔹 โหลดโมเดลและ Scaler ที่ฝึกไว้
try:
    model = keras.models.load_model("trained_model.h5", custom_objects={"mse": keras.losses.MeanSquaredError()})
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("🚨 ไม่พบไฟล์โมเดล `trained_model.h5` หรือ `scaler.pkl` กรุณาฝึกโมเดลก่อน!")
    st.stop()

# 🔹 ตั้งค่า UI ของเว็บ
st.title("พยากรณ์รายได้เฉลี่ยของครัวเรือนไทย")
st.write("กรอกปีที่ต้องการพยากรณ์ แล้วโมเดลจะคำนวณให้คุณ!")

# 🔹 รับค่าปีที่ผู้ใช้ต้องการพยากรณ์
selected_year = st.number_input("กรุณากรอกปีที่ต้องการพยากรณ์:", min_value=2025, max_value=2100, step=1, value=2030)

# 🔹 ตรวจสอบว่า Scaler ใช้งานได้หรือไม่
try:
    future_year_scaled = scaler.transform([[selected_year]])
except ValueError:
    st.error("มีข้อผิดพลาดในการแปลงค่าปี กรุณาตรวจสอบ `scaler.pkl`")
    st.stop()

# 🔹 พยากรณ์รายได้
predicted_income_scaled = model.predict(future_year_scaled)
predicted_income = scaler.inverse_transform(predicted_income_scaled)[0][0]

# 🔹 แสดงผลลัพธ์
st.subheader(f"คาดการณ์รายได้ในปี {selected_year}: {predicted_income:,.2f} บาท")

# 🔹 พล็อตกราฟแนวโน้มรายได้
st.subheader("กราฟแนวโน้มรายได้")
future_years = np.arange(2025, selected_year + 1)
future_years_scaled = scaler.transform(future_years.reshape(-1, 1))
predicted_incomes_scaled = model.predict(future_years_scaled)
predicted_incomes = scaler.inverse_transform(predicted_incomes_scaled)

# 🔹 สร้างกราฟ
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(future_years, predicted_incomes, marker='o', linestyle='-', color='b', label='คาดการณ์รายได้')
ax.set_xlabel("ปี")
ax.set_ylabel("รายได้เฉลี่ย (บาท)")
ax.set_title("แนวโน้มรายได้ของครัวเรือนในไทย")
ax.legend()
ax.grid(True)

# 🔹 แสดงกราฟบน Streamlit
st.pyplot(fig)
