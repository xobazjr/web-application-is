import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib

def show():

    matplotlib.rc("font", family="Tahoma")  # แก้ปัญหาฟอนต์ภาษาไทย

    # 🔹 โหลดโมเดลและ Scaler ที่ฝึกไว้
    try:
        model = keras.models.load_model("trained_model.keras")  # เปลี่ยนเป็นไฟล์ที่ถูกต้อง
        with open("scaler_year.pkl", "rb") as f:
            scaler_year = pickle.load(f)
        with open("scaler_income.pkl", "rb") as f:
            scaler_income = pickle.load(f)
    except (FileNotFoundError, AttributeError, IndexError) as e:
        st.error(f"🚨 เกิดข้อผิดพลาดขณะโหลดโมเดล: {str(e)} กรุณาตรวจสอบไฟล์โมเดลและ Scaler!")
        st.button("🔄 กลับไปหน้าเดิม", on_click=st.rerun)
        st.stop()

    # 🔹 ตั้งค่า UI ของเว็บ
    st.title("📊 พยากรณ์รายได้เฉลี่ยของครัวเรือนไทย")
    st.write("กรอกปีที่ต้องการพยากรณ์ c!")

    # 🔹 รับค่าปีที่ผู้ใช้ต้องการพยากรณ์
    selected_year = st.number_input("📅 กรุณากรอกปีที่ต้องการพยากรณ์:", min_value=2025, max_value=2100, step=1, value=2030)

    # 🔹 ตรวจสอบว่า Scaler ใช้งานได้หรือไม่
    try:
        future_year_df = pd.DataFrame([[selected_year]], columns=["YEAR"])  # แปลงเป็น DataFrame ก่อน
        future_year_scaled = scaler_year.transform(future_year_df)  # แปลงค่าให้ตรงกับโมเดล
    except ValueError:
        st.error("🚨 มีข้อผิดพลาดในการแปลงค่าปี กรุณาตรวจสอบ `scaler_year.pkl`")
        st.button("🔄 กลับไปหน้าเดิม", on_click=st.rerun)
        st.stop()

    # 🔹 พยากรณ์รายได้
    predicted_income_scaled = model.predict(future_year_scaled)
    predicted_income = scaler_income.inverse_transform(predicted_income_scaled)[0][0]  # แปลงกลับเป็นค่าปกติ

    st.subheader(f"📌 คาดการณ์รายได้ในปี {selected_year}: {predicted_income:,.2f} บาท")

    # 🔹 แสดงกราฟแนวโน้มรายได้
    st.subheader("📈 กราฟแนวโน้มรายได้")
    future_years = np.arange(2025, selected_year + 1).reshape(-1, 1)  # สร้างอาร์เรย์ของปี
    future_years_df = pd.DataFrame(future_years, columns=["YEAR"])  # แปลงเป็น DataFrame
    future_years_scaled = scaler_year.transform(future_years_df)  # ทำการ Normalize

    predicted_incomes_scaled = model.predict(future_years_scaled)  # ทำนายรายได้
    predicted_incomes = scaler_income.inverse_transform(predicted_incomes_scaled)  # แปลงกลับเป็นค่าปกติ

    # 🔹 พล็อตกราฟเส้น
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(future_years, predicted_incomes, marker='o', linestyle='-', color='b', label='คาดการณ์รายได้')
    ax.set_xlabel("ปี")
    ax.set_ylabel("รายได้เฉลี่ย (บาท)")
    ax.set_title("แนวโน้มรายได้ของครัวเรือนในไทย")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 🔹 พล็อตกราฟแท่ง
    st.subheader("📊 กราฟแท่งรายได้ในแต่ละปี")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(future_years.flatten(), predicted_incomes.flatten(), color='orange', alpha=0.7)
    ax.set_xlabel("ปี")
    ax.set_ylabel("รายได้เฉลี่ย (บาท)")
    ax.set_title("พยากรณ์รายได้ของครัวเรือน")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)