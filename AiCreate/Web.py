# ติดตั้งก่อนในเทอมินอล  pip install pandas streamlit matplotlib seaborn

import streamlit as st  # นำเข้าตัวทำเว็บ
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ระบุพาธของไฟล์ข้อมูล
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../processed_data.csv")  

# แสดงหัวข้อ
st.title("📊 รายได้ครัวเรือนในไทย")

# ลองโหลดข้อมูลจากไฟล์ Pickle ถ้าไม่มีให้โหลดจาก CSV
try:
    df = pd.read_pickle("processed_data.pkl")  
except:
    df = pd.read_csv(file_path, encoding="utf-8-sig")

# ตรวจสอบว่าคอลัมน์ที่ต้องการแสดงมีอยู่จริงหรือไม่
if "MONTHLY_INCOME" not in df.columns or "REGION" not in df.columns or "YEAR" not in df.columns:
    st.error("🚨 ไม่พบคอลัมน์ที่ต้องใช้ ('MONTHLY_INCOME', 'REGION', 'YEAR') ในไฟล์ข้อมูล กรุณาตรวจสอบ")
    st.write("📝 คอลัมน์ที่พบในไฟล์:", df.columns.tolist())  # แสดงคอลัมน์ที่มีอยู่
    st.stop()

# กำหนดตัวเลือกช่วงรายได้
income_options = [10000, 20000, 30000, 40000, 50000]

# ถ้าไม่มีค่า income_range ใน session_state หรือค่าที่มีอยู่ไม่ตรงกับตัวเลือก ให้ใช้ค่าแรกเป็นค่าเริ่มต้น
if "income_range" not in st.session_state or st.session_state["income_range"] not in income_options:
    st.session_state["income_range"] = income_options[0]  # ใช้ค่าตัวแรกเป็นค่าเริ่มต้น

# สร้าง dropdown ให้ผู้ใช้เลือกช่วงรายได้
income_range = st.selectbox("เลือกช่วงรายได้ที่ต้องการดู", 
                            income_options, 
                            index=income_options.index(st.session_state["income_range"]))

# อัปเดตค่าใน session_state
st.session_state["income_range"] = income_range

# ลบแถวที่มี NaN ออกจาก DataFrame
df = df.dropna(subset=["MONTHLY_INCOME", "REGION", "YEAR"])

# กรองข้อมูลเฉพาะแถวที่มีรายได้ >= income_range และแสดงเฉพาะ "REGION", "MONTHLY_INCOME", "YEAR"
filtered_df = df[df["MONTHLY_INCOME"] >= income_range][["REGION", "MONTHLY_INCOME", "YEAR"]]

# ถ้าไม่มีข้อมูลที่ตรงกับช่วงรายได้ที่เลือก ให้แสดงข้อความเตือน
if filtered_df.empty:
    st.error("🚨 ไม่มีข้อมูลที่ตรงกับเงื่อนไขที่เลือก กรุณาลองเลือกช่วงรายได้อื่น")
else:
    st.write(f"📌 รายได้ครัวเรือนที่มากกว่า {income_range:,} บาท/เดือน พร้อมพื้นที่ที่อยู่")
    st.dataframe(filtered_df)

    #  **Scatter Plot แสดงรายได้ครัวเรือนในแต่ละปี** 
    st.subheader("📊 รายได้ครัวเรือนในแต่ละปี")
    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(filtered_df["YEAR"], filtered_df["MONTHLY_INCOME"], 
                         c=filtered_df["MONTHLY_INCOME"], cmap="coolwarm", alpha=0.7)
    ax.set_xlabel("ในแต่ละปี")
    ax.set_ylabel("รายได้ต่อครัวเรือนของไทย")
    ax.set_title("รายได้ต่อครัวเรือนในประเทศไทย")
    plt.colorbar(scatter, label="ระดับรายได้")
    st.pyplot(fig)

    #  **Bar Chart แสดงจังหวัดที่มีรายได้เฉลี่ยสูงสุด** 
    income_by_region = filtered_df.groupby("REGION")["MONTHLY_INCOME"].mean().reset_index()
    income_by_region = income_by_region.sort_values(by="MONTHLY_INCOME", ascending=False)  # เรียงลำดับจากมากไปน้อย

    st.subheader("📊 จังหวัดที่มีรายได้เฉลี่ยสูงสุด")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="MONTHLY_INCOME", y="REGION", data=income_by_region, palette="coolwarm", ax=ax)
    ax.set_xlabel("รายได้เฉลี่ย (บาท)")
    ax.set_ylabel("จังหวัด")
    ax.set_title("รายได้เฉลี่ยสูงสุดในแต่ละจังหวัด")
    st.pyplot(fig)

    # **Bar Chart แสดงจำนวนครัวเรือนในแต่ละจังหวัด** 
    count_by_region = filtered_df["REGION"].value_counts().reset_index()
    count_by_region.columns = ["REGION", "HOUSEHOLD_COUNT"]
    count_by_region = count_by_region.sort_values(by="HOUSEHOLD_COUNT", ascending=False)  # เรียงลำดับจากมากไปน้อย

    st.subheader("📊 จำนวนครัวเรือนในแต่ละจังหวัด")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="HOUSEHOLD_COUNT", y="REGION", data=count_by_region, palette="viridis", ax=ax)
    ax.set_xlabel("จำนวนครัวเรือน")
    ax.set_ylabel("จังหวัด")
    ax.set_title("จำนวนครัวเรือนในแต่ละจังหวัด")
    st.pyplot(fig)
