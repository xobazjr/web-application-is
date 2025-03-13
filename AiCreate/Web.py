#ติดตั้งก่อนในเทอมินอล  pip install pandas streamlit


import streamlit as st #นำเข้าตัวทำเว็ป
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../processed_data.csv")  

def show():
    st.title("🏡 Welcome to the Home Page")
    st.write("นี่คือหน้าแรกของแอปพลิเคชัน")
if __name__ == "__main__":
    show() 

st.title("📊 รายได้ครัวเรือนในไทย")


try:
    df = pd.read_pickle("processed_data.pkl")  
except:
   df = pd.read_csv(file_path, encoding="utf-8-sig")


#  แสดงข้อมูล
st.write("ข้อมูลที่โหลดมาจาก NeuronTest.py:")
st.dataframe(df.head())

#  เลือกช่วงรายได้
income_range = st.selectbox("เลือกช่วงรายได้ที่ต้องการดู", 
                            [100000, 500000, 1000000, 5000000, 10000000])

# กรองข้อมูล
filtered_df = df[df["MONTHLY_INCOME"] >= income_range]

st.write(f"คนที่มีรายได้มากกว่า {income_range} บาท/เดือน")
st.dataframe(filtered_df)
