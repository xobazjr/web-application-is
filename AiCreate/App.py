import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

import Web
import machinelearning
import Detail
import streamlit as st

# 🔹 ตั้งค่า URL ของ Flask Server (back.py)
FLASK_URL = "http://127.0.0.1:5001"

# 🔹 ตรวจสอบว่า Flask Server ทำงานหรือไม่
try:
    response = requests.get(FLASK_URL)
    if response.status_code == 200:
        st.success("✅ เชื่อมต่อกับ Flask AI Server สำเร็จ!")
        video_feed_url = f"{FLASK_URL}/video_feed"
    else:
        st.error("❌ ไม่สามารถเชื่อมต่อกับ Flask AI Server")
        video_feed_url = None
except:
    st.error("❌ Flask Server ไม่ทำงาน")
    video_feed_url = None

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("เลือกหน้า", ["Home", "Development Approach", "Machine Learning"])

if page == "Home":
    Web.show(video_feed_url)  # ✅ ส่ง video_feed_url ให้ Web.show()
elif page == "Development Approach":
    Detail.show()
elif page == "Machine Learning":
    machinelearning.show()