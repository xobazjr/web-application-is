import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

import Web
import machinelearning

import streamlit as st

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("เลือกหน้า", ["Home", "Machine Learning"])

if page == "Home":
    Web.show()  # เรียกใช้ฟังก์ชันแสดงหน้า Web
elif page == "Machine Learning":
    machinelearning.show()  # เรียกใช้ฟังก์ชันแสดงหน้า Machine Learning
