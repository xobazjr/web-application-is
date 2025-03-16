import streamlit as st

def show(video_feed_url):
    st.title("📷 ตรวจจับอารมณ์แบบเรียลไทม์")

    if video_feed_url:
        st.image(video_feed_url)  # ✅ แสดงวิดีโอจาก Flask `/video_feed`
    else:
        st.error("⚠️ ไม่สามารถโหลดวิดีโอจาก Flask ได้")