import streamlit as st
import cv2
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Set page title
st.set_page_config(page_title="Face Emotion Detection")

# 🔹 ตั้งค่า Google Drive File ID ของโมเดล
GDRIVE_FILE_ID = "1d2UdtGOP-R0Hdg3vatxTWVH2tNyYtfo9"
MODEL_PATH = "NNmodel.h5"

@st.cache_resource
def load_face_model():
    # 🔹 ถ้าไม่มีไฟล์โมเดล ให้ดาวน์โหลดจาก Google Drive
    if not os.path.exists(MODEL_PATH):
        with st.spinner("กำลังดาวน์โหลดโมเดลจาก Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)
    
    # 🔹 โหลดโมเดลที่เทรนมา
    model = load_model(MODEL_PATH)
    return model

def show():
    st.title("ตรวจจับอารมณ์จากใบหน้า")
    st.write("กรุณาให้ใบหน้าอยู่กลางกล้อง แล้วดูผลการทำนาย!")
    
    # โหลดโมเดล
    with st.spinner("กำลังโหลดโมเดล..."):
        model = load_face_model()
        st.success("โหลดโมเดลสำเร็จ!")
    
    # 🔹 Map label index เป็นชื่อคลาส
    class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}
    
    # โหลด face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # สร้างตัวแปรเก็บผลการทำนายล่าสุด
    emotion_result = st.empty()
    
    # ฟังก์ชันประมวลผลแต่ละเฟรม
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # แปลงเป็น grayscale สำหรับการตรวจจับใบหน้า
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            # วาดกรอบรอบใบหน้า
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # ตัดภาพใบหน้า
            face = img[y:y+h, x:x+w]
            
            try:
                # ปรับขนาดและเตรียมข้อมูลสำหรับโมเดล
                face = cv2.resize(face, (200, 200))
                face_array = image.img_to_array(face) / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                
                # ทำนายอารมณ์
                prediction = model.predict(face_array)
                predicted_class = np.argmax(prediction)
                predicted_label = class_labels[predicted_class]
                
                # แสดงผลการทำนายบนภาพ
                cv2.putText(img, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # อัปเดตผลการทำนายในหน้า UI
                emotion_result.markdown(f"### อารมณ์ที่ตรวจพบ: *{predicted_label}*")
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # กำหนดค่า RTC configuration (STUN servers)
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # สร้าง WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if not webrtc_ctx.state.playing:
        st.info("กดปุ่ม 'Start' เพื่อเริ่มการตรวจจับอารมณ์")

if __name__ == "__main__":
    show()