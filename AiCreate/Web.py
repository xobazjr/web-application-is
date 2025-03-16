import streamlit as st
import av
import cv2
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

<<<<<<< HEAD
<<<<<<< HEAD
def show(video_feed_url):
    st.title("📷 ตรวจจับอารมณ์แบบเรียลไทม์")

    if video_feed_url:
        st.image(video_feed_url)  # ✅ แสดงวิดีโอจาก Flask `/video_feed`
    else:
        st.error("⚠️ ไม่สามารถโหลดวิดีโอจาก Flask ได้")
=======
def show():
    # 🔹 ตั้งค่า Google Drive File ID ของโมเดล
    GDRIVE_FILE_ID = "1d2UdtGOP-R0Hdg3vatxTWVH2tNyYtfo9"
    MODEL_PATH = "NNmodel.h5"

=======
def show():
    # 🔹 ตั้งค่า Google Drive File ID ของโมเดล
    GDRIVE_FILE_ID = "1d2UdtGOP-R0Hdg3vatxTWVH2tNyYtfo9"
    MODEL_PATH = "NNmodel.h5"

>>>>>>> parent of d1bccf4 (Update camera)
    # 🔹 ถ้าไม่มีไฟล์โมเดล ให้ดาวน์โหลดจาก Google Drive
    if not os.path.exists(MODEL_PATH):
        st.info("กำลังดาวน์โหลดโมเดลจาก Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

    # 🔹 โหลดโมเดล
    model = load_model(MODEL_PATH)
    st.success("โหลดโมเดลสำเร็จ!")

    # 🔹 Map label index เป็นชื่อคลาส
    class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}

    # 🔹 ตั้งค่า WebRTC Signaling Server
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # 🔹 ตัวแปลงวิดีโอสำหรับ Streamlit WebRTC
    class EmotionDetector(VideoTransformerBase):
        def __init__(self):
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face_array = image.img_to_array(face) / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                
                prediction = model.predict(face_array)
                predicted_class = np.argmax(prediction)
                predicted_label = class_labels[predicted_class]
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return img

    # 🔹 ส่วน UI ของ Streamlit
    st.title("ตรวจจับอารมณ์จากใบหน้าแบบเรียลไทม์")
    st.write("เปิดกล้องและให้ใบหน้าอยู่กลางเฟรมเพื่อดูผลลัพธ์!")

<<<<<<< HEAD
    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector, rtc_configuration=rtc_config)
>>>>>>> parent of 96cd212 (Revert "Update Web.py")
=======
    webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector, rtc_configuration=rtc_config)
>>>>>>> parent of d1bccf4 (Update camera)
