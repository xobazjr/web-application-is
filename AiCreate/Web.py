import streamlit as st
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

def show():
    # โหลดโมเดลเพียงครั้งเดียว
    @st.cache_resource
    def load_emotion_model():
        MODEL_PATH = "assets/NNmodel.h5"
        try:
            # ตรวจสอบว่าโมเดลมีอยู่จริงในตำแหน่งนั้น
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"❌ ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            st.success("✅ โหลดโมเดลสำเร็จ!")
            return model
        except FileNotFoundError as e:
            st.error(str(e))
            return None
        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
            return None

    # โหลด Haar Cascade เพียงครั้งเดียว
    @st.cache_resource
    def load_cascade():
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            # ตรวจสอบว่าไฟล์ Haar Cascade มีอยู่จริง
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"❌ ไม่พบไฟล์ Haar Cascade ที่ {cascade_path}")
            return cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลด Haar Cascade: {e}")
            return None

    # คลาสสำหรับประมวลผลวิดีโอจากกล้อง
    class EmotionVideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}
            self.model = load_emotion_model()  # โหลดโมเดล
            self.face_cascade = load_cascade()  # โหลด Haar Cascade

        def predict_emotion(self, face):
            face = cv2.resize(face, (200, 200))
            face_array = image.img_to_array(face) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = self.model.predict(face_array)
            predicted_class = np.argmax(prediction)
            return self.class_labels[predicted_class]

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")  # แปลงเป็น BGR สำหรับ OpenCV
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                if self.model:
                    emotion = self.predict_emotion(face)
                    # วาดกรอบและแสดงผลอารมณ์
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # UI ของ Streamlit
    st.title("📷 ตรวจจับอารมณ์จากใบหน้า")
    st.write("🔹 เปิดกล้องและดูการคาดการณ์อารมณ์แบบเรียลไทม์!")

    # เรียกใช้งาน streamlit-webrtc
    webrtc_streamer(key="face-emotion", video_transformer_factory=EmotionVideoTransformer)