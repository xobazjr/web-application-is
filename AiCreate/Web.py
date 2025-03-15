import streamlit as st
import av
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

def show():
    # โหลดโมเดล
    MODEL_PATH = "assets/NNmodel.h5"

    @st.cache_resource
    def load_emotion_model():
        try:
            model = load_model(MODEL_PATH)
            st.success("✅ โหลดโมเดลสำเร็จ!")
            return model
        except Exception as e:
            st.error(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
            return None

    model = load_emotion_model()

    # คลาสสำหรับประมวลผลวิดีโอจากกล้อง
    class EmotionVideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}
            self.model = model
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        def predict_emotion(self, face):
            face = cv2.resize(face, (200, 200))
            face_array = image.img_to_array(face) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = self.model.predict(face_array)
            predicted_class = np.argmax(prediction)
            return self.class_labels[predicted_class]

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
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