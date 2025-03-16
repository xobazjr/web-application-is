import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def show():
    # Set page title
    st.title("ตรวจจับอารมณ์จากใบหน้า")
    
    # โหลดโมเดลที่เทรนมา
    @st.cache_resource
    def load_face_model():
        try:
            MODEL_PATH = "/Users/xobazjr/Documents/GitHub/web-application-is/assets/NNmodel.h5"
            return load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
            return None
    
    model = load_face_model()
    
    if model is None:
        st.error("ไม่พบไฟล์โมเดล กรุณาตรวจสอบ!")
        st.stop()
    else:
        st.success("โหลดโมเดลสำเร็จ! ฟาดแซ้โดยพริด๊ม")
    
    # Map label index เป็นชื่อคลาส
    class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}
    
    # โหลด Haar Cascade classifier สำหรับการตรวจจับใบหน้า
    @st.cache_resource
    def load_face_cascade():
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    face_cascade = load_face_cascade()
    
    # สร้างตัวแปรสำหรับเก็บการทำนายล่าสุด
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
    # Placeholder สำหรับแสดงผลการทำนาย
    prediction_placeholder = st.empty()
    
    # กำหนดค่า RTC Configuration สำหรับ WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # ฟังก์ชันสำหรับประมวลผลแต่ละเฟรม
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # แปลงภาพเป็นสีเทาสำหรับการตรวจจับใบหน้า
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        # ถ้าตรวจพบใบหน้า
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # วาดกรอบรอบใบหน้า
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # ตัดภาพใบหน้าและประมวลผล
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face_array = image.img_to_array(face) / 255.0
                face_array = np.expand_dims(face_array, axis=0)
                
                # ทำนายอารมณ์
                prediction = model.predict(face_array)
                predicted_class = np.argmax(prediction)
                predicted_label = class_labels[predicted_class]
                
                # แสดงข้อความบนภาพ
                cv2.putText(img, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # อัปเดตการทำนายล่าสุด
                st.session_state.last_prediction = predicted_label
        
        # อัปเดตการแสดงผลการทำนาย
        if st.session_state.last_prediction:
            prediction_placeholder.write(f"### อารมณ์ที่ตรวจพบ: **{st.session_state.last_prediction}**")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # สร้าง WebRTC streamer
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # คำแนะนำสำหรับผู้ใช้
    st.write("กรุณาให้ใบหน้าอยู่กลางกล้อง แล้วดูผลการทำนาย!")
    
    # แสดงสถานะการเชื่อมต่อ
    if webrtc_ctx.state.playing:
        st.write("กำลังเชื่อมต่อกับกล้อง...")
    
if __name__ == "__main__":
    show()