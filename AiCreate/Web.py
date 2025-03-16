import streamlit as st
import cv2
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time

def show():
    # 🔹 ตั้งค่า Google Drive File ID ของโมเดล
    GDRIVE_FILE_ID = "1d2UdtGOP-R0Hdg3vatxTWVH2tNyYtfo9"
    MODEL_PATH = "NNmodel.h5"

    # 🔹 ถ้าไม่มีไฟล์โมเดล ให้ดาวน์โหลดจาก Google Drive
    if not os.path.exists(MODEL_PATH):
        st.info("กำลังดาวน์โหลดโมเดลจาก Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

    # 🔹 โหลดโมเดลที่เทรนมา
    model = load_model(MODEL_PATH)
    st.success("โหลดโมเดลสำเร็จ!")

    # 🔹 Map label index เป็นชื่อคลาส
    class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}

    # 🔹 ฟังก์ชันประมวลผลและทำนายอารมณ์จากภาพ
    def predict_emotion(frame):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            return None, None

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            face_array = image.img_to_array(face) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            prediction = model.predict(face_array)
            predicted_class = np.argmax(prediction)
            predicted_label = class_labels[predicted_class]

            return predicted_label, (x, y, w, h)

    # 🔹 ส่วน UI ของ Streamlit
    st.title("ตรวจจับอารมณ์จากใบหน้า")

    # 🔹 เปิดกล้อง
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("ไม่สามารถเปิดกล้องได้! ตรวจสอบการเชื่อมต่อ")
        st.stop()

    frame_placeholder = st.empty()
    prediction_placeholder = st.empty()

    st.write("กรุณาให้ใบหน้าอยู่กลางกล้อง แล้วดูผลการทำนาย!")

    # 🔹 วนลูปอ่านเฟรมจากกล้อง
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.error("ไม่สามารถอ่านข้อมูลจากกล้องได้!")
            break

        frame = cv2.flip(frame, 1)
        emotion, face_coords = predict_emotion(frame)

        if face_coords:
            x, y, w, h = face_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        if emotion:
            prediction_placeholder.write(f"### อารมณ์ที่ตรวจพบ: *{emotion}*")

        time.sleep(0)  # ลดความเร็วในการแสดงผล

    video_capture.release()
    cv2.destroyAllWindows()