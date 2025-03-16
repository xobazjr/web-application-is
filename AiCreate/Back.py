from flask import Flask, Response, render_template, jsonify
import cv2
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# 🔹 ตั้งค่า Google Drive File ID ของโมเดล
GDRIVE_FILE_ID = "1d2UdtGOP-R0Hdg3vatxTWVH2tNyYtfo9"
MODEL_PATH = "NNmodel.h5"

# 🔹 ถ้าไม่มีไฟล์โมเดล ให้ดาวน์โหลดจาก Google Drive
if not os.path.exists(MODEL_PATH):
    print("📥 กำลังดาวน์โหลดโมเดลจาก Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", MODEL_PATH, quiet=False)

# 🔹 โหลดโมเดล AI
model = load_model(MODEL_PATH)
print("✅ โหลดโมเดลสำเร็จ!")

# 🔹 โหลดตัวตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 🔹 Map label index เป็นชื่อคลาส
class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}

# 🔹 ฟังก์ชันสตรีมวิดีโอแบบเรียลไทม์
def generate_frames():
    cap = cv2.VideoCapture(0)  # ใช้กล้องเว็บแคม
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # 📌 แปลงเป็นขาวดำและตรวจจับใบหน้า
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        emotion = "Unknown"
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            face_array = image.img_to_array(face) / 255.0
            face_array = np.expand_dims(face_array, axis=0)

            # 🔹 ทำนายอารมณ์
            prediction = model.predict(face_array)
            predicted_class = np.argmax(prediction)
            emotion = class_labels.get(predicted_class, "Unknown")

            # วาดกรอบและแสดงอารมณ์
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 🔹 แปลงภาพเป็น JPEG และส่งไปที่เบราว์เซอร์
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return jsonify({"message": "Flask AI Server Running!", "video_feed_url": "/video_feed"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # เปลี่ยน port เป็น 5001 เพื่อใช้กับ Streamlit