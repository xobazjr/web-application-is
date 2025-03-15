# web-application-is


# ก่อนนำไฟล์ไปใช้ควรติดตั้งตามคำสั่งนี้ใน เทอมินอลก่อน

pip install tensorflow keras torch torchvision torchaudio scikit-learn pandas numpy matplotlib seaborn streamlit fastapi uvicorn opencv-python requests pillow tqdm jupyter notebook seaborn

pip install opencv-python
pip install opencv-python-headless
pip install opencv-python
pip install streamlit opencv-python tensorflow keras numpy matplotlib

# การเข้าทำงานเปิดเว้ป
    cd Aicreate
    python NeuronTest.py
    python machinelearning.py
    python iwanttodie.py
    streamlit run Web.py
    streamlit run machinelearning.py
    streamlit run app.py

# แหล่งที่มาอ้างองการทำงาน

    การใช้tensorflow 
        1.https://nakarinstk.medium.com/%E0%B9%80%E0%B8%A3%E0%B8%B4%E0%B9%88%E0%B8%A1%E0%B8%95%E0%B9%89%E0%B8%99-deep-learning-%E0%B8%94%E0%B9%89%E0%B8%A7%E0%B8%A2-keras-b13edc47b1b3

    ขอบตุณยูทรูปจากช่อง When Maths Meet Coding ที่ทำให้เรารอด
    2.https://youtu.be/uqomO_BZ44g?si=i9o4Ktd6Q2vlod6L


# ค่าการเทรน 

    Loss เป็นค่าที่ต้องการให้ต่ำที่สุด
    Loss ต่ำแปลว่าโมเดลสามารถ พยากรณ์ได้ใกล้เคียงค่าจริง มาก
    Loss สูง → โมเดลยังเรียนรู้ไม่ดี หรือมีความคลาดเคลื่อนเยอะ

    หมายความว่า โมเดลสามารถทำนายชุด Training ได้ถูกต้อง 100%
    ถ้า accuracy ใกล้ 1.0 → โมเดลเรียนรู้จากข้อมูล Training ได้ดีมาก
    แต่! ถ้า accuracy สูงเกินไป อาจเกิด Overfitting (โมเดลจำข้อมูลแทนที่จะเรียนรู้จริง ๆ)

    หมายความว่า โมเดลสามารถทำนายข้อมูล Validation ได้ 100% ถูกต้อง 
    ถ ้า val_accuracy สูงใกล้เคียงกับ accuracy → โมเดลอาจเรียนรู้ได้ดีจริง
    แต่ถ้า val_accuracy สูงเกินไป โมเดลอาจ Overfitting

    หมายความว่า โมเดลสามารถทำนายข้อมูล Validation ได้ 100% ถูกต้อง
    ถ้า val_accuracy สูงใกล้เคียงกับ accuracy → โมเดลอาจเรียนรู้ได้ดีจริง
    แต่ถ้า val_accuracy สูงเกินไป โมเดลอาจ Overfitting

    accuracy = 1.0000	โมเดลทำนายชุด Train ถูก 100%	อาจ Overfitting
    loss = 0.0129	โมเดลทำผิดพลาดน้อยมากใน Train	ดี (แต่ต้องเช็ค Overfitting)
    val_accuracy = 1.0000	โมเดลทำนายชุด Validation ถูก 100%	น่าสงสัยว่า Overfitting
    val_loss = 0.0119	โมเดลทำผิดพลาดน้อยใน Validation	ดีมาก (แต่ดูผิดปกติ)