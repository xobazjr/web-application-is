#ติดตั้งก่อนในเทอมินอล pip install tensorflow 


import os #มันเป็นอะไรไม่รู้ซึ้งผมก็ต้องปิดด้วยคำสั่งนี้
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"#มันเป็นอะไรไม่รู้ซึ้งผมก็ต้องปิดด้วยคำสั่งนี้

import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.rc("font", family="Tahoma") #ภาษาไทยไม่รองรับจึงต้องใส่

# นำเข้าข้อมูลตัวอย่าง
df = pd.read_csv("dataset/Data from NSO Catalog.csv", encoding="utf-8-sig")

print(df.head()) #แสดง5ตัวแรก

#เลือกข้อมูลที่จะเอาแกนXYจากcsvข้อมูลมาใช่
X = df["YEAR"]  
Y = df["MONTHLY_INCOME"]

# พล็อตกราฟ
plt.figure(figsize=(10, 5))#ขนาด
plt.plot(X, Y, 'o', alpha=0.5)  # ใช้ 'o' เพื่อให้เป็นจุด
plt.xlabel("ปี ") #ใส่ข้อความที่X
plt.ylabel("รายได้รายเดือน ") #ใส่ข้อความที่Y
plt.title("รายได้ต่อครัวเรือนในไทย") #ชื่อหัวข้อของกราฟ
plt.grid(True)
plt.show() #แสดง