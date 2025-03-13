#ติดตั้งก่อนในเทอมินอล pip install tensorflow 


import os #มันเป็นอะไรไม่รู้ซึ้งผมก็ต้องปิดด้วยคำสั่งนี้
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"#มันเป็นอะไรไม่รู้ซึ้งผมก็ต้องปิดด้วยคำสั่งนี้

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "../dataset/Data from NSO Catalog.csv")

import tensorflow as tf 
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.rc("font", family="Tahoma") #ภาษาไทยไม่รองรับจึงต้องใส่

# นำเข้าข้อมูลตัวอย่าง
df = pd.read_csv(file_path, encoding="utf-8-sig")



# print(df.head()) #แสดง5ตัวแรก

#เลือกข้อมูลที่จะเอาแกนXYจากcsvข้อมูลมาใช่
X = df["YEAR"]  
Y = df["MONTHLY_INCOME"]
income_mean = df.groupby("YEAR")["MONTHLY_INCOME"].mean()#เอาไว้ใช้กับกราฟแท่งโดยเฉพาะ


#8บันทึกข้อมูล
df.to_pickle("processed_data.pkl")  # ใช้ Pickle 
df.to_csv("processed_data.csv", index=False, encoding="utf-8-sig")  


# พล็อตกราฟ
plt.figure(figsize=(10, 5))#ขนาด
#plt.plot(X, Y, 'o', alpha=0.5)  # ใช้ 'o' เพื่อให้เป็นจุด 
#plt.plot(X, Y, marker="o", linestyle="-", color="b") #เป็นกราฟเส้นตรง
#plt.bar(income_mean.index, income_mean.values, color="blue", alpha=0.7) #กราฟแท่ง
#plt.hist(df["MONTHLY_INCOME"], bins=20, color="green", alpha=0.7, edgecolor="black") #ฮิสโตแกม
#plt.boxplot(df["MONTHLY_INCOME"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue")) #บอกพอต


plt.scatter(X, Y, c=Y, cmap="coolwarm", alpha=0.7) #สกาเลทวิช ดูสวยดูดีดูเวอร์
 

plt.xlabel("ในแต่ละปี ") #ใส่ข้อความที่X
plt.ylabel("รายได้ต่อเดือนของคนไทย ") #ใส่ข้อความที่Y
plt.title("รายได้ต่อครัวเรือนในประเทศไทย") #ชื่อหัวข้อของกราฟ
plt.grid(True)

plt.colorbar(label="ระดับรายได้")

#plt.xticks(rotation=45)  # หมุนชื่อปีให้อ่านง่าย
#plt.grid(axis="y") #กราฟแท่งกับฮิตโตแกรมนิยม
plt.show() #แสดง