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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pickle

matplotlib.rc("font", family="Tahoma") #ภาษาไทยไม่รองรับจึงต้องใส่

# นำเข้าข้อมูลตัวอย่าง
df = pd.read_csv(file_path, encoding="utf-8-sig")

#ลบค่าที่ไม่มี
df = df.replace("UNKNOWN_VALUE", np.nan) 
df.dropna(inplace=True)



#แปลง
df["MONTHLY_INCOME"] = pd.to_numeric(df["MONTHLY_INCOME"], errors="coerce")
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")

df.dropna(inplace=True)



# แปลงให้อย


scaler_year = MinMaxScaler(feature_range=(0, 1))

scaler_income = MinMaxScaler()

df["YEAR"] = scaler_year.fit_transform(df[["YEAR"]])
df["MONTHLY_INCOME"] = scaler_income.fit_transform(df[["MONTHLY_INCOME"]])



# print(df.head()) #แสดง5ตัวแรก

#เลือกข้อมูลที่จะเอาแกนXYจากcsvข้อมูลมาใช่
# X = df["YEAR"]  
# Y = df["MONTHLY_INCOME"]
# income_mean = df.groupby("YEAR")["MONTHLY_INCOME"].mean()#เอาไว้ใช้กับกราฟแท่งโดยเฉพาะ

# ตั้งxy
X = df[["YEAR"]].values  
y = df["MONTHLY_INCOME"].values  


#แบ่งข้อมูลไว้เทรนกับไว้เทส
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#ทำโมเดลขำๆ อันนี้ใหญ่ไปต้องลด
# model = keras.Sequential([
#     keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),  #L1
#     keras.layers.Dense(8, activation="relu"),  #L2
#     keras.layers.Dense(1, activation="linear")  
# ])
# จากอันบนนี่เลยคับขอเสนอลดขนาดลงมา
# model = keras.Sequential([
#     keras.layers.Dense(8, activation="relu", input_shape=(X_train.shape[1],)),  # l1
#     keras.layers.Dropout(0.2),  # ลด 20%
#     keras.layers.Dense(1, activation="linear") 
# ])

#ปรับปรุงโมเดล
# model = keras.Sequential([
#     keras.layers.Dense(12, activation="relu", input_shape=(X_train.shape[1],)),  #L1
#     keras.layers.Dropout(0.1),  # ลด 10%
#     keras.layers.Dense(6, activation="relu"),  #L2
#     keras.layers.Dense(1, activation="linear")
# ])

#ปรับปรุงโมเดล
# model = Sequential([
#    LSTM(32, activation="relu", return_sequences=True, input_shape=(X_train.shape[1], 1)),
#    Dropout(0.2),
#    LSTM(16, activation="relu", return_sequences=False),
#    Dropout(0.2),
#    Dense(8, activation="relu"),
#    Dense(1, activation="linear")
# ])

#ปรับปรุงโมเดล
model = keras.Sequential([
    keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)), 
    keras.layers.Dropout(0.1),  
    keras.layers.Dense(8, activation="relu"),  
    keras.layers.Dense(1, activation="linear") 
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

#ลดอาการโอเวอร์
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1, callbacks=[early_stopping])
#ตีโมเดลด้วยแซ้
# history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1, callbacks=[early_stopping])


#เอาไว้บอก
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f} | Test MAE: {test_mae:.4f}")


#8บันทึกข้อมูล
model.save("trained_model.keras")


with open("scaler_year.pkl", "wb") as f:
    pickle.dump(scaler_year, f)

with open("scaler_income.pkl", "wb") as f:
    pickle.dump(scaler_income, f)


# # พล็อตกราฟ
# plt.figure(figsize=(10, 5))#ขนาด
# #plt.plot(X, Y, 'o', alpha=0.5)  # ใช้ 'o' เพื่อให้เป็นจุด 
# #plt.plot(X, Y, marker="o", linestyle="-", color="b") #เป็นกราฟเส้นตรง
# #plt.bar(income_mean.index, income_mean.values, color="blue", alpha=0.7) #กราฟแท่ง
# #plt.hist(df["MONTHLY_INCOME"], bins=20, color="green", alpha=0.7, edgecolor="black") #ฮิสโตแกม
# #plt.boxplot(df["MONTHLY_INCOME"], vert=False, patch_artist=True, boxprops=dict(facecolor="lightblue")) #บอกพอต


# plt.scatter(X, Y, c=Y, cmap="coolwarm", alpha=0.7) #สกาเลทวิช ดูสวยดูดีดูเวอร์
 

# plt.xlabel("ในแต่ละปี ") #ใส่ข้อความที่X
# plt.ylabel("รายได้ต่อเดือนของคนไทย ") #ใส่ข้อความที่Y
# plt.title("รายได้ต่อครัวเรือนในประเทศไทย") #ชื่อหัวข้อของกราฟ
# plt.grid(True)

# plt.colorbar(label="ระดับรายได้")

# #plt.xticks(rotation=45)  # หมุนชื่อปีให้อ่านง่าย
# #plt.grid(axis="y") #กราฟแท่งกับฮิตโตแกรมนิยม
# plt.show() #แสดง

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="การสูญเสียของชุดข้อมูลฝึก", color="blue")
plt.plot(history.history["val_loss"], label="การสูญเสียของชุดข้อมูลตรวจสอบความถูกต้อง ", color="red")
plt.xlabel("จำนวนรอบการตีด้วยแซ่ อีดอก(Epochs)")
plt.ylabel("ค่าการสูญเสีย (MAE)")
plt.title("ค่าสูญเสียนะเคิฟ")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="MAE ของชุดข้อมูลฝึก", color="blue")
plt.plot(history.history["val_mae"], label="MAE ของชุดข้อมูลตรวจสอบความถูกต้อง", color="red")
plt.xlabel("จำนวนรอบการตีด้วยแซ่ อีดอก(Epochs)")
plt.ylabel("ค่าข้อผิดพลาดเฉลี่ย (MAE)")
plt.title("กราฟข้อผิดพลาดนะเคิฟ")
plt.legend()

plt.tight_layout()
plt.show()



# โหลด Scaler

with open("scaler_year.pkl", "rb") as f:
    scaler_year = pickle.load(f)

with open("scaler_income.pkl", "rb") as f:
    scaler_income = pickle.load(f)

#ทดสอบการเดา
future_years = np.array([[2015], [2070]]) 
future_years_df = pd.DataFrame(future_years, columns=["YEAR"])
future_years_scaled = scaler_year.transform(future_years_df)

predicted_income = model.predict(future_years_scaled)
predicted_income = scaler_income.inverse_transform(predicted_income)  # แปลงกลับ

print(f"คาดการณ์รายได้ในปี 2015: {predicted_income[0][0]:,.2f} บาท")
print(f"คาดการณ์รายได้ในปี 2070: {predicted_income[1][0]:,.2f} บาท")

df["YEAR"] = scaler_year.inverse_transform(df[["YEAR"]])  
df["MONTHLY_INCOME"] = scaler_income.inverse_transform(df[["MONTHLY_INCOME"]])

plt.figure(figsize=(12, 6))
plt.bar(df["YEAR"], df["MONTHLY_INCOME"], color="blue", alpha=0.7)
plt.xlabel("ปี")
plt.ylabel("รายได้เฉลี่ย (บาท)")
plt.title("รายได้เฉลี่ยของครัวเรือนในแต่ละปี")
plt.xticks(rotation=45)
plt.grid(axis="y")

plt.show()