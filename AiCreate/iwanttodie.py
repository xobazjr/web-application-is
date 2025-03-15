from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt  
import numpy as np
import matplotlib
import os
import cv2

matplotlib.rc("font", family="Tahoma")

img_path = "basedata/train/happy/auh1.jpg"

if not os.path.exists(img_path):
    raise FileNotFoundError(f"ไม่พบไฟล์: {img_path}")

img = image.load_img(img_path)
plt.imshow(img)
plt.axis("off")
plt.title("ภาพจาก Keras")

img_cv2 = cv2.imread(img_path)  
img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB) 

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_cv2)
plt.axis("off")
plt.title("ภาพ RGB")

img_gray = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap="gray")
plt.colorbar()
plt.title("เมทริกซ์ค่าพิกเซล (Grayscale)")

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=50,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    brightness_range=[0.2, 1.8],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)


train_dataset = train.flow_from_directory('basedata/train/',
                                          target_size=(200, 200),
                                          batch_size=32,
                                          class_mode='categorical')

validation_dataset = validation.flow_from_directory('basedata/validation/',
                                                    target_size=(200, 200),
                                                    batch_size=32,
                                                    class_mode='categorical')

print("Class Indices:", train_dataset.class_indices)
print("Labels Mapping:", dict(enumerate(train_dataset.class_indices)))
print("Labels ของแต่ละภาพใน train_dataset:", train_dataset.classes)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  
])

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200,200,3)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),
    
#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.3), 
#     tf.keras.layers.Dense(4, activation='softmax')
# ])

optimizer = Adam(learning_rate=0.00005)
# optimizer = RMSprop(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',  
              optimizer=optimizer,
              metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# model.fit(train_dataset,
#           steps_per_epoch=5, 
#           epochs=100,
#           validation_data=validation_dataset)
model.fit(train_dataset,
          validation_data=validation_dataset,
          epochs=50,
          callbacks=[early_stopping])

plt.plot(model.history.history['accuracy'], label='Train Accuracy')
plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

plt.plot(model.history.history['loss'], label='Train Loss')
plt.plot(model.history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")
plt.show()

model.save("NNmodel.h5")

class_labels = {0: "Angry", 1: "Happy", 2: "Normal", 3: "Sleep"}


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    
    img_array = img_array.astype('float32') / 255.0
 
    img_cv = img_array.copy()
    
   
    alpha = 1.2 
    beta = 10  
    img_cv = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)
    
    
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_cv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    img_array = img_cv / 255.0 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img_cv

dir_path = 'basedata/test'

if not os.path.exists(dir_path):
    raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {dir_path}")

for img_name in os.listdir(dir_path):
    img_path = os.path.join(dir_path, img_name)

    if img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        img = image.load_img(img_path, target_size=(200, 200))  
        img_array = image.img_to_array(img) 
        img_array = np.expand_dims(img_array, axis=0) 
        img_array /= 255.0  
      
        val = model.predict(img_array)
        predicted_class = np.argmax(val)  
        predicted_label = class_labels[predicted_class]

        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{img_name}\nPrediction: {predicted_label}")
        plt.show()
    else:
        print(f"ไฟล์ {img_name} ไม่ใช่ภาพ, ข้ามไป")

