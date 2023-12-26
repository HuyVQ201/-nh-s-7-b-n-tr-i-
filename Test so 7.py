import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf



# Đường dẫn đến ảnh
image_path = "D:/Document/NGHIÊN CỨU KHOA HỌC/Ảnh số 7 bên trái/Ảnh số 7 trái/Ảnh số 7 trái crop.jpg"

# Load ảnh và điều chỉnh kích thước
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Tạo input layer
inputs = tf.keras.layers.Input(shape=(128, 128, 3))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

###################

c1 = tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1) 

c2 = tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', kernel_initializer='he_normal', padding = 'same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation= 'relu', kernel_initializer='he_normal', padding = 'same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c3)
p3 = tf.keras.layers.MaxPooling2D(2,2)(c3)

c4 = tf.keras.layers.Conv2D(128 , (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128 , (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(c4)
p4 = tf.keras.layers.MaxPooling2D(2,2)(c4) 

c5 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(c5)

u6 = tf.keras.layers.Conv2DTranspose (128, (3,3), strides = (2,2), padding = 'same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation= 'relu', kernel_initializer= 'he_normal', padding = 'same') (c6)

u7 = tf.keras.layers.Conv2DTranspose (64, (3,3), strides = (2,2), padding = 'same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', kernel_initializer= 'he_normal', padding = 'same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation= 'relu', kernel_initializer= 'he_normal', padding = 'same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (3,3), strides= (2,2), padding = 'same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer='he_normal', padding = 'same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16 ,(3,3), strides = (2,2), padding = 'same')(c8)
u9 = tf.keras.layers.concatenate ([u9, c1], axis = 3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding = 'same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer= 'he_normal', padding= 'same')(c9)

# Tạo mô hình
outputs = tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(c9)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


# Hiển thị cấu trúc mô hình
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# Hiển thị ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(img_array[0].astype('uint8'))  # Đổi kiểu dữ liệu về uint8 để hiển thị đúng
plt.title('Original Image')

# Dự đoán với ảnh đầu vào
output_feature_map = model.predict(img_array)

# Hiển thị feature map đầu ra của mô hình
plt.subplot(1, 2, 2)
plt.imshow(output_feature_map[0, :, :, 0], cmap='viridis')
plt.title('Output Feature Map')

plt.show()
