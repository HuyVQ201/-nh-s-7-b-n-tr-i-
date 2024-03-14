import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model


# Đường dẫn đến ảnh
image_path = "D:/Document/NGHIÊN CỨU KHOA HỌC/Ảnh số 7 bên trái/Ảnh số 7 trái/Ảnh-7-trái-2.png"

# Tải hình ảnh từ đường dẫn đã cho và điều chỉnh kích thước của nó thành 128x128 pixel
img = image.load_img(image_path, target_size=(128, 128))

# Chuyển đổi hình ảnh thành mảng numpy và mở rộng chiều
img_array = image.img_to_array(img) 
img_array = np.expand_dims(img_array, axis=0)

# Chuyển đổi hình ảnh thành tensor
img_tensor = tf.convert_to_tensor(img_array)

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


#Expansttive path

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

# Tạo một danh sách chứa tất cả các lớp Conv2D và MaxPooling2D trong mô hình
conv_maxpool_layers = [layer for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]

# Tạo một mô hình mới với cùng đầu vào như mô hình gốc, nhưng đầu ra là các đầu ra giữa
feature_map_model = Model(inputs=model.inputs, outputs=[layer.output for layer in conv_maxpool_layers])

# Lấy tất cả các feature map
feature_maps = feature_map_model.predict(img_tensor)


# Hiển thị tất cả các feature map
for i, feature_map in enumerate(feature_maps):
    num_feature_maps = feature_map.shape[-1]
    
    # Số hàng và cột cho subplot
    rows = int(np.sqrt(num_feature_maps))
    cols = int(np.ceil(num_feature_maps / rows))

    # Hiển thị tất cả các feature map đầu ra của mô hình
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(f'Feature maps of layer {i+1}')

    # Luôn đảm bảo rằng `axes` là một mảng hai chiều
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for j in range(min(num_feature_maps, rows * cols)):
        ax = axes[j // cols, j % cols]
        ax.imshow(feature_map[0, :, :, j], cmap='viridis')
        ax.axis('off')

    # Loại bỏ trục thừa
    for j in range(num_feature_maps, rows * cols):
        axes.flatten()[j].axis('off')

    plt.show()
