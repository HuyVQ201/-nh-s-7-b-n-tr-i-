import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model

# Đường dẫn đến ảnh
image_path = r"C:/Users/Administrator/Pictures/Nền/Quang.jpg"

# Tải hình ảnh từ đường dẫn đã cho và điều chỉnh kích thước của nó thành 128x128 pixel
img = image.load_img(image_path, target_size=(128, 128))

# Chuyển đổi hình ảnh thành mảng numpy và mở rộng chiều
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Chuyển đổi hình ảnh thành tensor
img_tensor = tf.convert_to_tensor(img_array)

# Tạo input layer
inputs = tf.keras.layers.Input(shape=(128, 128, 3))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

###################
layer_names = ['input_layer', 'lambda', 'conv2d', 'dropout', 'conv2d_1', 'max_pooling2d', 'conv2d_2', 'dropout_1', 'conv2d_3', 'max_pooling2d_1', 'conv2d_4', 'dropout_2', 'conv2d_5', 'max_pooling2d_2', 'conv2d_6', 'dropout_3', 'conv2d_7', 'max_pooling2d_3', 'conv2d_8', 'dropout_4', 'conv2d_9', 'conv2d_transpose', 'concatenate', 'conv2d_10', 'dropout_5', 'conv2d_11', 'conv2d_transpose_1', 'concatenate_1', 'conv2d_12', 'dropout_6', 'conv2d_13', 'conv2d_transpose_2', 'concatenate_2', 'conv2d_14', 'dropout_7', 'conv2d_15', 'conv2d_transpose_3', 'concatenate_3', 'conv2d_16', 'dropout_8', 'conv2d_17', 'conv2d_18']

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D(2, 2)(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(2, 2)(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansttive path

u6 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

# Tạo mô hình
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Hiển thị cấu trúc mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Tạo một danh sách chứa tất cả các lớp Conv2D và MaxPooling2D trong mô hình
conv_maxpool_layers = [layer for layer in model.layers if isinstance(layer, (Conv2D, MaxPooling2D))]

# Tạo một mô hình mới với cùng đầu vào như mô hình gốc, nhưng đầu ra là các đầu ra giữa
feature_map_model = Model(inputs=model.inputs, outputs=[layer.output for layer in conv_maxpool_layers])

# Lấy tất cả các feature map
feature_maps = feature_map_model.predict(img_tensor)


import matplotlib.pyplot as plt
import numpy as np

for i, feature_map in enumerate(feature_maps):
    num_feature_maps = feature_map.shape[-1]

    # Số hàng và cột cho subplot
    rows = int(np.sqrt(num_feature_maps))
    cols = int(np.ceil(num_feature_maps / rows))

    # Hiển thị tất cả các feature map đầu ra của mô hình
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    fig.suptitle(f'Feature maps of layer {i + 1}')
    
    sess = tf.compat.v1.Session()
    layer = model.get_layer(layer_names[i])
    # Check the shape of the feature_map tensor
    print(f'Shape of feature_map tensor for layer {i + 1}: {feature_map.shape}')

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
        if rows * cols < num_feature_maps:
            axes[j % num_filters].axis('off')
        else:
            axes[j // cols, j % cols].axis('off')

    plt.show()


rows = 4
cols = 4

for i, layer in enumerate(model.layers):
    # Hiển thị tất cả các filter
    if hasattr(layer, 'kernel'):
        kernel = layer.kernel.numpy()
        num_filters = kernel.shape[3]
        fig, axes = plt.subplots(num_filters, 1, figsize=(5, 5 * num_filters))
        fig.suptitle(f'Filters for Layer {i + 1}')

        for j in range(num_filters):
            filter = kernel[:, :, :, j]
            filter_np = np.squeeze(filter)
            if filter_np.shape[-1] == 3:
                filter_np = np.transpose(filter_np, (1, 0, 2))
            if rows * cols < num_filters:
                ax = axes[j % num_filters]
            else:
                ax = axes[j // cols, j % cols]
            ax.imshow(filter_np, cmap='gray')
            ax.axis('off')

        plt.show()
    else:
        print(f'Layer {i + 1} is an InputLayer and does not have weights.')