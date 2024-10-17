# model/train.py

import sys
import os

# Thêm thư mục gốc vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import data_processing as dp
from model_definition import create_model
import tensorflow as tf
import numpy as np
import os

# Đường dẫn dữ liệu
IMAGE_DIR = os.path.join(parent_dir, 'data', 'images')
IMG_SIZE = (224, 224)

if not os.path.exists(IMAGE_DIR):
    print(f"Đường dẫn {IMAGE_DIR} không tồn tại.")
    sys.exit(1)


# Tải dữ liệu
X_train, X_test, y_train, y_test, class_names, class_indices = dp.load_data(IMAGE_DIR, IMG_SIZE)

# Lưu class_indices
dp.save_class_indices(class_indices, filename='class_indices.npy')

# Số lớp
num_classes = len(class_names)

# Tạo mô hình
model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=num_classes)

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Lưu mô hình
if not os.path.exists('saved_model'):
    os.makedirs('saved_model')
model.save('saved_model/animal_classification_model.h5')
