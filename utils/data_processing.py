# utils/data_processing.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(image_dir, img_size=(224, 224)):
    images = []
    labels = []
    class_names = []
    class_indices = {}

    # Duyệt qua các thư mục con
    for idx, class_name in enumerate(sorted(os.listdir(image_dir))):
        class_path = os.path.join(image_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            class_indices[class_name] = idx
            # Duyệt qua các file hình ảnh trong thư mục con
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Đọc và tiền xử lý hình ảnh
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                        labels.append(idx)
                    else:
                        print(f"Lỗi khi đọc hình ảnh: {img_path}")
    images = np.array(images)
    labels = np.array(labels)

    # Chuẩn hóa giá trị pixel
    images = images / 255.0

    # Chuyển đổi labels thành one-hot
    num_classes = len(class_names)
    labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Chia dữ liệu thành tập huấn luyện và kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, class_names, class_indices

def save_class_indices(class_indices, filename='class_indices.npy'):
    """
    Lưu mapping giữa tên lớp và chỉ số lớp.
    """
    np.save(filename, class_indices)

def load_class_indices(filename='class_indices.npy'):
    """
    Tải mapping giữa tên lớp và chỉ số lớp.
    """
    return np.load(filename, allow_pickle=True).item()
