# app.py

from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Đường dẫn tải lên
UPLOAD_FOLDER = 'static/uploads/'
RESULT_FOLDER = 'static/results/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Tạo các thư mục nếu chưa tồn tại
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Tải mô hình
model_path = os.path.join('saved_model', 'animal_classification_model.h5')
model = tf.keras.models.load_model(model_path)

# Tải class_indices
class_indices_path = 'class_indices.npy'
class_indices = np.load(class_indices_path, allow_pickle=True).item()
idx_to_class = {v: k for k, v in class_indices.items()}

# Trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Xử lý tải lên ảnh
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Xử lý ảnh
        result_image, class_name, confidence = process_image(filepath)
        return render_template('results.html', result_image=result_image, class_name=class_name, confidence=confidence)
    return redirect('/')

def process_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    # Tiền xử lý
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Dự đoán
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    class_name = idx_to_class[class_idx]
    
    # Lưu ảnh kết quả
    result_filename = os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, image)
    return result_filename, class_name, confidence

if __name__ == '__main__':
    app.run(debug=True)
