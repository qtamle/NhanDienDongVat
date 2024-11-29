from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import io
import sqlite3
from database import init_db, add_user, get_user, log_request
import random
import string
import os

app = Flask(__name__)

def generate_random_secret_key(length=24):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for i in range(length))

app.config['SECRET_KEY'] = generate_random_secret_key()
CORS(app)
bcrypt = Bcrypt(app)

init_db()

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()

model_path = './model/fasterrcnn_finetuned.pth'
try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    print(f"Checkpoint loaded from {model_path}. Model state_dict keys:")
except FileNotFoundError:
    print(f"No checkpoint found at {model_path}. Using default COCO weights.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")

ANIMAL_CLASSES = [
    'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe'
]

def load_coco_categories(file_path):
    with open(file_path, 'r') as file:
        categories = [line.strip() for line in file.readlines()]
    return categories

COCO_INSTANCE_CATEGORY_NAMES = load_coco_categories('coco_categories.txt')

def get_prediction(image, threshold):
    transform = T.Compose([T.ToTensor()])
    img = transform(image)
    pred = model([img])
    
    pred_labels = list(pred[0]['labels'].numpy())
    pred_boxes = list(pred[0]['boxes'].detach().numpy())
    pred_scores = list(pred[0]['scores'].detach().numpy())
    
    filtered_boxes = []
    filtered_classes = []
    filtered_scores = []
    
    for idx, score in enumerate(pred_scores):
        if score > threshold:
            class_name = COCO_INSTANCE_CATEGORY_NAMES[pred_labels[idx]]
            if class_name in ANIMAL_CLASSES:
                filtered_boxes.append([(float(pred_boxes[idx][0]), float(pred_boxes[idx][1])), 
                                       (float(pred_boxes[idx][2]), float(pred_boxes[idx][3]))])
                filtered_classes.append(class_name)
                filtered_scores.append(float(score))
    
    return filtered_boxes, filtered_classes, filtered_scores

def draw_boxes(image, boxes, classes, scores):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    for i, box in enumerate(boxes):
        pt1 = (int(box[0][0]), int(box[0][1]))
        pt2 = (int(box[1][0]), int(box[1][1]))
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=2)
        label = f"{classes[i]}: {int(scores[i] * 100)}%"
        cv2.putText(img, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

@app.route('/')
def serve_index():
    return send_from_directory('drool-html', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory('drool-html', path)

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        add_user(username, hashed_password)
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists'}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = get_user(username)
    if user and bcrypt.check_password_hash(user[2], password):
        return jsonify({'message': 'Login successful', 'user_id': user[0]}), 200
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route('/detect', methods=['POST'])
def detect():
    user_id = request.form.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(file.stream)
    
    threshold = float(request.form.get('threshold', 0.5))
    boxes, pred_cls, pred_score = get_prediction(image, threshold)
    
    img_with_boxes = draw_boxes(image, boxes, pred_cls, pred_score)
    
    _, img_encoded = cv2.imencode('.jpg', img_with_boxes)
    img_bytes = img_encoded.tobytes()

    log_request(user_id)
    
    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

@app.route('/log_request', methods=['POST'])
def log_user_request():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401
    
    log_request(user_id)
    return jsonify({'message': 'Request logged successfully'}), 200

@app.route('/get_request_history', methods=['POST'])
def get_request_history():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({'error': 'Unauthorized'}), 401

    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('SELECT timestamp FROM requests WHERE user_id = ?', (user_id,))
    history = [{'timestamp': row[0]} for row in c.fetchall()]
    conn.close()

    return jsonify({'history': history}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
