from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from utils import preprocess, tensor_to_list, get_feature_map_from_image, query_top_k_most_similar_images
import torch
from model import MODEL
import warnings
import mysql.connector

warnings.filterwarnings("ignore")

# Kết nối tới cơ sở dữ liệu MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",  # Tên người dùng MySQL
    password="123456",  # Mật khẩu MySQL
    database="numpy_array",  # Tên database
)

cursor = connection.cursor()

table_name = "v6"

app = Flask(__name__)
CORS(app)  # Cho phép CORS để trang HTML có thể giao tiếp với máy chủ

# Các loại tệp ảnh được phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Endpoint để phục vụ tệp index.html
@app.route('/', methods=['GET'])
def serve_index():
    return send_file('index.html')

# Endpoint để xử lý tải lên và xử lý ảnh
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            feature_map = get_feature_map_from_image(file.stream)

            list_distances, list_base64_images = query_top_k_most_similar_images(feature_map, cursor, table_name, k1=25, k2=10000)
            list_distances = list_distances[0].tolist()
            # print('List distances:', list_distances)
            # print('List images:', list_base64_images)

            return jsonify({'message': 'Image processed successfully', 'image_data': list_base64_images, 'list_distances': list_distances}), 200
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)
