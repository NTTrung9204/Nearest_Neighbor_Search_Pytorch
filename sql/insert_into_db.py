import mysql.connector
from utils import save_vector_to_db
from load_data import LOAD_DATA
from utils import preprocess, tensor_to_list
from model import MODEL
from PIL import Image
import torch
import warnings
import sys
from EfficientNet_model import load_model

from pathlib import Path

# Xác định đường dẫn tới thư mục project
current_dir = Path(__file__).resolve().parent  # Thư mục hiện tại (sql/)
project_root = current_dir.parent  # Thư mục gốc (project)

# Tạo đường dẫn tới thư mục Dataset
dataset_dir = project_root / "Dataset"

# Kết nối tới cơ sở dữ liệu MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",       # Tên người dùng MySQL
    password="123456",  # Mật khẩu MySQL
    database="numpy_array"  # Tên database
)

warnings.filterwarnings("ignore")

class_names = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MODEL(class_names, device).load_model()
model = load_model(class_names, device)
model.load_state_dict(torch.load('model_efficientnet_b5_v6.pth', map_location=device))
model.to(device)
model.eval()

feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
feature_extractor.to(device)
feature_extractor.eval()


if __name__ == '__main__':
    cursor = connection.cursor()

    myData = LOAD_DATA(str(dataset_dir), ["Training Data", "Validation Data", "Testing Data"])
    testing_dataset = myData.dataloaders['Training Data'].dataset

    # Trích xuất đặc trưng cho tập dữ liệu thử nghiệm
    list_feature_map = []
    for i in range(len(testing_dataset.samples)):
        sys.stdout.write(f"\rĐang trích xuất feature map từ ảnh thứ {i+1}/{len(testing_dataset.samples)}")
        img_path = testing_dataset.samples[i][0]
        img_sp = Image.open(img_path)
        ft_map = preprocess(img_sp).unsqueeze(0).to(device)
        with torch.no_grad():
            output_sp = feature_extractor(ft_map)
        feature_map = tensor_to_list(output_sp)
        list_feature_map.append(feature_map)

    print()
    
    print("Trích xuất xong feature map.")
    print("kích thước feature map:", len(list_feature_map))
    print("kích thước vector:", len(list_feature_map[0]))


    # Lưu feature map vào MySQL
    for i, feature_map in enumerate(list_feature_map):
        sys.stdout.write(f"\rĐang lưu feature map thứ {i+1}/{len(list_feature_map)}")
        img_path = testing_dataset.samples[i][0]
        save_vector_to_db(feature_map, connection, cursor, img_path)

    print("\nĐã lưu xong feature map.")
    # Đóng kết nối
    cursor.close()
    connection.close()

