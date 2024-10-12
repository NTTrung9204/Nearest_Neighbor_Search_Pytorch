import torch
from model import MODEL
from torchsummary import summary
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import evaluate_model, class_names, preprocess, tensor_to_list
from load_data import LOAD_DATA
import warnings
from sklearn.neighbors import KDTree
import time
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

warnings.filterwarnings("ignore")


class_names = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MODEL(class_names, device).load_model()
model.load_state_dict(torch.load('model_efficientnet_b5_v5.pth', map_location=device))
model.to(device)
model.eval()

# Tạo feature extractor bằng cách loại bỏ lớp cuối cùng
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

if __name__ == '__main__':
    myData = LOAD_DATA("dataset", ["Training Data", "Validation Data", "Testing Data"]) # load dữ liệu từ lớp LOAD_DATA
    testing_dataset = myData.dataloaders['Validation Data'].dataset # lấy tập dữ liệu thử nghiệm
    
    input_image_path = 'actual_test_img/desert_1.jpg' # đường dẫn ảnh input
    try:
        img_tg = Image.open(input_image_path) # đọc ảnh input dưới dạng numpy array, kích thước (H, W, C), kênh màu RGB
    except Exception as e:
        print(f"Lỗi khi mở ảnh: {e}")
        exit(1)
    
    # unsqueeze(0) để thêm một chiều mới vào tensor, kích thước (1, H, W, C)
    # tại sao phải thêm một chiều mới vào tensor? vì mô hình yêu cầu đầu vào có kích thước (N, C, H, W)
    # chuyển ảnh input về tensor và chuyển về device
    # tại sao phải chuyển về device? vì mô hình đang ở trên device (GPU hoặc CPU)
    # device là gì? device là nơi mà tensor được lưu trữ và xử lý, có thể là CPU hoặc GPU
    img_tensor = preprocess(img_tg).unsqueeze(0).to(device) # tiền xử lý ảnh input và chuyển về tensor

    # no_grad để tắt việc tính toán đạo hàm, giúp tăng tốc quá trình tính toán
    # tại sao phải tắt việc tính toán đạo hàm? vì ở đây ta chỉ cần trích xuất feature map
    # chỉ tính toán đạo hàm khi huấn luyện mô hình
    with torch.no_grad():
        start_time = time.time()
        # output_tg là vector đặc trưng của ảnh input, có kích thước (1, 2048)
        output_tg = feature_extractor(img_tensor)
        # chuyển tensor về list vì KDTree yêu cầu dữ liệu dạng list
        feature_extractor_target = tensor_to_list(output_tg)
        extraction_time = time.time() - start_time
        print(f"Thời gian để trích xuất feature map từ ảnh input: {extraction_time:.4f} giây")

    # Trích xuất đặc trưng cho tập dữ liệu thử nghiệm
    list_feature_map = []
    start_time = time.time()

    # Lặp qua tất cả các ảnh trong tập dữ liệu thử nghiệm và trích xuất đặc trưng
    for i in range(len(testing_dataset.samples)):
        sys.stdout.write(f"\rĐang trích xuất feature map từ ảnh thứ {i+1}/{len(testing_dataset.samples)}")
        img_path = testing_dataset.samples[i][0]
        img_sp = Image.open(img_path)
        ft_map = preprocess(img_sp).unsqueeze(0).to(device)
        with torch.no_grad():
            output_sp = feature_extractor(ft_map)
        feature_map = tensor_to_list(output_sp)
        list_feature_map.append(feature_map)

    extraction_dataset_time = time.time() - start_time
    print(f"Tổng số các feature map: {len(list_feature_map)}")
    if len(list_feature_map) > 0:
        print(f"Kích thước đặc trưng mỗi ảnh trong tập dữ liệu: {len(list_feature_map[0])}")
    print(f"Thời gian để trích xuất feature map từ tập dữ liệu: {extraction_dataset_time:.4f} giây")


    # Kiểm tra kích thước đặc trưng
    if len(feature_extractor_target) != len(list_feature_map[0]):
        print("Kích thước đặc trưng không khớp!")
        exit(1)

    start_time = time.time()
    # tạo cây KDTree để tìm kiếm 5 ảnh tương đồng nhất
    kdtree = NearestNeighbors(n_neighbors=5, metric='euclidean')
    # distance, indexs = kdtree.query(np.array([feature_extractor_target]), k = 5)
    kdtree.fit(list_feature_map)
    # thu được khoảng cách và vị trí của 5 ảnh tương đồng nhất, ở dạng numpy array
    distance, indexs = kdtree.kneighbors([feature_extractor_target], n_neighbors=5)
    search_time = time.time() - start_time
    print(f"Thời gian để tìm kiếm: {search_time:.4f} giây")
    # print(list_feature_map[index])

    print("Các khoảng cách tương ứng:", distance)
    print("Vị trí của các ảnh tương đồng nhất:", indexs)

    # Lấy đường dẫn của danh sách ảnh tương đồng nhất
    # print("Các ảnh tương đồng nhất:")
    # for i in indexs:
    #     print(testing_dataset.samples[i][0])
    
    # Hiển thị tất cả ảnh tương đồng nhất trong 1 cửa sổ
    fig = plt.figure(figsize=(10, 6))

    ax1 = fig.add_subplot(2, 1, 1)  # 2 hàng, 1 cột, vị trí 1
    ax1.imshow(img_tg, cmap='gray')
    ax1.axis('off')  # Tắt trục
    ax1.title.set_text('Input image')
    
    for k, v in enumerate(indexs[0]):
        ax = fig.add_subplot(2, 5, 6 + k)  # 2 hàng, 5 cột, bắt đầu từ vị trí 6
        img = Image.open(testing_dataset.samples[v][0])
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.title.set_text(f'{distance[0][k]:.4f}')
    
    plt.show()