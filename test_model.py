import torch
from model import MODEL
from torchsummary import summary
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import evaluate_model
from load_data import LOAD_DATA
import warnings
from sklearn.neighbors import KDTree
import time
from torch.utils.data import DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


class_names = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Thiết bị sử dụng: {device}")

model = MODEL(class_names, device).load_model()
model.load_state_dict(torch.load('model_efficientnet_b5_v5.pth', map_location=device))
model.to(device)
model.eval()

# print(model)
# summary(model, (3, 224, 224))

# Tạo feature extractor bằng cách loại bỏ lớp cuối cùng
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# image = Image.open('actual_test_img/mountain_1.jpg')
# image = Image.open('actual_test_img/desert_1.jpg')

# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# image = transform(image).unsqueeze(0).to(device)

# prediction = model(image)
# _, preds = torch.max(prediction, 1)
# probabilities = F.softmax(prediction, dim=1)
# print(probabilities)
# print(f'Probability: {probabilities[0][preds].item()}')
# print(f'Prediction: {class_names[preds]}')

def show_image(img, label, prediction):
    print(img.shape)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    print(img.shape)
    plt.imshow(img)
    plt.title(f'Label: {class_names[label]} | Prediction: {class_names[prediction]}')
    plt.show()

def tensor_to_list(tensor):
    return tensor.squeeze(0).view(-1).cpu().tolist()

# Định nghĩa các bước tiền xử lý ảnh (giống như lúc huấn luyện)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  
    transforms.ToTensor(),          # Chuyển ảnh thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

if __name__ == '__main__':

    myData = LOAD_DATA("dataset", ["Training Data", "Validation Data", "Testing Data"])
    testing_dataset = myData.dataloaders['Validation Data'].dataset
    testing_loader = DataLoader(testing_dataset, batch_size=32, shuffle=False)
    # print("Size of testing dataset: ", myData.dataset_sizes['Testing Data'])
    # evaluate_model(model, myData.dataloaders, myData.dataset_sizes, device, callback=show_image)


    # # Load ảnh từ file
    # img = Image.open('actual_test_img/desert_1.jpg')

    # # Áp dụng các bước tiền xử lý lên ảnh
    # img_tensor = preprocess(img)

    # # Thêm một chiều batch (vì mô hình dự đoán theo batch)
    # img_tensor = img_tensor.unsqueeze(0)

    # # Chuyển ảnh sang GPU nếu có
    # img_tensor = img_tensor.to(device)

    # Đặt mô hình ở chế độ đánh giá
    # model.eval()

    # # Dự đoán với ảnh mới
    # with torch.no_grad():  # Không cần tính gradient trong quá trình dự đoán
    #     output = model(img_tensor)

    # # Lấy lớp dự đoán có xác suất cao nhất
    # _, predicted_class = torch.max(output, 1)

    # # In ra kết quả dự đoán
    # print("Dự đoán:", class_names[predicted_class.item()])
    # probabilities = F.softmax(output, dim=1)
    # print(probabilities[0])
    # print("Xác suất:", probabilities[0][predicted_class].item())

    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # # summary(feature_extractor)

    # feature_extractor.eval()

    # start_time = time.time()
    # with torch.no_grad():
    #     output = feature_extractor(img_tensor)

    # # print(tensor_to_list(output[0]))

    # feature_extractor_target = tensor_to_list(output[0])
    # print("Thời gian để trích xuất feature map từ ảnh input: ", time.time() - start_time)

    # start_time = time.time()
    # list_feature_map = []

    # testing_data = myData.dataloaders['Testing Data']
    # for index in range(len(testing_data.dataset.samples)):
    #     img_path = testing_data.dataset.samples[index][0]
    #     img = Image.open(img_path)
    #     img_tensor = preprocess(img)
    #     img_tensor = img_tensor.unsqueeze(0)
    #     img_tensor = img_tensor.to(device)

    #     with torch.no_grad():
    #         output = feature_extractor(img_tensor)

    #     feature_map = tensor_to_list(output[0])
    #     list_feature_map.append(feature_map)

    # print("Tổng số các feature map: ", len(list_feature_map))
    # print("Tổng số các feature map: ", len(list_feature_map[0]))

    # print("Thời gian để trích xuất feature map từ tập dữ liệu: ", time.time() - start_time)

    # start_time = time.time()
    # kdtree = KDTree(list_feature_map)
    # distance, index = kdtree.query(feature_extractor_target)

    # print("Khoảng cách: ", distance)
    # print("Vị trí: ", index)

    # print("Thời gian để tìm kiếm: ", time.time() - start_time)

    # # get the image path of the most similar image
    # print("The most similar image: ", testing_data.dataset.samples[index][0])

    # # show the most similar image
    # img = Image.open(testing_data.dataset.samples[index][0])
    # plt.imshow(img)
    # plt.show()

    # Tạo DataLoader với batch size phù hợp
    # Trích xuất đặc trưng cho ảnh input
    input_image_path = 'actual_test_img/glacier_2.jpeg'  # Thay đổi đường dẫn ảnh input tại đây
    # input_image_path = 'dataset/Testing Data/Forest/Forest-Test (10).jpeg'  # Thay đổi đường dẫn ảnh input tại đây
    try:
        img_tg = Image.open(input_image_path)
        # print(np.array(img))
    except Exception as e:
        print(f"Lỗi khi mở ảnh: {e}")
        exit(1)
    
    img_tensor = preprocess(img_tg).unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time.time()
        output_tg = feature_extractor(img_tensor)
        feature_extractor_target = tensor_to_list(output_tg)
        extraction_time = time.time() - start_time
        print(f"Thời gian để trích xuất feature map từ ảnh input: {extraction_time:.4f} giây")

    # print(f"Feature map của ảnh input: {feature_extractor_target}")
    # Trích xuất đặc trưng cho tập dữ liệu thử nghiệm
    list_feature_map = []
    start_time = time.time()
    # with torch.no_grad():
    #     for batch in testing_loader:
    #         images, labels = batch
    #         images = images.to(device)
    #         outputs = feature_extractor(images)
    #         outputs = outputs.view(outputs.size(0), -1)
    #         list_feature_map.extend(outputs.cpu().tolist())

    for i in range(len(testing_dataset.samples)):
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

    # Tạo KDTree
    start_time = time.time()
    # kdtree = KDTree(list_feature_map, metric='euclidean')
    kdtree = NearestNeighbors(n_neighbors=5, metric='euclidean')
    # distance, indexs = kdtree.query(np.array([feature_extractor_target]), k = 5)
    kdtree.fit(list_feature_map)
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
    
    for k, v in enumerate(indexs[0]):
        ax = fig.add_subplot(2, 5, 6 + k)  # 2 hàng, 5 cột, bắt đầu từ vị trí 6
        img = Image.open(testing_dataset.samples[v][0])
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.show()

        








