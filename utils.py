import torch
import sys
import matplotlib.pyplot as plt
from torchvision import transforms
import json
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import io
import base64
from model import MODEL
from EfficientNet_model import load_model


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('=' * 50)
        sub_epoch = 0

        # Mỗi epoch có 2 pha: training và validation
        for phase in ['Training Data', 'Validation Data']:
            if phase == 'Training Data':
                model.train()  # Đặt mô hình ở chế độ huấn luyện
            else:
                model.eval()   # Đặt mô hình ở chế độ đánh giá

            running_loss = 0.0
            running_corrects = 0

            # Lặp qua dữ liệu
            for inputs, labels in dataloaders[phase]:
                sys.stdout.write(f'\r{phase} - Batch {str(sub_epoch+1):<3}|{len(dataloaders[phase])}')

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history nếu ở pha huấn luyện
                with torch.set_grad_enabled(phase == 'Training Data'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize chỉ ở pha huấn luyện
                    if phase == 'Training Data':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                sub_epoch += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Lưu lại lịch sử accuracy
            if phase == 'Training Data':
                train_acc_history.append(epoch_acc)
            else:
                val_acc_history.append(epoch_acc)
                # Deep copy mô hình nếu đạt accuracy tốt nhất
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()

        print()

    print(f'\nTraining complete. Best validation Acc: {best_acc:.4f}')

    # Tải lại trọng số tốt nhất
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history

def evaluate_model(model, dataloader, dataset_sizes, device):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloader['Testing Data']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['Testing Data']
    print(f'Test Acc: {test_acc:.4f}')

dir_names = ["Training Data", "Validation Data", "Testing Data"]

class_names = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']

def show_image(img, label, prediction):
    print(img.shape)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    print(img.shape)
    plt.imshow(img)
    plt.title(f'Label: {dir_names[label]} | Prediction: {dir_names[prediction]}')
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

# Hàm lưu vector vào MySQL
def save_vector_to_db(vector, connection, cursor, path_name):
    # Chuyển vector sang dạng JSON
    vector_json = json.dumps(vector)
    
    # Câu lệnh SQL để chèn vector vào bảng
    query = """
        INSERT INTO v6 
            (vector, path_name) VALUES (%s, %s);
    """
    
    # Thực thi câu lệnh SQL
    cursor.execute(query, (vector_json, path_name,))
    connection.commit()

def query_all_vectors(cursor, table_name):
    create_temp_table_query = f"""
        CREATE TEMPORARY TABLE temp_ids AS
        SELECT id
        FROM {table_name};
    """
        
    cursor.execute(create_temp_table_query)
    
    select_vectors_query = f"""
        SELECT id, vector, path_name
        FROM {table_name}
        WHERE id IN (SELECT id FROM temp_ids);
    """

    cursor.execute(select_vectors_query)
    
    # Lấy kết quả truy vấn
    list_vectors = cursor.fetchall()

    # Xóa bảng tạm
    drop_temp_table_query = "DROP TEMPORARY TABLE temp_ids;"
    
    cursor.execute(drop_temp_table_query)
    
    return list_vectors

def query_random_k_vectors(cursor, table_name, k = 1000):
    create_temp_table_query = f"""
        CREATE TEMPORARY TABLE temp_ids AS
        SELECT id
        FROM {table_name}
        ORDER BY RAND()
        LIMIT {k};
    """
    cursor.execute(create_temp_table_query)
    
    select_vectors_query = f"""
        SELECT id, vector, path_name
        FROM {table_name}
        WHERE id IN (SELECT id FROM temp_ids);
    """
    cursor.execute(select_vectors_query)
    
    # Lấy kết quả truy vấn
    list_vectors = cursor.fetchall()

    # Xóa bảng tạm
    drop_temp_table_query = "DROP TEMPORARY TABLE temp_ids;"
    
    cursor.execute(drop_temp_table_query)
    
    return list_vectors

def query_top_k_most_similar_vectors(vector, cursor, table_name, k1=5, k2 = 1000):
    if k2 < 10000:
        list_vectors_db = query_random_k_vectors(cursor, table_name, k2)
    else:
        list_vectors_db = query_all_vectors(cursor, table_name)

    # Tạo cây KDTree để tìm kiếm k vector tương đồng nhất
    kdtree = NearestNeighbors(n_neighbors=k1, metric='euclidean')

    # Chuyển list_vectors từ dạng tuple sang dạng list
    list_vectors = [json.loads(vector[1]) for vector in list_vectors_db]

    # Tìm kiếm k vector tương đồng nhất
    kdtree.fit(list_vectors)

    distances, indexs = kdtree.kneighbors([vector], n_neighbors=k1)

    image_paths = [list_vectors_db[index][2] for index in indexs[0]]
    return distances, image_paths

def query_top_k_most_similar_images(vector, cursor, table_name, k1=5, k2 = 1000):
    # Tìm kiếm k vector tương đồng nhất
    distances, image_paths = query_top_k_most_similar_vectors(vector, cursor, table_name, k1, k2)
    
    # Đọc ảnh từ đường dẫn
    images_base64 = [Convert_PIL_image_to_base64(Image.open(image_path)) for image_path in image_paths]

    return distances, images_base64

def Convert_PIL_image_to_base64(PIL_image):
    img_io = io.BytesIO()
    PIL_image.save(img_io, format='PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return img_base64

def get_feature_map_from_image(image_stream):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = MODEL(['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain'], device).load_model()
    model = load_model(['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain'], device)
    model.load_state_dict(torch.load('model_efficientnet_b5_v6.pth', map_location=device))
    model.to(device)
    model.eval()

    feature_extractor = torch.nn.Sequential(*list(model.children())[:-2])
    feature_extractor.to(device)
    feature_extractor.eval()

    img_sp = Image.open(image_stream)
    ft_map = preprocess(img_sp).unsqueeze(0).to(device)
    with torch.no_grad():
        output_sp = feature_extractor(ft_map)
    feature_map = tensor_to_list(output_sp)

    return feature_map
