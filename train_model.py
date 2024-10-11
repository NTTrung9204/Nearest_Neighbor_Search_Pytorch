from load_data import LOAD_DATA
from model import MODEL
import torch
from torchsummary import summary
from torch import nn, optim
import matplotlib.pyplot as plt
from utils import train_model, evaluate_model, class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Thiết bị sử dụng: {device}")

if __name__ == '__main__':
    # Load dữ liệu
    data = LOAD_DATA("dataset", class_names)
    model = MODEL(data.get_class_names(), device).load_model()

    # Định nghĩa loss function và optimizer
    criterion = nn.CrossEntropyLoss()

    # Chỉ tối ưu hóa các tham số của lớp fc mới
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)

    # Huấn luyện mô hình
    num_epochs = 50
    model, train_acc, val_acc = train_model(model, data.dataloaders, data.dataset_sizes, criterion, optimizer, device, num_epochs)

    # Đánh giá mô hình
    evaluate_model(model, data.dataloaders, data.dataset_sizes, device)

    # Lưu mô hình
    torch.save(model.state_dict(), 'model_efficientnet_b5_v5.pth')
    print("Đã lưu mô hình.")

    plt.plot(range(1, num_epochs+1), [acc.cpu().numpy() for acc in train_acc], label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), [acc.cpu().numpy() for acc in val_acc], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()