import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import models
import torch.nn.functional as F

# Xây dựng mô hình
class CustomEfficientNetB5(nn.Module):
    def __init__(self, num_classes):
        super(CustomEfficientNetB5, self).__init__()
        # Tải mô hình pre-trained EfficientNet-B5 với trọng số hiện tại
        weights = models.EfficientNet_B5_Weights.IMAGENET1K_V1
        self.base_model = models.efficientnet_b5(weights=weights)
        
        # Lưu lại số lượng in_features từ lớp classifier gốc
        # Thông thường, classifier là Sequential với [Dropout, Linear]
        # Vì vậy, chúng ta truy cập lớp Linear để lấy in_features
        in_features = self.base_model.classifier[1].in_features
        
        # Loại bỏ lớp classifier gốc
        self.base_model.classifier = nn.Identity()
        
        # Thêm các lớp mới
        self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, 256)
        self.dropout = nn.Dropout(p=0.45)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Regularization
        self.l2_reg = 0.016
        self.l1_reg = 0.006

    def forward(self, x):
        x = self.base_model(x)  # Output của base_model là [batch_size, in_features]
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

def load_model(class_names, device):
    feature_name_no_train = ["features.6", "features.5", "features.4", "features.3", "features.2", "features.1", "features.0"]
    model = CustomEfficientNetB5(len(class_names))
    for name, param in model.named_parameters():
        if any(no_train in name for no_train in feature_name_no_train):
            param.requires_grad = False

    model = model.to(device)

    return model
