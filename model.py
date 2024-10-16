from torchvision import models
from torch import nn

class MODEL:
    def __init__(self, class_names: list[str], device) -> None:
        self.class_names = class_names
        self.device = device

    def load_model(self):
        model = models.efficientnet_b5(pretrained=True)

        # Đóng băng các tham số của mô hình để không cập nhật trong quá trình huấn luyện
        for name, param in model.named_parameters():
            # print(name)
            if name.startswith('features.6') or name.startswith('features.5') or name.startswith('features.4') or name.startswith('features.3') or name.startswith('features.2') or name.startswith('features.1') or name.startswith('features.0'):
                param.requires_grad = False

        num_ftrs = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(self.class_names)),
            nn.LogSoftmax(dim=1)
        )

        # Chuyển mô hình đến device
        model = model.to(self.device)

        return model