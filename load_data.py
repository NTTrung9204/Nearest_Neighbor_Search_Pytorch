import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os


class LOAD_DATA:
    def __init__(self, dir_path: str, list_folder: list[str]) -> None:
        self.dir_path = dir_path
        self.list_folder = list_folder
        self.data_transforms = self.load_data_transforms()
        self.image_datasets = self.load_image_datasets()
        self.dataloaders = self.load_dataloaders()
        self.dataset_sizes = self.load_dataset_sizes()

    def load_image_datasets(self):
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(self.dir_path, x), self.data_transforms[x]
            )
            for x in self.list_folder
        }
        class_names = image_datasets[self.list_folder[0]].classes
        print(f"Lớp phân loại: {class_names}")
        return image_datasets

    def load_dataloaders(self):
        dataloaders = {
            x: DataLoader(
                self.image_datasets[x], batch_size=64, shuffle=True, num_workers=6
            )
            for x in self.list_folder
        }
        return dataloaders

    def load_dataset_sizes(self):
        dataset_sizes = {x: len(self.image_datasets[x]) for x in self.list_folder}
        return dataset_sizes

    def load_data_transforms(self):
        data_transforms = {}
        for name_folder in self.list_folder:
            if name_folder == "Training Data":
                data_transforms[name_folder] = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            else:
                data_transforms[name_folder] = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])
                ])
        return data_transforms

    def get_class_names(self):
        return self.image_datasets["Training Data"].classes
