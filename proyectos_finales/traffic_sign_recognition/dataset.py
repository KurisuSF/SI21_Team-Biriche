# from pathlib import Path

# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torchvision.utils import make_grid
# import matplotlib.pyplot as plt


# def get_dataloaders():
#     """Returns train and validation dataloaders for the traffic sign recognition dataset"""
#     file_path = Path(__file__).parent.absolute()
#     root_path = file_path / "data/crop_dataset/crop_dataset/"

#     # https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html    
#     dataset = ImageFolder(root=root_path,
#                           transform=get_transform())

#     train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

#     # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
#     train_dataloader = DataLoader(train_dataset,
#                                   batch_size=64,
#                                   shuffle=True)
#     val_dataloader = DataLoader(val_dataset,
#                                 batch_size=64,
#                                 shuffle=False)
#     return train_dataloader, val_dataloader


# def get_transform():
#     return transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Resize((32, 32)),
#         ]
#     )


# def visualize_data():
#     train_dataloader, val_dataloader = get_dataloaders()

#     # Visualize some training images
#     plt.figure(figsize=(8, 8))
#     for data, target in train_dataloader:
#         img_grid = make_grid(data)
#         plt.axis("off")
#         plt.imshow(img_grid.permute(1, 2, 0))
#         plt.show()
#         print(data.shape)
#         print(target.shape)
#         break

#     # Visualize some validation images with labels
#     plt.figure(figsize=(8, 8))
#     for data, target in val_dataloader:
#         for i in range(16):
#             plt.subplot(4, 4, i + 1)
#             plt.axis("off")
#             plt.imshow(data[i].permute(1, 2, 0))
#             plt.title(target[i].item())
#         plt.show()
#         break


# def main():
#     visualize_data()


# if __name__ == "__main__":
#     main()

import torch
import albumentations as A
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from albumentations.pytorch import ToTensorV2
# Required constants.
ROOT_DIR = '../input/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
VALID_SPLIT = 0.1
RESIZE_TO = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 128
NUM_WORKERS = 4 # Number of parallel processes for data preparation.

# Training transforms.
class TrainTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.RandomBrightnessContrast(),
            A.RandomFog(),
            A.RandomRain(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']
# Validation transforms.
class ValidTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']
    
def get_datasets():
    """
    Function to prepare the Datasets.
    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(TrainTransforms(RESIZE_TO))
    )
    dataset_test = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(ValidTransforms(RESIZE_TO))
    )
    dataset_size = len(dataset)
    # Calculate the validation dataset size.
    valid_size = int(VALID_SPLIT*dataset_size)
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])
    return dataset_train, dataset_valid, dataset.classes
def get_data_loaders(dataset_train, dataset_valid):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, valid_loader
