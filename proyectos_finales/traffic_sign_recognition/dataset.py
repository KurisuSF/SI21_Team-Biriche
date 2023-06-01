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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#to plot accuracy
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split #to split training and testing data
from keras.utils import to_categorical #to convert the labels present in y_train and t_test into one-hot encoding
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout#to create CNN
data = []
labels = []
classes = 43
cur_path = os.path.dirname(os.path.abspath(__file__))
# print(cur_path)

# Retrieving the images and their labels
for i in range(classes):
   path = os.path.join(cur_path,'Train',str(i))
   images = os.listdir(path)
   for a in images:
        try:
           image = Image.open(path + '\\'+ a)
           image = image.resize((30,30))
           image = np.array(image)
          #sim = Image.fromarray(image)
           data.append(image)
           labels.append(i)
        except:
           print("Error loading image")
