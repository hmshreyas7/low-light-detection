import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os

# Load images and preview
data_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor()
])
data_root = "./data/ExDark"
exdark_dataset = torchvision.datasets.ImageFolder(root = data_root, transform = data_transform)
data_loader = torch.utils.data.DataLoader(dataset = exdark_dataset, batch_size = 10, shuffle = True)

batch = next(iter(data_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow = 10)
plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1, 2, 0)))
plt.show()

# Exploratory Analysis
data_folders = os.listdir(data_root)
classes = []
class_sizes = []
for folder in data_folders:
    classes.append(folder)
    class_files = os.listdir(data_root + "/" + folder)
    class_sizes.append(len(class_files))

plt.bar(classes, class_sizes)
plt.show()