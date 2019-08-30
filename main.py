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

# Perform exploratory analysis
data_folders = os.listdir(data_root)
classes = []
class_sizes = []
for folder in data_folders:
    classes.append(folder)
    class_files = os.listdir(data_root + "/" + folder)
    class_sizes.append(len(class_files))

plt.bar(classes, class_sizes)
plt.show()

# Define YOLO network
class YOLO(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size 448 x 448
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 7, stride = 2)
        # Input size 112 x 112
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 192, kernel_size = 3)
        # Input size 56 x 56
        self.conv3 = nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 1)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 1)
        self.conv6 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        # Input size 28 x 28
        self.conv7 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1)
        self.conv8 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        self.conv9 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1)
        self.conv10 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        self.conv11 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1)
        self.conv12 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        self.conv13 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1)
        self.conv14 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3)
        self.conv15 = nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 1)
        self.conv16 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3)
        # Input size 14 x 14
        self.conv17 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 1)
        self.conv18 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3)
        self.conv19 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 1)
        self.conv20 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3)
        self.conv21 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3)
        self.conv22 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, stride = 2)
        # Input size 7 x 7
        self.conv23 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3)
        self.conv24 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3)
        
        self.fc1 = nn.Linear(in_features = 1024 * 7 * 7, out_features = 4096)
        self.out = nn.Linear(in_features = 4096, out_features = 7 * 7 * 30)

    def forward(self, t):
        # Implement forward pass here
        return t