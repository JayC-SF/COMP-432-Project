import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClassicCNN, self).__init__()

        # Input (1, 128, 313)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # pooling & dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        # fully connected layers
        # after 3 maxpools, 128x313 becomes roughly 16x39
        self.fc1 = nn.Linear(128 * 16 * 39, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # flatten for the dense layers
        x = x.view(x.size(0), -1)

        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
