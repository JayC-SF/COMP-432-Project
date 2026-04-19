import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ClassicCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
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
        self.dropout = nn.Dropout(dropout)

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

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class AudioCLSTM(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super(AudioCLSTM, self).__init__()

        # 1. Feature Extraction (CNN)
        # We use a slightly deeper CNN to extract high-level features from the spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces Mel-bins

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 2. Sequence Learning (LSTM)
        # input_size = (Channels * (Mel_bins / 8))
        # If your input is 128 Mel bins, after 3 MaxPools, height is 16.
        # 128 channels * 16 = 2048
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # 3. Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: [Batch, 1, Mel, Time]

        # CNN Phase
        x = self.cnn(x)  # [Batch, 128, Mel/8, Time/8]

        # Reshape for LSTM: [Batch, Time, Features]
        # We want the LSTM to walk along the 'Time' dimension
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # Move Time (w) to the second dimension
        x = x.view(b, w, c * h)                # Flatten channels and height into features

        # LSTM Phase
        out, (h_n, c_n) = self.lstm(x)

        # We take the output of the very last time step
        x = out[:, -1, :]

        # Final Classification
        return self.fc(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input shape doesn't match output shape, we need a 'shortcut' layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # This is the "Skip Connection"
        out = F.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet, self).__init__()
        self.in_channels = 64

        # Initial layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet Layers (blocks)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out