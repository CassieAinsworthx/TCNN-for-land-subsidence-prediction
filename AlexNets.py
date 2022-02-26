import torch.nn as nn

class AlexNet_4x4(nn.Module):
    def __init__(self):
        super(AlexNet_4x4, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=2,stride=1,padding=0 ), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            nn.Conv2d(128, 256, kernel_size=2,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=2,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

        )
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=1,stride=1,padding=0 ), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),

            nn.Conv2d(64, 256, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1,stride=1,padding=0),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
