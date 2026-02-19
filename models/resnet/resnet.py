import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

        if in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=bias,
                ),
                nn.BatchNorm2d(num_features=out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        L1 = self.layer1(x)
        L2 = self.layer2(L1)
        output = self.relu(L2 + self.shortcut(x))

        return output


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_1 = Block(64, 64)
        self.conv2_2 = Block(64, 64)
        self.conv2_3 = Block(64, 64)
        
        self.conv3_1 = Block(64, 128, stride=2)
        self.conv3_2 = Block(128, 128)
        self.conv3_3 = Block(128, 128)
        self.conv3_4 = Block(128, 128)
        
        self.conv4_1 = Block(128, 256, stride=2)
        self.conv4_2 = Block(256, 256)
        self.conv4_3 = Block(256, 256)
        self.conv4_4 = Block(256, 256)
        self.conv4_5 = Block(256, 256)
        self.conv4_6 = Block(256, 256)

        self.conv5_1 = Block(256, 512, stride=2)
        self.conv5_2 = Block(512, 512)
        self.conv5_3 = Block(512, 512)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def forward(self, x):
        C1 = self.pool1(self.conv1(x))
        C2 = self.conv2_3(self.conv2_2(self.conv2_1(C1)))
        C3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(C2))))
        C4 = self.conv4_6(self.conv4_5(self.conv4_4(self.conv4_3(self.conv4_2(self.conv4_1(C3))))))
        C5 = self.conv5_3(self.conv5_2(self.conv5_1(C4)))
        output = self.fc(torch.flatten(self.pool2(C5), 1))

        return output


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_data = torch.randn(1, 3, 256, 256)
    model = ResNet34()

    output = model(input_data)

    print(f"Input Shape: {input_data.shape}")
    print(f"Output Shape: {output.shape}")