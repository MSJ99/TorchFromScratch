import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                bias=bias
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
    

    def forward(self, x):
        x = self.conv_bn_relu(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1_1 = Block(1, 64)
        self.enc1_2 = Block(64, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2_1 = Block(64, 128)
        self.enc2_2 = Block(128, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3_1 = Block(128, 256)
        self.enc3_2 = Block(256, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4_1 = Block(256, 512)
        self.enc4_2 = Block(512, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5_1 = Block(512, 1024)
        self.dec5_1 = Block(1024, 512)

        self.unpool4 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec4_2 = Block(2 * 512, 512)
        self.dec4_1 = Block(512, 256)

        self.unpool3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.dec3_2 = Block(2 * 256, 256)
        self.dec3_1 = Block(256, 128)

        self.unpool2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.dec2_2 = Block(2 * 128, 128)
        self.dec2_1 = Block(128, 64)

        self.unpool1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1_2 = Block(2 * 64, 64)
        self.dec1_1 = Block(64, 64)

        self.fc = nn.Conv2d(64, 2, 1)


    def forward(self, x):
        x = self.enc1_1(x)
        x = self.enc1_2(x)
        enc1_2 = x
        x = self.pool1(x)

        x = self.enc2_1(x)
        x = self.enc2_2(x)
        enc2_2 = x
        x = self.pool2(x)

        x = self.enc3_1(x)
        x = self.enc3_2(x)
        enc3_2 = x
        x = self.pool3(x)

        x = self.enc4_1(x)
        x = self.enc4_2(x)
        enc4_2 = x
        x = self.pool4(x)

        x = self.enc5_1(x)
        
        x = self.dec5_1(x)

        x = self.unpool4(x)
        x = torch.cat((x, enc4_2), dim=1)
        x = self.dec4_2(x)
        x = self.dec4_1(x)

        x = self.unpool3(x)
        x = torch.cat((x, enc3_2), dim=1)
        x = self.dec3_2(x)
        x = self.dec3_1(x)

        x = self.unpool2(x)
        x = torch.cat((x, enc2_2), dim=1)
        x = self.dec2_2(x)
        x = self.dec2_1(x)

        x = self.unpool1(x)
        x = torch.cat((x, enc1_2), dim=1)
        x = self.dec1_2(x)
        x = self.dec1_1(x)

        x = self.fc(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet().to(device)
    input_data = torch.randn(1, 1, 256, 256).to(device)

    output = model(input_data)

    print(f"Input Shape: {input_data.shape}")
    print(f"Output Shape: {output.shape}")