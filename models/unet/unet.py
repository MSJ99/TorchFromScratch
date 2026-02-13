import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(Block, self).__init__()
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
        return self.conv_bn_relu(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
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
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_data = torch.randn(1, 1, 256, 256)
    model = UNet()

    output = model(input_data)

    print(f"Input Shape: {input_data.shape}")
    print(f"Output Shape: {output.shape}")