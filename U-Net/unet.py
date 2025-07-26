import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.encoder2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.encoder3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(256, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder3 = ConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder2 = ConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final(dec1))