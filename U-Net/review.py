import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)
    

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        
        encoder_channels = [in_channels, 64, 128, 256]
        decoder_channels = [256, 128, 64]
        
        self.encoders = nn.ModuleList()
        for i in range(len(encoder_channels)-1):
            self.encoders.append(ConvBlock(encoder_channels[i], encoder_channels[i+1]))
        
        self.bottleneck = nn.Sequential(
            ConvBlock(encoder_channels[-1], 512),
            nn.Dropout(0.5)
        )
        
        self.upconvs = nn.ModuleList()
        for i in range(len(decoder_channels)):
            in_ch = 512 if i == 0 else decoder_channels[i-1]
            out_ch = decoder_channels[i]
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
        
        self.decoders = nn.ModuleList()
        decoder_in_channels = [512, 256, 128]
        for i in range(len(decoder_channels)):
            self.decoders.append(ConvBlock(decoder_in_channels[i], decoder_channels[i]))
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc_features = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_features.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            enc_feat = enc_features[-(i+1)]
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoders[i](x)
        
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x