import torch
import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ELU(),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return x+self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=input_channels,
                         out_channels=input_channels),
            nn.ELU(),
            ResidualUnit(in_channels=input_channels,
                         out_channels=input_channels),
            nn.ELU(),
            ResidualUnit(in_channels=input_channels,
                         out_channels=input_channels),
            nn.ELU(),
            nn.Linear(input_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_channels,out_channels),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),

        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C=768, D=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
            # nn.Linear(768, 512),
            # nn.ELU(),
            # nn.Linear(512, 256),
            # nn.ELU(),
            # nn.Linear(256, 128),
            # nn.ELU(),
            # nn.Linear(128, 64),
            # nn.ELU(),
            # EncoderBlock(768, out_channels=512),
            # nn.Linear(768, 512),
            # # nn.ELU(),
            # EncoderBlock(512, out_channels=256),
            # nn.ELU(),
            # EncoderBlock(256, out_channels=128),
            # nn.ELU(),
            # EncoderBlock(128, out_channels=64),
            # nn.ELU(),
            # # EncoderBlock(64, out_channels=32),
            # # nn.ELU(),
            #
            # # EncoderBlock(256, out_channels=512),
            # # nn.ELU(),
            # # EncoderBlock(512, out_channels=512),
            # # nn.ELU(),
            # # EncoderBlock(512, out_channels=512),
            # # nn.ELU(),
            # # EncoderBlock(512, out_channels=512),
            # # nn.ELU(),
            # # EncoderBlock(512, out_channels=512),
            # # nn.ELU(),
            # nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            # nn.ELU(),
            # nn.Linear(64, 128),
            # nn.ELU(),
            # nn.Linear(128, 256),
            # nn.ELU(),
            # nn.Linear(256, 512),
            # nn.ELU(),
            # DecoderBlock(input_channels=32, out_channels=64),
            # nn.ELU(),
            # DecoderBlock(input_channels=64, out_channels=128),
            # nn.ELU(),
            # DecoderBlock(input_channels=32, out_channels=128),
            # nn.ELU(),
            # DecoderBlock(input_channels=128, out_channels=256),
            # nn.ELU(),
            # DecoderBlock(input_channels=256, out_channels=512),
            # nn.ELU(),
            # # DecoderBlock(input_channels=512, out_channels=768),
            # # # # DecoderBlock(256, out_channels=128),
            # # nn.ELU(),
            # # # DecoderBlock(128, out_channels=64),
            # # # nn.ELU(),
            # nn.Linear(512, 768)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    input = torch.zeros(16, 768).cuda()
    enc_model = Encoder(768, 96).cuda()
    dec_model = Decoder(768, 96).cuda()

    enc_feat = enc_model(input)
    dec_feat = dec_model(enc_feat)
    print("?")
