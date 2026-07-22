import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(

            DiscriminatorBlock(in_channels * 2, 64, normalize=False),

            DiscriminatorBlock(64, 128),

            DiscriminatorBlock(128, 256),

            nn.Conv2d(
                256,
                512,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(512),

            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, damaged, restored):

        x = torch.cat([damaged, restored], dim=1)

        return self.model(x)


if __name__ == "__main__":

    damaged = torch.randn(1, 3, 256, 256)

    restored = torch.randn(1, 3, 256, 256)

    model = Discriminator()

    output = model(damaged, restored)

    print("Damaged Shape :", damaged.shape)
    print("Restored Shape:", restored.shape)
    print("Output Shape  :", output.shape)