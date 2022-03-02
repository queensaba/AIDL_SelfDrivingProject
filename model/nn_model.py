import torch.nn as nn
import torch

class YoloV1Model(nn.Module):
    def __init__(self, channels=3, classes=13, boxes=2, grid_size=7):
        super(YoloV1Model, self).__init__()
        self.c = channels
        self.grid_size = grid_size
        self.classes = classes
        self.num_boxes = boxes  # Number of bounding boxes per cell
        # 448x448 input
        self.yolomodel = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7,
                      padding=1,
                      stride=2,
                      bias=False),  # 64x224x224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),  # -> 64x112x112

            nn.Conv2d(in_channels=64,
                      out_channels=192,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 192x112x112
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),  # -> 192x56x56

            nn.Conv2d(in_channels=192,
                      out_channels=128,
                      kernel_size=1,
                      bias=False),  # -> 192x56x56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 256x56x56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=1,
                      bias=False),  # -> 256x56x56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 512x56x56
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),  # -> 512x28x28

            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=1,
                      bias=False),  # -> 256x28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 512x28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=1,
                      bias=False),  # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 512x28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=1,
                      bias=False),  # -> 256x28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 51x28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=1,
                      bias=False),  # -> 256x28x28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 512x28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=1,
                      bias=False),  # -> 512x28x28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x28x28
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),  # -> 1024x14x14

            nn.Conv2d(in_channels=1024,
                      out_channels=512,
                      kernel_size=1,
                      bias=False),  # -> 512x14x14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x14x14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=1024,
                      out_channels=512,
                      kernel_size=1,
                      bias=False),  # -> 512x14x14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=512,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x14x14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x14x14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x14x14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),

            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x14x14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),
            nn.Conv2d(in_channels=1024,
                      out_channels=1024,
                      kernel_size=3,
                      padding=1,
                      bias=False),  # -> 1024x14x14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1,
                         inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(1024 * self.grid_size * self.grid_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, self.grid_size * self.grid_size * (self.num_classes + self.num_boxes * 5)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.yolomodel(x)
        x = torch.flatten(x,
                          start_dim=1)
        x = self.linear(x)
        return x





        
