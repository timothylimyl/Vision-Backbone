'''
Purpose of the code is to show a simple cnn and how SPP are integrated to it.
'''

import torch
import torch.nn as nn
from torchsummary import summary


class SPP(nn.Module):
    def __init__(self, final_channels, pooling_scales):
        super().__init__()

        self.pool1 = nn.AdaptiveMaxPool2d(pooling_scales[0])
        self.pool2 = nn.AdaptiveMaxPool2d(pooling_scales[1])
        self.pool3 = nn.AdaptiveMaxPool2d(pooling_scales[2])

        self.flatten = nn.Flatten()

    def forward(self, x):
        a = self.flatten(self.pool1(x))
        b = self.flatten(self.pool2(x))
        c = self.flatten(self.pool3(x))

        return torch.cat([a, b, c], dim=1)


class SimpleSPP(nn.Module):
    def __init__(self, final_channels, scales, num_classes):
        super().__init__()

        self.simple_cnn_block = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, final_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.spp = SPP(final_channels, scales)

        # Calculate the output of the SPP layer (dependent only on channels and scales)
        fixed_spp_output = 0
        for i in scales:
            fixed_spp_output += final_channels * i ** 2

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fixed_spp_output, num_classes) # fc layer input is fixed by spp output size and not the size of the image.
        )

    def forward(self, x):
        x = self.simple_cnn_block(x)
        x = self.spp(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    num_classes = 1000

    # Now with SPP, you can change the input size and the network will still work
    h = 100
    w = 100
    x = torch.randn(batch_size, 3, h, w)

    final_channels = 100  # The thing you need to fix is the amount of channels for your layer prior to SPP
    scales = [1, 4, 8]  # and also fix the pooling scales that you want to use
    model = SimpleSPP(final_channels, scales, num_classes)
    output = model(x)

    print(output.shape)
    # # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes

    summary(model, (3, h, w))
