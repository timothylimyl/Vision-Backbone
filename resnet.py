'''
Purpose of code is to aid understanding of residual connections set up.
Code is only for ResNet18 and ResNet34 as it is more clearly indicated in the paper.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# Basic building blocks, just call this with different input and output channels accordingly to build up the network
# This is the block used for resnet18/34.
class ResidualBlock(nn.Module):

    def __init__(self, out_, strides=1, downsample=None):
        super().__init__()

        self.no_downsample = downsample is None  # Flag to check whether are we downsampling

        if self.no_downsample:
            in_ = out_  # non-downsampling layers has same channels dimension input and output
        else:
            in_ = int(out_ / 2)  # downsampling layers input channels is 2 times smaller than output channels.

        # Paper Notes: We adopt batch normalization (BN) right after each convolution and before activation
        self.block = nn.Sequential(
            nn.Conv2d(in_, out_, kernel_size=3, stride=strides, padding=1),
            nn.BatchNorm2d(out_),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_, out_, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_),
            nn.ReLU(inplace=True)
        )

        self.downsample = downsample

    def forward(self, x):
        identity = x  # following what the paper is calling it
        x = self.block(x)

        if not self.no_downsample:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, channel_configs, block_configs, num_classes):
        super().__init__()

        # Residual block setup: x3 64, x4 128, x6 256, x3 512 (34-layer setup as per paper)
        # First conv1 is 7x7, 64 channels, stride 2 (do not really fancy this personally)
        self.input_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.blocks = self._build_blocks(block, channels= channel_configs, num_blocks=block_configs)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.input_block(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x

    # I am making this function specifically for ResNet18/34 to be super explicit in the code.
    # I believe that it will be easier to follow and learn how to build 1 network type first and
    # then scale to another networks (easily doable too)
    def _build_blocks(self, block, channels, num_blocks):

        layers = []
        downsample = None

        for i, total_block in enumerate(num_blocks):  # Go through all the blocks

            # Note only the first block no need downsample (stride=1), the rest is stride=2
            strides = 2 if i != 0 else 1
            out_ = channels[i]  # the dimension of channels for the particular block

            for _ in range(total_block):
                layers += [block(out_, strides, downsample)]
                # Only first layer in new block need downsampling (all layers that follows are back to default setting)
                strides = 1
                downsample = None

            if i == len(
                    num_blocks) - 1:  # Basically if you are at the very last block, there's no need for downsampling anymore
                break

            # After first block, next blocks (for each starting layer) will need downsampling
            downsample = nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=1, stride=2),
                nn.BatchNorm2d(channels[i + 1])
            )

        return nn.Sequential(*layers)


if __name__ == "__main__":
    batch_size = 10
    num_classes = 1000

    x = torch.randn(batch_size, 3, 224, 224)

    # ResNet18 : block = ResidualBlock, channel_configs =[64, 128, 256, 512], block_configs = [2, 2, 2, 2]
    # ResNet34 : block = ResidualBlock, channel_configs =[64, 128, 256, 512], block_configs = [3, 4, 6, 3]
    resnet18 = [ResidualBlock, [64, 128, 256, 512], [2, 2, 2, 2], num_classes]
    resnet34 = [ResidualBlock, [64, 128, 256, 512], [3, 4, 6, 3], num_classes]

    model = ResNet(*resnet18)
    output = model(x)

    # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes
    print(f"Output shape: {output.shape}, batch size: {batch_size}, number of classes: {num_classes}")
    summary(model, (3, 224, 224))

    # Trying out tensorboard visualiser for PyTorch (`pip install tensorboard` to use)
    writer = SummaryWriter()
    writer.add_graph(model, x)
    writer.close()