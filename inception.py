'''
This script serves as a tutorial to building Inception-v1 (learn how to go wide!)
Reference Paper: https://arxiv.org/pdf/1409.4842.pdf

Very easy to implement for inception-v1 to get an idea on how to go wide. v2-v3 is just basically having different
branches but it is equally easy to implement, but the author never write down the individual branches depth (pre-concat)
for the inception blocks . You can easily edit the branches in InceptionBlock to suit your own experiments.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# After every convolution or activation operation, batch norm will be applied after it. It became a standard post-2015.
# to keep it to v1, batch norm has not been introduced yet, feel free to uncomment and compare the results. It helps you
# conclude the effectiveness of batch normalisation.
def standard_2dconv(in_, out_, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_, out_, kernel_size, stride=stride, padding=padding),
        #nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


# Scale our class block to be flexible to create all blocks.
# All you need to do is set up a list of channels number following Table 1 of the Paper (Go across the columns of Table)
# Pool channels is also in the num_channels list
class InceptionBlock(nn.Module):
    def __init__(self, in_, num_channels):
        super().__init__()

        self.branch_1x1 = standard_2dconv(in_, num_channels[0], kernel_size=1)

        self.stack_3x3 = nn.Sequential(
            standard_2dconv(in_, num_channels[1], kernel_size=1),
            standard_2dconv(num_channels[1], num_channels[2], kernel_size=3, padding=1),
        )

        self.stack_5x5 = nn.Sequential(
            standard_2dconv(in_, num_channels[3], kernel_size=1),
            standard_2dconv(num_channels[3], num_channels[4], kernel_size=5, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            standard_2dconv(in_, num_channels[5], kernel_size=1)
        )

    def forward(self, x):
        # First branch (1x1) - 64 channels
        branch1 = self.branch_1x1(x)
        # Second branch (reduce+3x3) - 128 channels
        branch2 = self.stack_3x3(x)
        # Third Branch (reduce+5x5) - 32 channels
        branch3 = self.stack_5x5(x)
        # Fourth Branch (pooling+1x1) - 32 channels
        branch4 = self.branch_pool(x)
        # Concat all branches in channel dimension (dim=1)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class InceptionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.base_extractor = nn.Sequential(
            standard_2dconv(3, 64, 7, stride=2),
            standard_2dconv(64, 64, 3, stride=2, padding=1),
            standard_2dconv(64, 192, 3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Follow Table 1 of Paper, inception (3a, 3b, 4a, 4b, 4c, 4d, 4e, 5a, 5b)
        # Open the paper to follow along with the code
        self.inception_block_3a = InceptionBlock(in_=192, num_channels=(64, 96, 128, 16, 32, 32))
        self.inception_block_3b = InceptionBlock(in_=256, num_channels=(128, 128, 192, 32, 96, 64))
        self.inception_block_4a = InceptionBlock(in_=480, num_channels=(192, 96, 208, 16, 48, 64))
        self.inception_block_4b = InceptionBlock(in_=512, num_channels=(160, 112, 224, 24, 64, 64))
        self.inception_block_4c = InceptionBlock(in_=512, num_channels=(128, 128, 256, 24, 64, 64))
        self.inception_block_4d = InceptionBlock(in_=512, num_channels=(112, 144, 288, 32, 64, 64))
        self.inception_block_4e = InceptionBlock(in_=528, num_channels=(256, 160, 320, 32, 128, 128))
        self.inception_block_5a = InceptionBlock(in_=832, num_channels=(256, 160, 320, 32, 128, 128))
        self.inception_block_5b = InceptionBlock(in_=832, num_channels=(384, 192, 384, 48, 128, 128))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1000)
        )

    def forward(self, x):
        # Look at Table 1 (exact flow, easy to follow)
        x = self.base_extractor(x)
        x = self.inception_block_3a(x)
        x = self.inception_block_3b(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception_block_4a(x)
        x = self.inception_block_4b(x)
        x = self.inception_block_4c(x)
        x = self.inception_block_4d(x)
        x = self.inception_block_4e(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception_block_5a(x)
        x = self.inception_block_5b(x)
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = F.dropout(x, 0.4)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    # input into inception-v1 is fixed at 229x229
    x = torch.randn(10, 3, 229, 229)
    model = InceptionNet()
    output = model(x)
    print(output.shape)
    summary(model, (3, 229, 229))
