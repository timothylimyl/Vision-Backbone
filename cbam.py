'''
Purpose of the code is to show a simple cnn and how SEBlock are integrated to it.
Comments adds some explanation to the network design.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# Paper Equation: σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
class ChannelAttention(nn.Module):

    def __init__(self, in_, r=2): # reduction ratio (same as SEBlock)
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(nn.Flatten(),
                                 nn.Linear(in_, in_ // r),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(in_ // r, in_,),
                                 )

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        max = self.mlp(self.max_pool(x))
        out = F.sigmoid(avg + max)
        final_output = x * out.unsqueeze(2).unsqueeze(3)
        return final_output


# Paper Equation: σ(f7×7([AvgPool(F); MaxPool(F)]))
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding= (kernel_size-1) // 2)


    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max = torch.max(x, dim=1, keepdim=True)[0]
        pooled = torch.cat([avg, max], dim=1)
        spatial_out = self.spatial_conv(pooled)
        scaling = F.sigmoid(spatial_out)
        final_output = x * scaling
        return final_output


class SimpleCNN(nn.Module):
    def __init__(self, cnn_output_channels, num_classes):
        super().__init__()


        self.simple_cnn_block = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, cnn_output_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60 * 60 * cnn_output_channels, num_classes)
        )

        self.channel_attention = ChannelAttention(cnn_output_channels)
        self.spatial_attention = SpatialAttention()

    # Normal typical forward propagation
    # def forward(self, x):
    #
    #     x = self.simple_cnn_block(x)
    #     x = self.linear(x)
    #     return x

    # Forward propagation with SEBlock
    def forward(self, x):
        feature_map = self.simple_cnn_block(x)
        out_ = self.channel_attention(feature_map)
        out_ = self.spatial_attention(out_)
        out_ = self.linear(out_)
        return out_


if __name__ == "__main__":
    batch_size = 10
    num_classes = 10
    example_output_channels = 40
    x = torch.randn(batch_size, 3, 64, 64)
    model = SimpleCNN(example_output_channels, num_classes)
    output = model(x)

    print(output.shape)

    # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes

    print(f"Output shape: {output.shape}, batch size: {batch_size}, number of classes: {num_classes}")
    summary(model, (3, 64, 64))
