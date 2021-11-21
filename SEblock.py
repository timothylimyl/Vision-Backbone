'''
Purpose of the code is to show a simple cnn and how SEBlock are integrated to it.
Comments adds some explanation to the network design.
'''
import torch
import torch.nn as nn
from torchsummary import summary

class SEBlock(nn.Module):
    def __init__(self, cnn_output_channels, r=2): # r=reduction ratio (as per paper)
        super().__init__()

        # Squeeze
        self.squeeze = nn.Sequential(
            nn.AvgPool2d(kernel_size=cnn_output_channels), # Get Global Statistic of Each Channel
            # basically you are taking the mean across HxW of the feature maps
            nn.Flatten(),
            nn.Linear(cnn_output_channels, cnn_output_channels // r),
            nn.ReLU(inplace=True)
        )

        # Excite
        self.excite = nn.Sequential(
            nn.Linear(cnn_output_channels // r, cnn_output_channels),
            nn.Sigmoid()
            # sigmoid is used because we want to learn channel-wise dependencies.
            # sigmoid function can emphasise MULTIPLE channels
            # if we use function like softmax, we are just learning to emphasising one channel from all of the channels given.
        )


    def forward(self, feature_map):
        bn_squeeze = self.squeeze(feature_map)
        channel_weights = self.excite(bn_squeeze)
        # Ta-Da you got the learnt channel weights (supposedly it can adaptively re-calibrate channel features)
        # Now you multiply it with the feature map.
        # Due to the dimension difference, I will squeeze dim HxW (1x1) for weights so that it can be broadcasted
        out_ = feature_map * channel_weights.unsqueeze(2).unsqueeze(3)

        return out_


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

        self.SEBlock = SEBlock(cnn_output_channels, r=2)


    # Normal typical forward propagation
    # def forward(self, x):
    #
    #     x = self.simple_cnn_block(x)
    #     x = self.linear(x)
    #     return x

    # Forward propagation with SEBlock
    def forward(self, x):
        feature_map = self.simple_cnn_block(x)
        out_ = self.SEBlock(feature_map)
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
