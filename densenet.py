import torch
import torch.nn as nn
# from torchsummary import summary #not usable

# 2d conv as per paper (BN->RELU->Conv) Repeat 1x1 (bottleneck) then 3x3 (composite)
# bn_factor is set to be 4, in paper it writes 4k where k is growth rate
# bn_factor is the bottleneck layer (it serves to reduce the dimensionality before 3x3 conv)
# note that dimensionality reduction is not true at the first few layers of the FIRST dense block.
# k indicates the expected output of the feature map channels for each dense layer
# note that we need to separate the 1x1 (bn) and 3x3 conv (composite function) because 1x1 is used to concat features
class DenseLayer(nn.Module):
    def __init__(self, in_, bn_factor=4, growth_rate=32):
        super().__init__()

        self.bn_layer = nn.Sequential(
            nn.BatchNorm2d(in_),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_, bn_factor * growth_rate, kernel_size=1),
        )

        self.composite_layer = nn.Sequential(
            nn.BatchNorm2d(bn_factor * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_factor * growth_rate, growth_rate, kernel_size=3, padding=1),
        )

    # feed in the concated features (note: features are in the form of a Python List!)
    def bn_concat(self, concat_features):
        concated_features = torch.cat(concat_features, 1)
        out_ = self.bn_layer(concated_features)
        return out_

    # thus, when u call DenseLayer, you will feed in a list of features.
    def forward(self, x):
        x = self.bn_concat(x)
        x = self.composite_layer(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, in_, num_layers, bn_factor=4, growth_rate=32):
        super().__init__()

        self.num_layers = num_layers
        # Each feature, you will add i*k of feature maps (concat from preceding layers)
        # The first feature map (in_) you can set to anything.
        self.dense_layer = []
        for i in range(num_layers):
            self.dense_layer.append(
                DenseLayer(
                    in_ + i * growth_rate,
                    bn_factor,
                    growth_rate
                ))

    def forward(self, x):

        # Very important, the feature maps passed forward has to be appended!
        features = [x]  # Form a python list
        # now you go through the layers and append the feature maps as u go along
        for i in range(self.num_layers):
            new_features = self.dense_layer[i](features)
            features.append(new_features)  # as u append, later on in DenseLayer, it will concat.
        return torch.cat(features, 1)


class Transition(nn.Module):
    def __init__(self, in_, out_):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_, out_, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)


class DenseNet(nn.Module):
    def __init__(self, input_channels=64, dense_block_config=(6, 12, 24, 16), bn_factor=4, growth_rate=32,
                 num_classes=1000):
        super().__init__()

        self.base_features = nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # After every DenseBlock is a transition layer to downsample.
        self.Dense_and_Transition = []
        for i, num_layers in enumerate(dense_block_config):

            self.Dense_and_Transition.append(DenseBlock(input_channels, num_layers, bn_factor, growth_rate=32))
            # Each DenseBlock will have different input_channels due to transition layers downsampling channels too.
            # Author suggest to compress channels with the bottleneck layers (can refer to DenseNet-BC version).
            # I divided/compress the channels by half (you can experiment with different numbers)
            # Surprising results from paper is that the compressed versions performs better with less parameters.
            # remember that going to the next DenseBlock, your first channels now follows from the transition layers
            # Transition will take the outputs of DenseBlocks.
            output_channels = input_channels + num_layers * growth_rate  # amount of output_channels
            transition_output = output_channels // 2
            input_channels = transition_output  # update for next dense block (input to dense block is output from transition)

            # Note: at the last DenseBlock operation, there is no transition layer. I actually caught this by debugging the code
            #       and then realised that the paper clearly does not put it, ops.
            if i == len(dense_block_config) - 1:
                break  # avoid using transition layer at last DenseBlock
            self.Dense_and_Transition.append(Transition(output_channels, transition_output))

        # Linear layers
        self.classifier = nn.Sequential(
            nn.AvgPool2d(7),
            nn.Flatten(),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):

        x = self.base_features(x)
        for layer in self.Dense_and_Transition:
            x = layer(x)

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    num_classes = 1000

    x = torch.randn(batch_size, 3, 224, 224)

    # Configs as such (following paper), note that I am already compressing the channels by half at every transition.
    # - DenseNet-121 -> dense_block_config = (6, 12, 24, 16)
    # - DenseNet-169 -> dense_block_config = (6, 12, 32, 32)
    # - DenseNet-201 -> dense_block_config = (6, 12, 48, 32)
    # - DenseNet-161 -> dense_block_config = (6, 12, 36, 24), growth_rate=48

    model = DenseNet(input_channels=64, dense_block_config=(6, 12, 24, 16), bn_factor=4, growth_rate=32)
    output = model(x)
    print(output.shape)
    # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes
    print(f"Output shape: {output.shape}, batch size: {batch_size}, number of classes: {num_classes}")

    # Sadly, it seems that torchsummary don't work for Dense Connectivity set up.
    # summary(model, (3, 224, 224))
