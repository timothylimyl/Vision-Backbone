import torch
import torch.nn as nn
from torchsummary import summary


# Design choice: A lot of repetitive layers, good idea to set up common function for cleaner code.
# You will see that a lot of network designs have repetitive layers/blocks, so it is good practice to have a standard
# function/class.

def vgg_feature(*args):
    '''
    :param args: accept a list of [in_channels, out_channels] for standard 2d conv operation
    :return: feature layer sequence
    '''

    cnn_layers = []  # set up list for operations, later unpack to nn.Sequential method.
    for param in args:
        cnn_layers += [nn.Conv2d(param[0], param[1], kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    cnn_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*cnn_layers)


class vgg16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_layer1 = vgg_feature([3, 64], [64, 64])
        self.feature_layer2 = vgg_feature([64, 128], [128, 128])
        self.feature_layer3 = vgg_feature([128, 256], [256, 256], [256, 256])
        self.feature_layer4 = vgg_feature([256, 512], [512, 512], [512, 512])
        self.feature_layer5 = vgg_feature([512, 512], [512, 512], [512, 512])
        self.feature_extractor = [self.feature_layer1, self.feature_layer2, self.feature_layer3, self.feature_layer4,
                                  self.feature_layer5]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        for feature_extracting in self.feature_extractor:
            x = feature_extracting(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    num_classes = 10

    x = torch.randn(batch_size, 3, 224, 224)
    model = vgg16(num_classes)
    output = model(x)

    # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes
    print(f"Output shape: {output.shape}, batch size: {batch_size}, number of classes: {num_classes}")
    summary(model, (3, 224, 224))
