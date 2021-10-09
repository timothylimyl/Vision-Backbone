import torch
import torch.nn as nn
from torchsummary import summary


# Design choice: Simple architecture, just use nn.Sequential to group operations (easy to understand)
class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    num_classes = 1000

    x = torch.randn(batch_size, 3, 256, 256)
    model = AlexNet(num_classes)
    output = model(x)

    # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes

    print(f"Output shape: {output.shape}, batch size: {batch_size}, number of classes: {num_classes}")
    summary(model, (3, 256, 256))
