import torch
import torch.nn as nn
from torchsummary import summary


# Design choice: Simple architecture, just use nn.Sequential to group operations (easy to understand)
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    batch_size = 10
    num_classes = 1000
    x = torch.randn(batch_size, 3, 32, 32)
    model = LeNet5(num_classes)
    output = model(x)

    # Good sanity check to have for your output, expected output is [batch, class] size.
    assert output.shape[0] == batch_size and output.shape[1] == num_classes

    print(f"Output shape: {output.shape}, batch size: {batch_size}, number of classes: {num_classes}")
    summary(model, (3, 32, 32))
