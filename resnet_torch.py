import torchvision.models as models
from torchsummary import summary
model = models.resnet18()

summary(model, (3, 224, 224))
