from torchvision import models
from torchsummary import summary
from arch import custom_resnet18
from arch import custom_resnet34
from arch import custom_resnet50
from arch import custom_resnet152
from torch import cuda


device = 'cuda' if cuda.is_available() else 'cpu'

cresnet18 = custom_resnet152().to(device)

summary(cresnet18, (1, 28, 28))