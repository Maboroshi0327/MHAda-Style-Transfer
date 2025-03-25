import torch
import torch.nn as nn
from torchvision.models import vgg19

from utilities import imageNet1k_normalize


class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        # Relu1_1
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])

        # Relu2_1
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])

        # Relu3_1
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])

        # Relu4_1
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])

        # Relu5_1
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])

        # Freeze all VGG parameters by setting requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = imageNet1k_normalize(x)
        x = self.slice1(x)
        relu1_1 = x
        x = self.slice2(x)
        relu2_1 = x
        x = self.slice3(x)
        relu3_1 = x
        x = self.slice4(x)
        relu4_1 = x
        x = self.slice5(x)
        relu5_1 = x

        features = {
            "relu1_1": relu1_1,
            "relu2_1": relu2_1,
            "relu3_1": relu3_1,
            "relu4_1": relu4_1,
            "relu5_1": relu5_1,
        }
        return features


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG19().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    features = model(x)
    for key, value in features.items():
        print(key, value.shape)
