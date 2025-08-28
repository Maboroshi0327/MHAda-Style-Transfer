import torch
import torch.nn as nn
from torchvision.models import vgg19


def imageNet1k_normalize(batch: torch.Tensor):
    # normalize using imagenet mean and std
    batch = batch.float()
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(batch.device)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(batch.device)
    normalized_batch = (batch / 255.0 - mean) / std
    return normalized_batch


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


class VGG19_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights="VGG19_Weights.IMAGENET1K_V1")
        self.features = nn.Sequential()
        self.avgpool = vgg.avgpool
        self.classifier = vgg.classifier

        for x in range(30, 37):
            self.features.add_module(str(x), vgg.features[x])

        # Freeze all VGG parameters by setting requires_grad to False
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGG19().to(device)
    x = torch.randn(4, 3, 256, 256).to(device)
    features = model(x)
    for key, value in features.items():
        print(key, value.shape)

    model = VGG19_Classifier().to(device)
    x = features["relu5_1"]
    x = model(x)
    print(x.shape)
