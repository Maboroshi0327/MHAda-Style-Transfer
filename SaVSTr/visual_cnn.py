import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, models

from PIL import Image


def imageNet1k_normalize(batch: torch.Tensor):
    # normalize using imagenet mean and std
    batch = batch.float()
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(batch.device)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(batch.device)
    normalized_batch = (batch - mean) / std
    return normalized_batch


# ---- 1. Your VGG19 feature extractor ----
#    Assume imageNet1k_normalize and slice1~slice5 are defined
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(12, 21)])
        self.slice5 = nn.Sequential(*[vgg[x] for x in range(21, 30)])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = imageNet1k_normalize(x)
        r1 = self.slice1(x)
        r2 = self.slice2(r1)
        r3 = self.slice3(r2)
        r4 = self.slice4(r3)
        r5 = self.slice5(r4)
        return {"relu1_1": r1, "relu2_1": r2, "relu3_1": r3, "relu4_1": r4, "relu5_1": r5}


# ---- 2. Preparation ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG19().to(device).eval()

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
deprocess = transforms.Compose(
    [
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.ToPILImage(),
    ]
)

img = Image.open("./styles/Woman-with-Hat.jpg").convert("RGB")
orig = preprocess(img).unsqueeze(0).to(device)

# Extract target features from all five layers of the original image
with torch.no_grad():
    target_feats = model(orig)

# List of layer order
all_layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

# ---- 3. Progressive reconstruction loop ----
for k in range(1, len(all_layers) + 1):
    layers = all_layers[:k]
    print(f"\n--- Reconstruction matching layers: {layers} ---")

    # Initialize optimizable image with noise
    recon = torch.randn_like(orig, requires_grad=True, device=device)
    optimizer = optim.LBFGS([recon], max_iter=150, lr=1.0)

    iter_counter = [0]

    def closure():
        optimizer.zero_grad()
        feats = model(recon)
        # 把 k 層的 MSE loss 加總
        loss = sum(F.mse_loss(feats[l], target_feats[l]) for l in layers)
        loss.backward()
        iter_counter[0] += 1
        if iter_counter[0] % 50 == 0:
            print(f"  iter {iter_counter[0]:3d}  loss {loss.item():.4f}")
        return loss

    optimizer.step(closure)

    # Save result
    name = "_".join(layers)
    out = deprocess(recon.detach().cpu().squeeze())
    out.save(f"results/reconstructed_{name}.jpg")
    print(f"Saved → results/reconstructed_{name}.jpg")
