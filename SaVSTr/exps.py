import torch

import csv
import argparse
import numpy as np
from PIL import Image

from utilities import toTensor255, toPil
from network import VisionTransformer, AdaAttnTransformerMultiHead
from eval import lpips_loss, ssim_loss, kl_loss, gram_loss, nth_order_moment, uniformity, average_entropy


IMAGE_SIZE = (256, 256)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"

MODEL_EPOCH = 20
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

CONTENT_STYLE_PAIR = [
    ("./contents/Chair.png", "./styles/Brushstrokes.png"),
    ("./contents/Brad-Pitt.png", "./styles/Sketch.png"),
    ("./contents/Bird.png", "./styles/Tableau.png"),
]

opt = argparse.Namespace(
    path0="./results/stylized.png",
    path1="./results/style.png",
    device="cuda",
)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)
    vit_c.load_state_dict(torch.load(VITC_PATH, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, weights_only=True), strict=True)
    adaFormer.load_state_dict(torch.load(ADA_PATH, weights_only=True), strict=True)
    vit_c.eval()
    vit_s.eval()
    adaFormer.eval()

    result = list()
    for c_s_pair in CONTENT_STYLE_PAIR:
        content_path, style_path = c_s_pair

        # Load images
        c = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        c = toTensor255(c).unsqueeze(0).to(device)
        s = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            fc = vit_c(c)
            fs = vit_s(s)
            cs = adaFormer(fc, fs)
            cs = cs.clamp(0, 255)

        # Save images
        toPil(c.squeeze(0).byte()).save("./results/content.png")
        toPil(s.squeeze(0).byte()).save("./results/style.png")
        toPil(cs.squeeze(0).byte()).save("./results/stylized.png")

        # Evaluate images
        result_temp = list()

        opt.path1 = "./results/content.png"
        result_temp.append(lpips_loss(opt, no_print=True))
        result_temp.append(ssim_loss(opt, no_print=True))

        opt.path1 = "./results/style.png"
        result_temp.append(lpips_loss(opt, no_print=True))
        result_temp.append(ssim_loss(opt, no_print=True))
        result_temp.append(kl_loss(opt, no_print=True))
        result_temp.append(gram_loss(opt, no_print=True))
        result_temp.append(nth_order_moment(opt, no_print=True))
        result_temp.append(uniformity(opt, no_print=True))
        result_temp.append(average_entropy(opt, no_print=True))

        result.append(result_temp)

    # Calculate average results
    avg_result = np.mean(result, axis=0)

    # Print average results
    print("-" * 50)
    print("Average Results:")
    print(f"LPIPS (content): {avg_result[0]}")
    print(f"SSIM (content): {avg_result[1]}")
    print(f"LPIPS (style): {avg_result[2]}")
    print(f"SSIM (style): {avg_result[3]}")
    print(f"KL Divergence: {avg_result[4]}")
    print(f"Gram Matrix: {avg_result[5]}")
    print(f"Nth Order Moment: {avg_result[6]}")
    print(f"Uniformity: {avg_result[7]}")
    print(f"Average Entropy: {avg_result[8]}")
    print("-" * 50)

    # Print results
    for i, c_s_pair in enumerate(CONTENT_STYLE_PAIR):
        content_path, style_path = c_s_pair
        print("-" * 50)
        print(f"Content: {content_path}, Style: {style_path}")
        print(f"LPIPS (content): {result[i][0]}")
        print(f"SSIM (content): {result[i][1]}")
        print(f"LPIPS (style): {result[i][2]}")
        print(f"SSIM (style): {result[i][3]}")
        print(f"KL Divergence: {result[i][4]}")
        print(f"Gram Matrix: {result[i][5]}")
        print(f"Nth Order Moment: {result[i][6]}")
        print(f"Uniformity: {result[i][7]}")
        print(f"Average Entropy: {result[i][8]}")
        print("-" * 50)

    # Save results to csv
    with open("results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Content",
                "Style",
                "LPIPS (content)",
                "SSIM (content)",
                "LPIPS (style)",
                "SSIM (style)",
                "KL Divergence",
                "Gram Matrix",
                "Nth Order Moment",
                "Uniformity",
                "Average Entropy",
            ]
        )
        for i, c_s_pair in enumerate(CONTENT_STYLE_PAIR):
            content_path, style_path = c_s_pair
            writer.writerow([content_path, style_path] + result[i])
        writer.writerow(["Average", ""] + avg_result.tolist())
    print("Results saved to results.csv")
