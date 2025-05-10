import torch

import os
import csv
import argparse
import numpy as np
from PIL import Image

from utilities import toTensor255, toPil
from utilities import list_files, mkdir
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

opt = argparse.Namespace(
    path0="./results/stylized.png",
    path1="./results/style.png",
    device="cuda",
)


def infer(vit_c, vit_s, adaFormer, content_path, style_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load images
    c = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    c = toTensor255(c).unsqueeze(0).to(device)
    s = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        fc = vit_c(c)
        fs = vit_s(s)
        _, cs = adaFormer(fc, fs)
        cs = cs.clamp(0, 255)

    return c, s, cs


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
    for i, content_path in enumerate(list_files("./contents")):
        for j, style_path in enumerate(list_files("./styles")):
            print(f"Processing content {i} and style {j}...")

            # Create output directory
            save_path = f"./results/content_{i}_style_{j}"
            mkdir(save_path, delete_existing_files=True)

            # Infer the images
            c, s, cs = infer(vit_c, vit_s, adaFormer, content_path, style_path)

            # Save the results
            content_save_path = os.path.join(save_path, "content.png")
            style_save_path = os.path.join(save_path, "style.png")
            stylized_save_path = os.path.join(save_path, "stylized.png")
            toPil(c[0].byte()).save(content_save_path)
            toPil(s[0].byte()).save(style_save_path)
            toPil(cs[0].byte()).save(stylized_save_path)

            # Evaluate the results
            opt.path0 = stylized_save_path
            opt.path1 = content_save_path
            lpips_content = lpips_loss(opt, no_print=True)
            ssim_content = ssim_loss(opt, no_print=True)

            opt.path1 = style_save_path
            lpips_style = lpips_loss(opt, no_print=True)
            ssim_style = ssim_loss(opt, no_print=True)
            kl = kl_loss(opt, no_print=True)
            gram = gram_loss(opt, no_print=True)
            moment = nth_order_moment(opt, no_print=True)
            uni = uniformity(opt, no_print=True)
            entropy = average_entropy(opt, no_print=True)

            # Append the results
            result.append(
                {
                    "content_idx": i,
                    "style_idx": j,
                    "lpips_content": lpips_content,
                    "ssim_content": ssim_content,
                    "lpips_style": lpips_style,
                    "ssim_style": ssim_style,
                    "kl": kl,
                    "gram": gram,
                    "moment": moment,
                    "uniformity": uni,
                    "entropy": entropy,
                }
            )

    # Calculate average results
    avg_result = np.mean([list(row.values())[2:] for row in result], axis=0)
    result.append(
        {
            "content_idx": "average",
            "style_idx": "average",
            "lpips_content": avg_result[0],
            "ssim_content": avg_result[1],
            "lpips_style": avg_result[2],
            "ssim_style": avg_result[3],
            "kl": avg_result[4],
            "gram": avg_result[5],
            "moment": avg_result[6],
            "uniformity": avg_result[7],
            "entropy": avg_result[8],
        }
    )

    # Save the results to a CSV file
    with open("results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "content_idx",
            "style_idx",
            "lpips_content",
            "ssim_content",
            "lpips_style",
            "ssim_style",
            "kl",
            "gram",
            "moment",
            "uniformity",
            "entropy",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in result:
            writer.writerow(row)
    print("Results saved to results.csv")
