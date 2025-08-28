import torch

import os
import csv
import argparse
import numpy as np
from PIL import Image

from utilities import toTensor255, toPil, mkdir
from network import VisionTransformer, AdaAttnTransformerMultiHead
from eval import lpips_loss, ssim_loss, sifid, kl_loss, gram_loss, nth_order_moment, uniformity, average_entropy


IMAGE_SIZE = (512, 512)
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
    ("./contents/Cornell.jpg", "./styles/Untitled-1964.jpg"),
    ("./contents/Bird.jpg", "./styles/Sketch.jpg"),
    ("./contents/RiverBoat.jpg", "./styles/Blue-3.jpg"),
    ("./contents/Sailboat.jpg", "./styles/Another-colorful-world.jpg"),
    ("./contents/Streets.jpg", "./styles/Composition.jpg"),
    ("./contents/Tubingen.jpg", "./styles/Volga-Landscape.jpg"),
]

opt = argparse.Namespace(
    path0="./results/stylized.jpg",
    path1="./results/style.jpg",
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

    # Load images
    print("Loading images...")
    contents_styles = list()
    for content_path, style_path in CONTENT_STYLE_PAIR:
        c = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        c = toTensor255(c).unsqueeze(0).to(device)
        s = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)
        contents_styles.append([c, s, content_path, style_path])

    result = list()
    for i, (c, s, content_path, style_path) in enumerate(contents_styles):
        print(f"Processing {i + 1} ...")

        # Model inference
        with torch.no_grad():
            fc = vit_c(c)
            fs = vit_s(s)
            _, cs = adaFormer(fc, fs)
            cs = cs.clamp(0, 255)

        # Create output directory
        save_path = f"./results/{i + 1}"
        mkdir(save_path, delete_existing_files=True)

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
        sifid_content = sifid(opt, no_print=True)
        kl_c = kl_loss(opt, no_print=True)

        opt.path1 = style_save_path
        lpips_style = lpips_loss(opt, no_print=True)
        ssim_style = ssim_loss(opt, no_print=True)
        sifid_style = sifid(opt, no_print=True)
        kl_s = kl_loss(opt, no_print=True)
        gram = gram_loss(opt, no_print=True)
        moment = nth_order_moment(opt, no_print=True)
        uni = uniformity(opt, no_print=True)
        entropy = average_entropy(opt, no_print=True)

        # Append the results
        result.append(
            {
                "content": content_path,
                "style": style_path,
                "lpips_content": lpips_content,
                "ssim_content": ssim_content,
                "sifid_content": sifid_content,
                "kl_c": kl_c,
                "lpips_style": lpips_style,
                "ssim_style": ssim_style,
                "sifid_style": sifid_style,
                "kl_s": kl_s,
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
            "content": "average",
            "style": "average",
            "lpips_content": avg_result[0],
            "ssim_content": avg_result[1],
            "sifid_content": avg_result[2],
            "kl_c": avg_result[3],
            "lpips_style": avg_result[4],
            "ssim_style": avg_result[5],
            "sifid_style": avg_result[6],
            "kl_s": avg_result[7],
            "gram": avg_result[8],
            "moment": avg_result[9],
            "uniformity": avg_result[10],
            "entropy": avg_result[11],
        }
    )

    # Save the results to a CSV file
    with open("./results/results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "content",
            "style",
            "lpips_content",
            "ssim_content",
            "sifid_content",
            "kl_c",
            "lpips_style",
            "ssim_style",
            "sifid_style",
            "kl_s",
            "gram",
            "moment",
            "uniformity",
            "entropy",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in result:
            writer.writerow(row)
    print("Results saved to ./results/results.csv")
