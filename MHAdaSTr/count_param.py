import torch

from network import VisionTransformer, AdaAttnTransformerMultiHead


NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"


def count_parameters_in_mb(model):
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_count * 4 / (1024 ** 2)  # 4 bytes per parameter, convert to MB


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)

    vit_c_params_mb = count_parameters_in_mb(vit_c)
    vit_s_params_mb = count_parameters_in_mb(vit_s)
    adaFormer_params_mb = count_parameters_in_mb(adaFormer)

    print(f"ViT-C Parameters: {vit_c_params_mb:.2f} MB")
    print(f"ViT-S Parameters: {vit_s_params_mb:.2f} MB")
    print(f"AdaFormer Parameters: {adaFormer_params_mb:.2f} MB")
