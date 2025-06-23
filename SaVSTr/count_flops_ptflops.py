import torch
from ptflops import get_model_complexity_info

from network import VisionTransformer, AdaAttnTransformerMultiHead


NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"


def ada_input_constructor(resolution, device="cuda"):
    return ([torch.randn(1, *resolution).to(device) for _ in range(NUM_LAYERS)], [torch.randn(1, *resolution).to(device) for _ in range(NUM_LAYERS)])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)
    vit_c.eval()
    vit_s.eval()
    adaFormer.eval()

    # Calculate FLOPs and Params
    flops_c, params_c = get_model_complexity_info(vit_c, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs for ViT-C: {flops_c}", f"Parameters for ViT-C: {params_c}")

    flops_s, params_s = get_model_complexity_info(vit_s, (3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print(f"FLOPs for ViT-S: {flops_s}", f"Parameters for ViT-S: {params_s}")

    flops_a, params_a = get_model_complexity_info(
        adaFormer,
        (512, 32, 32),
        as_strings=True,
        print_per_layer_stat=False,
        input_constructor=ada_input_constructor,
    )
    print(f"FLOPs for AdaFormer: {flops_a}", f"Parameters for AdaFormer: {params_a}")
