import torch
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table

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

    # Calculate FLOPs and Params using fvcore
    input_c = torch.randn(1, 3, 256, 256).to(device)
    fca_c = FlopCountAnalysis(vit_c, input_c)
    print(flop_count_table(fca_c))
    # ViT-C
    flops_c = fca_c.total()
    flops_c_g = flops_c / 1e9
    params_c = parameter_count(vit_c)[""] / 1e6
    print(f"FLOPs for ViT-C: {flops_c_g:.2f} GFlops", f"Parameters for ViT-C: {params_c:.2f} M")

    # ViT-S
    input_s = torch.randn(1, 3, 256, 256).to(device)
    fca_s = FlopCountAnalysis(vit_s, input_s)
    print(flop_count_table(fca_s))
    flops_s = fca_s.total()
    flops_s_g = flops_s / 1e9
    params_s = parameter_count(vit_s)[""] / 1e6
    print(f"FLOPs for ViT-S: {flops_s_g:.2f} GFlops", f"Parameters for ViT-S: {params_s:.2f} M")

    # AdaFormer
    ada_inputs = ada_input_constructor((512, 32, 32), device)
    fca_a = FlopCountAnalysis(adaFormer, ada_inputs)
    print(flop_count_table(fca_a))
    flops_a = fca_a.total()
    flops_a_g = flops_a / 1e9
    params_a = parameter_count(adaFormer)[""] / 1e6
    print(f"FLOPs for AdaFormer: {flops_a_g:.2f} GFlops", f"Parameters for AdaFormer: {params_a:.2f} M")
