import torch
import torch.nn as nn
from torch import linalg as LA

from typing import List

from .vit import VisionTransformer
from .conv import Decoder


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k):
        return self.softmax(torch.bmm(q, k))


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        """
        q:   (b, t, d)
        k:   (b, d, t)
        out: (b, t, t)
        """
        q_norm = LA.vector_norm(q, dim=-1, keepdim=True)
        k_norm = LA.vector_norm(k, dim=1, keepdim=True)
        s = torch.bmm(q, k) / torch.bmm(q_norm, k_norm) + 1
        a = s / s.sum(dim=-1, keepdim=True)
        return a


# Original AdaAttN without learnable parameters
class AdaAttnForLoss(nn.Module):
    def __init__(self, v_dim, qk_dim, activation="softmax"):
        super().__init__()
        self.norm_q = nn.InstanceNorm2d(qk_dim, affine=False)
        self.norm_k = nn.InstanceNorm2d(qk_dim, affine=False)
        self.norm_v = nn.InstanceNorm2d(v_dim, affine=False)

        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, c_x, s_x, c_1x, s_1x):
        # Q^T
        Q = self.norm_q(c_1x)
        b, _, h, w = Q.size()
        Q = Q.view(b, -1, h * w).permute(0, 2, 1)

        # K
        K = self.norm_k(s_1x)
        b, _, h, w = K.size()
        K = K.view(b, -1, h * w)

        # V^T
        V = s_x
        b, _, h, w = V.size()
        V = V.view(b, -1, h * w).permute(0, 2, 1)

        # A * V^T
        A = self.activation(Q, K)
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h, w = c_x.size()
        M = M.view(b, h, w, -1).permute(0, 3, 1, 2)
        S = S.view(b, h, w, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(c_x) + M


# Modified AdaAttN with learnable parameters
class AdaAttN(nn.Module):
    def __init__(self, qkv_dim, activation="softmax"):
        super().__init__()
        self.f = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.g = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.h = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.norm_q = nn.InstanceNorm2d(qkv_dim, affine=False)
        self.norm_k = nn.InstanceNorm2d(qkv_dim, affine=False)
        self.norm_v = nn.InstanceNorm2d(qkv_dim, affine=False)

        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, fc: torch.Tensor, fs: torch.Tensor, fcs: torch.Tensor):
        # Q^T
        Q = self.f(self.norm_q(fc))
        b, _, h, w = Q.size()
        Q = Q.view(b, -1, h * w).permute(0, 2, 1)

        # K
        K = self.g(self.norm_k(fs))
        b, _, h, w = K.size()
        K = K.view(b, -1, h * w)

        # V^T
        V = self.h(fs)
        b, _, h, w = V.size()
        V = V.view(b, -1, h * w).permute(0, 2, 1)

        # A * V^T
        A = self.activation(Q, K)
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h_c, w_c = fc.size()
        M = M.view(b, h_c, w_c, -1).permute(0, 3, 1, 2)
        S = S.view(b, h_c, w_c, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(fcs) + M


class AdaAttnMultiHead(nn.Module):
    def __init__(self, qkv_dim, num_heads, activation="softmax"):
        super().__init__()
        if qkv_dim % num_heads != 0:
            raise ValueError("qkv_dim 必須能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = qkv_dim // num_heads

        # Create conv and norm modules for each head.
        self.f_list = nn.ModuleList([nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1) for _ in range(num_heads)])
        self.g_list = nn.ModuleList([nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1) for _ in range(num_heads)])
        self.h_list = nn.ModuleList([nn.Conv2d(self.head_dim, self.head_dim, kernel_size=1) for _ in range(num_heads)])

        self.norm_q_list = nn.ModuleList([nn.InstanceNorm2d(self.head_dim, affine=False) for _ in range(num_heads)])
        self.norm_k_list = nn.ModuleList([nn.InstanceNorm2d(self.head_dim, affine=False) for _ in range(num_heads)])
        self.norm_v_out_list = nn.ModuleList([nn.InstanceNorm2d(self.head_dim, affine=False) for _ in range(num_heads)])

        # Apply a conv layer after concat.
        self.out_conv = nn.Conv2d(qkv_dim, qkv_dim, kernel_size=1)

        # Activation function
        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, fc: torch.Tensor, fs: torch.Tensor, fcs: torch.Tensor):
        # fc, fs and fcs: (b, qkv_dim, h, w)
        b, _, h, w = fc.size()
        out_heads = []
        for i in range(self.num_heads):
            # Split the input for each head, with each head having head_dim channels.
            fc_i = fc[:, i * self.head_dim : (i + 1) * self.head_dim, :, :]
            fs_i = fs[:, i * self.head_dim : (i + 1) * self.head_dim, :, :]
            fcs_i = fcs[:, i * self.head_dim : (i + 1) * self.head_dim, :, :]

            # Q^T
            Q = self.f_list[i](self.norm_q_list[i](fc_i))
            Q = Q.reshape(b, self.head_dim, h * w).permute(0, 2, 1)

            # K
            b_fs, _, h_fs, w_fs = fs_i.size()
            K = self.g_list[i](self.norm_k_list[i](fs_i))
            K = K.reshape(b, self.head_dim, h_fs * w_fs)

            # V^T
            V = self.h_list[i](fs_i)
            V = V.reshape(b, self.head_dim, h_fs * w_fs).permute(0, 2, 1)

            # A * V^T
            A = self.activation(Q, K)
            M = torch.bmm(A, V)

            # S
            Var = torch.bmm(A, V**2) - M**2
            S = torch.sqrt(Var.clamp(min=1e-6))  # (b, seq_len, head_dim)

            # Reshape M and S
            M = M.reshape(b, h, w, self.head_dim).permute(0, 3, 1, 2)
            S = S.reshape(b, h, w, self.head_dim).permute(0, 3, 1, 2)

            # S * V + M
            out = S * self.norm_v_out_list[i](fcs_i) + M
            out_heads.append(out)

        # Concatenate the results of all heads along the channel dimension.
        out_cat = torch.cat(out_heads, dim=1)  # (b, qkv_dim, h, w)

        # Pass through a 1x1 convolution layer after concatenation.
        out = self.out_conv(out_cat)
        return out


class AdaAttnTransformer(nn.Module):
    def __init__(self, num_layers: int = 3, qkv_dim: int = 512, activation: str = "softmax"):
        super().__init__()
        self.num_layers = num_layers

        self.adaAttNs = nn.ModuleList(
            [
                AdaAttN(
                    qkv_dim,
                    activation,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder = Decoder()

    def forward(self, fc: List[torch.Tensor], fs: List[torch.Tensor]) -> torch.Tensor:
        fcs = fc[0]
        for i in range(self.num_layers):
            fcs = self.adaAttNs[i](fc[i], fs[i], fcs)

        cs = self.decoder(fcs)
        return cs


class AdaAttnTransformerMultiHead(nn.Module):
    def __init__(self, num_layers: int = 3, qkv_dim: int = 512, num_heads: int = 8, activation: str = "softmax"):
        super().__init__()
        self.num_layers = num_layers

        self.adaAttnHead = nn.ModuleList(
            [
                AdaAttnMultiHead(
                    qkv_dim=qkv_dim,
                    num_heads=num_heads,
                    activation=activation,
                )
                for _ in range(num_layers * 2)
            ]
        )

        self.decoder = Decoder()

    def forward(self, fc: List[torch.Tensor], fs: List[torch.Tensor]) -> torch.Tensor:
        fcs = fc[0]
        for i in range(self.num_layers):
            fcs = self.adaAttnHead[2 * i](fc[i], fs[i], fcs)
            fcs = self.adaAttnHead[2 * i + 1](fcs, fs[i], fcs)

        cs = self.decoder(fcs)
        return fcs, cs


def test_AdaFormer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a random tensor and normalize it
    c = torch.rand(8, 3, 256, 256).to(device)
    s = torch.rand(8, 3, 256, 256).to(device)

    # Create a StylizingNetwork model and forward propagate the input tensor
    vit_c = VisionTransformer(pos_embedding=True).to(device)
    vit_s = VisionTransformer(pos_embedding=False).to(device)
    model = AdaAttnTransformerMultiHead().to(device)

    # Forward pass
    fc = vit_c(c)
    fs = vit_s(s)
    cs = model(fc, fs)
    print(c.shape)
    print(cs.shape)


if __name__ == "__main__":
    test_AdaFormer()
