"""CLEAR3D-MSF v2: Exact Center-Symmetric (33^3 @ 1mm + 25^3 @ 2mm) Multi-Scale 3D SE-ResNet
with Cross-Scale Self-Attention. Reflection & axial-rot90 symmetry preserve exact center index."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def maxpool2(x: torch.Tensor) -> torch.Tensor:
    b, c, d, h, w = x.shape
    return x.reshape(b, c, d // 2, 2, h // 2, 2, w // 2, 2).amax((3, 5, 7))


class SE3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean((2, 3, 4))
        w = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))[:, :, None, None, None]
        return x * w


class SEResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        g = 8 if cout % 8 == 0 else 4
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, cout)
        self.se = SE3D(cout)
        self.short = nn.Identity() if cin == cout else nn.Sequential(
            nn.Conv3d(cin, cout, 1, bias=False), nn.GroupNorm(g, cout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.short(x)
        h = F.silu(self.gn1(self.conv1(x)))
        h = self.se(self.gn2(self.conv2(h)))
        return F.silu(h + r)


class CLEAR3DMSFv2(nn.Module):
    """Center-symmetric 3-scale 3D architecture:
    - v_fine: (B, 3, 33, 33, 33) at 1.0 mm, exact candidate center at (16, 16, 16)
    - v_ctx:  (B, 3, 25, 25, 25) at 2.0 mm (~50 mm FOV), exact candidate center at (12, 12, 12)
    """

    def __init__(self, in_channels: int = 3, dim: int = 160, dropout: float = 0.25):
        super().__init__()
        # Scale 1 (Focal Core): central 21^3 -> Conv(k=2) -> 20^3 -> 10^3 -> 5^3
        self.f1_stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 2, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            SEResBlock3D(32, 32),
        )
        self.f1_s2 = SEResBlock3D(32, 64)
        self.f1_s3 = SEResBlock3D(64, 128)
        self.f1_proj = nn.Sequential(nn.Linear(32 + 32 + 64 + 128 + 128, dim), nn.LayerNorm(dim), nn.SiLU())

        # Scale 2 (Medium 33^3): Conv(k=2) -> 32^3 -> 16^3 -> 8^3 -> 4^3
        self.f2_stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 2, padding=0, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        self.f2_s1 = SEResBlock3D(32, 64)   # 16^3
        self.f2_s2 = SEResBlock3D(64, 96)   # 8^3
        self.f2_s3 = SEResBlock3D(96, 128)  # 4^3
        self.f2_proj = nn.Sequential(nn.Linear(64 + 96 + 128 + 128, dim), nn.LayerNorm(dim), nn.SiLU())

        # Scale 3 (Context 25^3 @ 2mm = 50mm FOV): Conv(k=2) -> 24^3 -> 12^3 -> 6^3 -> 3^3
        self.f3_stem = nn.Sequential(
            nn.Conv3d(in_channels, 24, 2, padding=0, bias=False),
            nn.GroupNorm(8, 24),
            nn.SiLU(),
        )
        self.f3_s1 = SEResBlock3D(24, 48)   # 12^3
        self.f3_s2 = SEResBlock3D(48, 96)   # 6^3
        self.f3_s3 = SEResBlock3D(96, 128)  # 3^3
        self.f3_proj = nn.Sequential(nn.Linear(96 + 128 + 128, dim), nn.LayerNorm(dim), nn.SiLU())

        self.scale_emb = nn.Parameter(torch.zeros(1, 3, dim))
        nn.init.normal_(self.scale_emb, std=0.02)
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=4, dropout=0.1, batch_first=True)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Dropout(0.1), nn.Linear(dim * 2, dim))

        self.head = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

    def forward(self, v_fine: torch.Tensor, v_ctx: torch.Tensor) -> torch.Tensor:
        # Scale 1: central 21^3 around index 16 -> [6:27]
        x1 = v_fine[:, :, 6:27, 6:27, 6:27]
        h1_1 = self.f1_stem(x1)                    # (B, 32, 20, 20, 20)
        h1_2 = self.f1_s2(maxpool2(h1_1))          # (B, 64, 10, 10, 10)
        h1_3 = self.f1_s3(maxpool2(h1_2))          # (B, 128, 5, 5, 5)
        t1 = self.f1_proj(torch.cat([
            h1_1[:, :, 6:14, 6:14, 6:14].mean((2, 3, 4)),  # inner 8mm core
            h1_1.mean((2, 3, 4)),
            h1_2.mean((2, 3, 4)),
            h1_3.mean((2, 3, 4)),
            h1_3.amax((2, 3, 4)),
        ], dim=1))

        # Scale 2: 33^3 -> 32^3 -> 16^3 -> 8^3 -> 4^3
        h2_0 = self.f2_stem(v_fine)
        h2_1 = self.f2_s1(maxpool2(h2_0))
        h2_2 = self.f2_s2(maxpool2(h2_1))
        h2_3 = self.f2_s3(maxpool2(h2_2))
        t2 = self.f2_proj(torch.cat([
            h2_1[:, :, 4:12, 4:12, 4:12].mean((2, 3, 4)),
            h2_2.mean((2, 3, 4)),
            h2_3.mean((2, 3, 4)),
            h2_3.amax((2, 3, 4)),
        ], dim=1))

        # Scale 3: 25^3 -> 24^3 -> 12^3 -> 6^3 -> 3^3
        h3_0 = self.f3_stem(v_ctx)
        h3_1 = self.f3_s1(maxpool2(h3_0))
        h3_2 = self.f3_s2(maxpool2(h3_1))
        h3_3 = self.f3_s3(maxpool2(h3_2))
        t3 = self.f3_proj(torch.cat([
            h3_2.mean((2, 3, 4)),
            h3_3.mean((2, 3, 4)),
            h3_3.amax((2, 3, 4)),
        ], dim=1))

        tokens = torch.stack([t1, t2, t3], dim=1) + self.scale_emb
        normed = self.ln_attn(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ff(self.ln_ff(tokens))
        return self.head(tokens.flatten(1))


def build_focus_grids(device: torch.device):
    g33 = torch.arange(33, device=device, dtype=torch.float32) - 16.0
    zz, yy, xx = torch.meshgrid(g33, g33, g33, indexing="ij")
    f_fine = torch.exp(-(zz * zz + yy * yy + xx * xx) / (2.0 * 4.5 * 4.5))[None, None]

    g25 = (torch.arange(25, device=device, dtype=torch.float32) - 12.0) * 2.0
    zz2, yy2, xx2 = torch.meshgrid(g25, g25, g25, indexing="ij")
    f_ctx = torch.exp(-(zz2 * zz2 + yy2 * yy2 + xx2 * xx2) / (2.0 * 10.0 * 10.0))[None, None]
    return f_fine, f_ctx


def make_inputs(fine_img: torch.Tensor, fine_val: torch.Tensor,
                ctx_img: torch.Tensor, ctx_val: torch.Tensor,
                f_fine: torch.Tensor, f_ctx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = fine_img.shape[0]
    fval = fine_val.bool()
    cval = ctx_val.bool()

    cnt = fval.flatten(1).sum(1, keepdim=True).clamp_min(1).float()
    mean = (fine_img * fval).flatten(1).sum(1, keepdim=True) / cnt
    var = (((fine_img - mean[:, :, None, None]) ** 2) * fval).flatten(1).sum(1, keepdim=True) / cnt
    std = var.sqrt().clamp_min(1e-3)
    mean4 = mean[:, :, None, None]
    std4 = std[:, :, None, None]

    f_hu = torch.where(fval, fine_img * 2.0 - 1.0, torch.full_like(fine_img, -1.0))
    f_z = torch.where(fval, ((fine_img - mean4) / std4).clamp(-5.0, 5.0), torch.zeros_like(fine_img))
    v_fine = torch.cat([f_hu[:, None], f_z[:, None], f_fine.expand(b, -1, -1, -1, -1)], dim=1)

    c_hu = torch.where(cval, ctx_img * 2.0 - 1.0, torch.full_like(ctx_img, -1.0))
    c_z = torch.where(cval, ((ctx_img - mean4) / std4).clamp(-5.0, 5.0), torch.zeros_like(ctx_img))
    v_ctx = torch.cat([c_hu[:, None], c_z[:, None], f_ctx.expand(b, -1, -1, -1, -1)], dim=1)
    return v_fine, v_ctx


def log_odds_tta8(model: nn.Module, v_fine: torch.Tensor, v_ctx: torch.Tensor) -> torch.Tensor:
    """8-view deterministic 3D reflection symmetry average of class log-odds (logit1 - logit0)."""
    acc = torch.zeros(v_fine.shape[0], device=v_fine.device, dtype=torch.float32)
    for fz in (False, True):
        for fy in (False, True):
            for fx in (False, True):
                dims = [d for d, flag in ((2, fz), (3, fy), (4, fx)) if flag]
                xf = torch.flip(v_fine, dims) if dims else v_fine
                xc = torch.flip(v_ctx, dims) if dims else v_ctx
                logits = model(xf, xc).float()
                acc = acc + (logits[:, 1] - logits[:, 0])
    return acc * 0.125
