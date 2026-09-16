"""New 3D implementation; no code/weights imported from the other projects."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def norm(channels):
    return nn.GroupNorm(math.gcd(8, channels), channels)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(incoming, outgoing, 3, stride, 1, bias=False)
        self.norm1 = norm(outgoing)
        self.conv2 = nn.Conv3d(outgoing, outgoing, 3, 1, 1, bias=False)
        self.norm2 = norm(outgoing)
        self.shortcut = nn.Identity() if incoming == outgoing and stride == 1 else nn.Sequential(
            nn.Conv3d(incoming, outgoing, 1, stride, bias=False), norm(outgoing))

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(x)) + residual)


class Encoder(nn.Module):
    """18-layer residual topology, 3x3x3 stride-2 stem, no max-pool."""
    def __init__(self, width=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, width, 3, 2, 1, bias=False), norm(width), nn.ReLU())
        stages = []
        incoming = width
        for stage, outgoing in enumerate([width, 2*width, 4*width, 8*width]):
            stages.append(nn.Sequential(BasicBlock(incoming, outgoing, 1 if stage == 0 else 2),
                                        BasicBlock(outgoing, outgoing)))
            incoming = outgoing
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        features = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i > 0:
                features.append(x)
        return features


def position_3d(grid, dimension, device, dtype):
    """Fixed separable sinusoidal z/y/x coordinates on [-1,1].

    Concatenate axis sine/cosine channels, then truncate to embedding width.
    This is a new explicit engineering choice, not claimed pre-existing code.
    """
    axes = [torch.linspace(-1, 1, n, device=device, dtype=dtype) if n > 1
            else torch.zeros(1, device=device, dtype=dtype) for n in grid]
    coordinates = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    frequencies = torch.exp(torch.arange(math.ceil(dimension/6), device=device, dtype=dtype)
                            * (-math.log(10000)/max(math.ceil(dimension/6)-1, 1)))
    channels = []
    for axis in range(3):
        angles = math.pi * coordinates[:, axis:axis+1] * frequencies[None]
        channels.extend([angles.sin(), angles.cos()])
    return torch.cat(channels, dim=1)[:, :dimension][None]


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads, ff, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(dim, ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(ff, dim), nn.Dropout(dropout))

    def forward(self, x, maps=False):
        normalized = self.norm1(x)
        value, weights = self.attention(normalized, normalized, normalized,
                                       need_weights=maps, average_attn_weights=False)
        x = x + self.drop(value)
        x = x + self.ff(self.norm2(x))
        # B,H,Q,K -> B,K: received attention, NOT the constant key-row mean.
        received = weights.mean(dim=2).mean(dim=1) if maps else None
        return x, received


class ScaleTransformer(nn.Module):
    def __init__(self, channels, cfg):
        super().__init__()
        dim = cfg["embed_dim"]
        self.input_projection = nn.Conv3d(channels, dim, 1)
        self.output_projection = nn.Conv3d(dim, channels, 1)
        self.position = cfg["positional_encoding"]
        self.blocks = nn.ModuleList([AttentionBlock(dim, cfg["heads"], cfg["ff_dim"], cfg["dropout"])
                                     for _ in range(cfg["layers"])])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, maps=False):
        projected = self.input_projection(x)
        grid = projected.shape[-3:]
        tokens = projected.flatten(2).transpose(1, 2)
        if self.position:
            tokens = tokens + position_3d(grid, tokens.shape[-1], tokens.device, tokens.dtype)
        attention = []
        for block in self.blocks:
            tokens, received = block(tokens, maps)
            if maps:
                attention.append(received)
        result = self.final_norm(tokens).transpose(1, 2).reshape_as(projected)
        received = torch.stack(attention).mean(0).reshape(x.shape[0], 1, *grid) if maps else None
        return self.output_projection(result), received


class DetectorClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = dict(cfg)
        self.encoder = Encoder(cfg["width"])
        self.hybrid = cfg["kind"] == "hybrid"
        if cfg["kind"] not in ("cnn", "hybrid"):
            raise ValueError("Unknown classifier kind")
        channels = [cfg["width"]*v for v in (2, 4, 8)]
        self.transformers = nn.ModuleDict()
        self.local_fusion = nn.ModuleList()
        self.gates = nn.ModuleDict()
        if self.hybrid:
            for i in cfg["scales"]:
                if i not in (0, 1, 2):
                    raise ValueError("Scale must be 0, 1 or 2")
                self.transformers[str(i)] = ScaleTransformer(channels[i], cfg)
            for i, ch in enumerate(channels):
                incoming = ch if cfg["fusion"] == "gated" else ch*2
                self.local_fusion.append(nn.Sequential(nn.Conv3d(incoming, 128, 1, bias=False), norm(128), nn.ReLU()))
                if cfg["fusion"] == "gated":
                    self.gates[str(i)] = nn.Sequential(nn.Conv3d(2*ch, ch, 1), nn.Sigmoid())
            if cfg["fusion"] in ("hierarchical", "gated"):
                self.merge = nn.ModuleList([nn.Sequential(nn.Conv3d(256, 128, 1, bias=False), norm(128), nn.ReLU()) for _ in range(2)])
            elif cfg["fusion"] == "concat":
                self.concat = nn.Sequential(nn.Conv3d(384, 128, 1, bias=False), norm(128), nn.ReLU())
            else:
                raise ValueError("Unknown fusion")
        self.head = nn.Sequential(nn.Linear(128 if self.hybrid else channels[-1], 128),
                                  nn.ReLU(), nn.Dropout(cfg["head_dropout"]), nn.Linear(128, 2))

    def forward(self, x, explain=False):
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError("Expected B,1,Z,Y,X")
        features = self.encoder(x)
        attention = {}
        if not self.hybrid:
            fused = features[-1]
        else:
            local = []
            for i, feature in enumerate(features):
                key = str(i)
                if key in self.transformers:
                    transformed, received = self.transformers[key](feature, explain)
                    if explain:
                        attention[key] = received
                else:
                    transformed = torch.zeros_like(feature)
                pair = torch.cat([feature, transformed], dim=1)
                value = feature + self.gates[key](pair)*transformed if key in self.gates else pair
                local.append(self.local_fusion[i](value))
            if self.config["fusion"] == "concat":
                fine = local[0].shape[-3:]
                fused = self.concat(torch.cat([F.interpolate(f, size=fine, mode="trilinear", align_corners=False)
                                               for f in local], dim=1))
            else:
                fused = local[-1]
                for merge, i in zip(self.merge, (1, 0)):
                    up = F.interpolate(fused, size=local[i].shape[-3:], mode="trilinear", align_corners=False)
                    fused = merge(torch.cat([local[i], up], dim=1))
        logits = self.head(F.adaptive_avg_pool3d(fused, 1).flatten(1))
        return {"logits": logits, "features": features, "attention": attention} if explain else logits


def signature(model, size=48):
    """Measured shapes/counts and explicit operator MAC estimate, batch=1.

    MACs cover Conv3d/Linear and attention projections/matrix products, excluding
    normalization/activation/interpolation. They are not measured GPU latency.
    """
    shapes, macs, handles = {}, [0], []
    def hook(name):
        def capture(module, inputs, output):
            if isinstance(module, nn.Conv3d):
                macs[0] += output.numel()*module.in_channels//module.groups*math.prod(module.kernel_size)
            elif isinstance(module, nn.Linear):
                macs[0] += output.numel()*module.in_features
            elif isinstance(module, nn.MultiheadAttention):
                b, n, d = inputs[0].shape
                macs[0] += b*(4*n*d*d + 2*n*n*d)
            if name in ("encoder.stages.1", "encoder.stages.2", "encoder.stages.3"):
                shapes[name] = list(output.shape)
        return capture
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.Linear, nn.MultiheadAttention)) or name.startswith("encoder.stages.") and name.count(".") == 2:
            handles.append(module.register_forward_hook(hook(name)))
    previous = model.training
    try:
        model.eval()
        with torch.no_grad():
            logits = model(torch.zeros(1, 1, size, size, size, device=next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    return {"parameters": sum(p.numel() for p in model.parameters()), "input": [1, 1, size, size, size],
            "features": shapes, "output": list(logits.shape), "estimated_macs": macs[0],
            "normalization": "GroupNorm (gcd(8, channels) groups)",
            "stem": "3x3x3 stride 2 padding 1; no max-pool",
            "stage_strides": [1, 2, 2, 2], "blocks_per_stage": [2, 2, 2, 2],
            "cnn_receptive_fields_voxels": [47, 103, 215],
            "note": "CNN path receptive fields only; self-attention spans its entire token grid"}
