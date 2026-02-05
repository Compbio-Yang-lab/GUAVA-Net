import math, torch
import torch.nn as nn
import torch.nn.functional as F



# =========================
# VAN DOWN / UP
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),)
    def forward(self, x): return self.block(x)

class VANDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), ConvBlock(in_ch, out_ch))
    def forward(self, x): return self.block(x)

class VANUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))


# =========================
# unet_swt
# =========================
def window_partition(x, win):
    # x: (N,C,H,W) -> (NW, C, wh, ww), NW = N * nH * nW
    N, C, H, W = x.shape
    assert H % win == 0 and W % win == 0
    x = x.view(N, C, H//win, win, W//win, win).permute(0,2,4,1,3,5)
    return x.reshape(-1, C, win, win)

def window_reverse(xw, win, H, W):
    # xw: (NW, C, wh, ww) -> (N,C,H,W)
    NW, C, wh, ww = xw.shape
    nH, nW = H//wh, W//ww
    N = NW // (nH * nW)
    x = xw.view(N, nH, nW, C, wh, ww).permute(0,3,1,4,2,5)
    return x.reshape(N, C, H, W)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hid = int(dim*mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hid, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid, dim, 1)
    def forward(self, x):  # (N,C,H,W)
        return self.fc2(self.act(self.fc1(x)))

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.h = num_heads
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

    def forward(self, xw):  # xw: (NW, C, wh, ww)
        NW, C, H, W = xw.shape
        qkv = self.qkv(xw)   # (NW, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # flatten spatial to tokens
        q = q.flatten(2).transpose(1,2)  # (NW, HW, C)
        k = k.flatten(2).transpose(1,2)
        v = v.flatten(2).transpose(1,2)
        # split heads
        d = C // self.h
        q = q.view(NW, -1, self.h, d).transpose(1,2)  # (NW, h, HW, d)
        k = k.view(NW, -1, self.h, d).transpose(1,2)
        v = v.view(NW, -1, self.h, d).transpose(1,2)

        attn = (q @ k.transpose(-2,-1)) / math.sqrt(d)  # (NW, h, HW, HW)
        attn = attn.softmax(dim=-1)
        out  = attn @ v                                 # (NW, h, HW, d)
        out  = out.transpose(1,2).reshape(NW, H*W, C)
        out  = out.transpose(1,2).reshape(NW, C, H, W)
        return self.proj(out)

class SWTBlock(nn.Module):
    # ---------------------------
    # Swin-style block (win attn + shift)
    # ---------------------------
    def __init__(self, dim, win=8, num_heads=4, shift=False, mlp_ratio=4):
        super().__init__()
        self.win = win
        self.shift = shift
        self.norm1 = nn.GroupNorm(1, dim)   # channel LN substitute
        self.attn  = WindowAttention(dim, num_heads)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp   = MLP(dim, mlp_ratio)

    def forward(self, x):  # (N,C,H,W)
        N,C,H,W = x.shape
        shortcut = x
        x = self.norm1(x)

        # cyclic shift (Swin trick)
        if self.shift:
            sh = self.win//2
            x = torch.roll(x, shifts=(-sh, -sh), dims=(2,3))

        # window attention
        xw = window_partition(x, self.win)
        xw = self.attn(xw)
        x  = window_reverse(xw, self.win, H, W)

        if self.shift:
            sh = self.win//2
            x = torch.roll(x, shifts=(+sh, +sh), dims=(2,3))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

def swt_stage(in_ch, out_ch, depth=2, win=8, heads=4):
    layers = [nn.Conv2d(in_ch, out_ch, 1)]
    for i in range(depth):
        layers += [SWTBlock(out_ch, win=win, num_heads=heads, shift=(i%2==1))]
    return nn.Sequential(*layers)

class SWTDown(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2, win=8, heads=4):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.body = swt_stage(in_ch, out_ch, depth, win, heads)
    def forward(self, x):
        return self.body(self.pool(x))

class SWTUp(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2, win=8, heads=4):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.fuse = nn.Conv2d(in_ch, out_ch, 1)
        self.body = swt_stage(out_ch, out_ch, depth, win, heads)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad for odd sizes
        diffY, diffX = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.body(self.fuse(torch.cat([x2, x1], dim=1)))






# ----------------------------- factory -----------------------------
DOWN_REGISTRY = {
    "base": VANDown,
    "swt":SWTDown# kwargs: depth, win, heads
}
UP_REGISTRY = {
    "base": VANUp,
    "swt": SWTUp
}
BLOCK_REGISTRY = {
    "base":ConvBlock,
    'swt':swt_stage
}


def make_down(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in DOWN_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(DOWN_REGISTRY.keys())}")
    return DOWN_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_up(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in UP_REGISTRY:
        raise ValueError(f"Unknown up type '{name}'. Available: {list(UP_REGISTRY.keys())}")
    if name == "poisson":
        skip_ch = kwargs.pop("skip_ch")
        return UP_REGISTRY[name](in_ch, skip_ch, out_ch, **kwargs)
    return UP_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_block(name: str, in_ch: int, out_ch: int, **kwargs):
    name = name.lower()
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[name](in_ch, out_ch, **kwargs)
# ----------------------------- sanity test -----------------------------
if __name__ == "__main__":
    # 与 convnext_stage 完全可互换
    stage = physics_stage(64, 128, depth=3, kind="heat", dt=0.15)  # 扩散
    stage = physics_stage(64, 128, depth=3, kind="pm", sigma=0.12, dt=0.2)  # PM 各向异性
    stage = physics_stage(64, 128, depth=3, kind="shock", dt=0.1)  # shock 锐化
    stage = physics_stage(64, 128, depth=3, kind="rd", a_init=0.0, b_init=0.1)  # 反应扩散
    stage = physics_stage(64, 128, depth=3, kind="helmholtz", alpha=0.6)  # 稳定平滑

    x = torch.randn(2, 32, 128, 128)

    # DOWN
    d0 = make_down("base", 32,64)


    # UP (single-skip)
    u_van = make_up("base", 64, 32)
    up = u_van(d0(x), torch.randn(2, 32, 128, 128)); print(up.shape)




    # UP (two-skip)
    # PoissonCrossUp expects two skips; you can wrap it in your Up class to pass (skip, deep_skip)
    #u_xp  = make_up("xpoisson", 64, 32, skip1_ch=32, skip2_ch=64, iters=6, tau=0.2, w=0.5)
    #up = u_xp(d7(x), torch.randn(2, 32, 128, 128), torch.randn(2, 64, 128, 128));print(up.shape)
