import math, torch
import torch.nn as nn
import torch.nn.functional as F

from .down_up import make_down, make_up, make_block


MODEL_REGISTRY = {}
def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

# =========================
# unet_swt: Uncertainty-aware Beta Gate + Cross-Attention Skip Injector (CASI)
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

class BetaGate(nn.Module):
    """
    Produces gate g in (0,1) with temperature tau.
    Return g and pre-sigmoid logits for regularization.
    """
    def __init__(self, ch, reduction=4, tau=1.0):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.tau = tau
        self.net = nn.Sequential(
            nn.Conv2d(2*ch, mid, 1), nn.GELU(),
            nn.Conv2d(mid, ch, 1)
        )

    def forward(self, f_main, f_aux):
        z = torch.cat([f_main, f_aux], dim=1)
        logits = self.net(z) / self.tau
        g = torch.sigmoid(logits)  # (N,C,H,W)
        fused = g * f_main + (1 - g) * f_aux
        return fused, g, logits

class CASI(nn.Module):
    """
    Local-window cross-attention from aux (K,V) to main skip (Q).
    Reuses the same window size as SWT for locality/efficiency.
    """
    def __init__(self, ch, num_heads=4, win=8):
        super().__init__()
        self.h = num_heads
        self.win = win
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, s_main, a_aux):
        if a_aux.shape[-2:] != s_main.shape[-2:]:
            a_aux = F.interpolate(a_aux, size=s_main.shape[-2:], mode='bilinear', align_corners=False)
        q = self.q(s_main)
        k = self.k(a_aux)
        v = self.v(a_aux)

        # windowed attention
        qw, kw, vw = map(lambda t: window_partition(t, self.win), (q,k,v))  # (NW,C,wh,ww)
        NW, C, wh, ww = qw.shape
        HW = wh*ww
        d  = C // self.h

        def reshape(t):
            t = t.flatten(2).transpose(1,2)     # (NW, HW, C)
            return t.view(NW, HW, self.h, d).transpose(1,2)  # (NW, h, HW, d)

        qh, kh, vh = map(reshape, (qw,kw,vw))
        attn = (qh @ kh.transpose(-2,-1)) / math.sqrt(d)
        attn = attn.softmax(dim=-1)
        out  = attn @ vh                                  # (NW,h,HW,d)
        out  = out.transpose(1,2).reshape(NW, HW, C).transpose(1,2)
        out  = out.view(NW, C, wh, ww)
        out  = window_reverse(out, self.win, s_main.shape[2], s_main.shape[3])
        return s_main + self.proj(out)

@register_model('unet_swt')
class UNetSegHead_SWT(nn.Module):
    """
    - Shifted-Window Transformer stages (encoder/decoder)
    - Beta-gated cross-modal fusion before encoder
    - CASI to pass aux features into every skip
    """
    def __init__(self, base=24, depth=(1,1,2,1), win=4, heads=(2,2,2,2), gate_tau=1.0):
        super().__init__()
        d1,d2,d3,d4 = depth
        h1,h2,h3,h4 = heads

        self.inc_aux = make_block('base', 12, base)  # light stem
        self.inc_main = make_block('base', 3, base)

        # gate
        self.gate = BetaGate(base, tau=gate_tau)

        # Encoder
        self.down1 = make_down("base", base, base * 2)
        self.down2 = make_down("base", base * 2, base * 4)
        self.down3 = make_down("swt", base * 4, base * 8, depth=d3, win=win, heads=h4)
        self.down4 = make_down("swt", base * 8, base * 16, depth=d4, win=win, heads=h4)

        # Store aux features at each scale
        self.aux1 = make_down("base", base, base * 2)
        self.aux2 = make_down("base", base * 2, base * 4)
        self.aux3 = make_down("swt", base * 4, base * 8, depth=1, win=win, heads=h4)
        self.aux4 = make_down("swt", base * 8, base * 16, depth=1, win=win, heads=h4)

        self.up1 = make_up("swt", base * 16, base * 8, depth=1, win=win, heads=h4)
        self.up2 = make_up("swt", base * 8, base * 4, depth=1, win=win, heads=h4)
        self.up3 = make_up("base", base * 4, base * 2)
        self.up4 = make_up("base", base * 2, base)

        # CASI adapters for skips
        self.casi4 = CASI(base*8,  num_heads=h4, win=win)
        self.casi3 = CASI(base*4,  num_heads=h3, win=win)
        self.casi2 = CASI(base*2,  num_heads=h2, win=win)
        self.casi1 = CASI(base,    num_heads=h1, win=win)

        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):  # x: (N,15,H,W)
        assert x.ndim==4 and x.size(1)>=15
        xa, xm = x[:, :12], x[:, 12:15]

        f_aux0  = self.inc_aux(xa)   # (N,B,H,W)
        f_main0 = self.inc_main(xm)  # (N,B,H,W)

        x1, g, logits = self.gate(f_main0, f_aux0)  # fused + gate

        # main pyramid
        e2 = self.down1(x1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)

        # aux pyramid
        a2 = self.aux1(f_aux0)
        a3 = self.aux2(a2)
        a4 = self.aux3(a3)
        a5 = self.aux4(a4)

        # CASI-modulated skips
        y  = self.up1(e5, self.casi4(e4, a4))
        y  = self.up2(y,  self.casi3(e3, a3))
        y  = self.up3(y,  self.casi2(e2, a2))
        y  = self.up4(y,  self.casi1(x1, f_aux0))

        return self.outc(y), g, {"gate_name":'beta'}





MODEL_Registry = {'unet_swt':UNetSegHead_SWT}