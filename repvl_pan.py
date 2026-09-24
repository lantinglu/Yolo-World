import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class CSPLayerWithTwoConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 3):
        super().__init__()
        hidden = out_channels
        self.cv1 = ConvBNAct(in_channels, hidden, k=1, s=1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(num_blocks)])
        self.cv2 = ConvBNAct(hidden, out_channels, k=1, s=1)

    def forward(self, x):
        x = self.cv1(x)
        x = self.blocks(x)
        x = self.cv2(x)
        return x


class YOLOWorldPAFPN(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: List[int],
                 num_csp_blocks: int = 3,
                 upsample_mode: str = 'nearest'):
        super().__init__()
        assert len(in_channels) == 3 and len(out_channels) == 3
        c3, c4, c5 = in_channels
        p3, p4, p5 = out_channels

        self.reduce_layers = nn.ModuleList([
            ConvBNAct(c3, p3, k=1, s=1),
            ConvBNAct(c4, p4, k=1, s=1),
            ConvBNAct(c5, p5, k=1, s=1),
        ])

        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False) if upsample_mode=='bilinear'
            else nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False) if upsample_mode=='bilinear'
            else nn.Upsample(scale_factor=2, mode=upsample_mode),
        ])
        self.top_down_layers = nn.ModuleList([
            CSPLayerWithTwoConv(in_channels=p4 + p5, out_channels=p4, num_blocks=num_csp_blocks),
            CSPLayerWithTwoConv(in_channels=p3 + p4, out_channels=p3, num_blocks=num_csp_blocks),
        ])

        self.downsample_layers = nn.ModuleList([
            ConvBNAct(p3, p3, k=3, s=2),
            ConvBNAct(p4, p4, k=3, s=2),
        ])
        self.bottom_up_layers = nn.ModuleList([
            CSPLayerWithTwoConv(in_channels=p3 + p4, out_channels=p4, num_blocks=num_csp_blocks),
            CSPLayerWithTwoConv(in_channels=p4 + p5, out_channels=p5, num_blocks=num_csp_blocks),
        ])

        self.out_layers = nn.ModuleList([
            ConvBNAct(p3, p3, k=1, s=1),
            ConvBNAct(p4, p4, k=1, s=1),
            ConvBNAct(p5, p5, k=1, s=1),
        ])

    def forward(self, C3: torch.Tensor, C4: torch.Tensor, C5: torch.Tensor):
        P3_l = self.reduce_layers[0](C3)
        P4_l = self.reduce_layers[1](C4)
        P5_l = self.reduce_layers[2](C5)

        up5 = self.upsample_layers[0](P5_l)
        td4 = self.top_down_layers[0](torch.cat([P4_l, up5], dim=1))
        up4 = self.upsample_layers[1](td4)
        td3 = self.top_down_layers[1](torch.cat([P3_l, up4], dim=1))

        down3 = self.downsample_layers[0](td3)
        bu4   = self.bottom_up_layers[0](torch.cat([down3, td4], dim=1))
        down4 = self.downsample_layers[1](bu4)
        bu5   = self.bottom_up_layers[1](torch.cat([down4, P5_l], dim=1))

        P3 = self.out_layers[0](td3)
        P4 = self.out_layers[1](bu4)
        P5 = self.out_layers[2](bu5)
        return P3, P4, P5


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None: p = k//2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = nn.SiLU() if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = ConvBNAct(c, c, 1,1)
        self.cv2 = ConvBNAct(c, c, 3,1)
    def forward(self, x): return x + self.cv2(self.cv1(x))

class TextAdapter(nn.Module):
    def __init__(self, d_text: int, chans: List[int]):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_text, d_text),
            nn.ReLU()
        )
        self.g = nn.ModuleList([nn.Linear(d_text, c) for c in chans])
        self.b = nn.ModuleList([nn.Linear(d_text, c) for c in chans])
    def forward(self, T: torch.Tensor):
        tbar = T.mean(0)
        h = self.proj(tbar)
        outs=[]
        for gi, bi in zip(self.g, self.b):
            gamma = torch.sigmoid(gi(h))
            beta  = bi(h)
            outs.append((gamma, beta))
        return outs

class TCSPLayer(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.main = nn.Sequential(Bottleneck(c), Bottleneck(c))
        self.aux  = nn.Conv2d(c, c, 1,1,0, bias=True)
        self.bn   = nn.BatchNorm2d(c)
    def forward(self, x, gamma, beta):
        y = self.main(x)
        y = y + gamma.view(1,-1,1,1) * self.aux(x)
        y = self.bn(y + beta.view(1,-1,1,1))
        return F.silu(y)
    @torch.no_grad()
    def fold(self, gamma: torch.Tensor, beta: torch.Tensor):
        C, Cin = self.aux.weight.size(0), self.aux.weight.size(1)
        W = self.aux.weight.view(C, Cin) * gamma.view(-1,1)
        b = (self.aux.bias * gamma) if self.aux.bias is not None else torch.zeros(C, device=W.device)
        a = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        c = self.bn.bias - self.bn.running_mean * a
        c = c + beta
        Wf = (a.view(-1,1) * W).view(C, Cin, 1,1).contiguous()
        bf = (a*b + c).contiguous()
        eq = nn.Conv2d(Cin, C, 1,1,0, bias=True).to(W.device)
        eq.weight.copy_(Wf); eq.bias.copy_(bf)
        return eq.eval()

class IPooling(nn.Module):
    def __init__(self, d_text: int, chans: List[int]):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(sum(chans), d_text),
            nn.ReLU(),        
            nn.Linear(d_text, d_text)
        )
    def forward(self, feats: List[torch.Tensor], T: torch.Tensor):
        gs = [f.mean(dim=[2,3]) for f in feats]
        g  = torch.cat(gs, 1).mean(0)
        delta = self.fc(g)
        return F.normalize(T + delta, dim=-1)

class RepVLPANDual(nn.Module):
    def __init__(self, base_pan: nn.Module, c_chans: List[int], p_chans: List[int], d_text: int):
        super().__init__()
        self.base = base_pan
        self.tcsp_C = nn.ModuleList([TCSPLayer(c) for c in c_chans])
        self.tcsp_P = nn.ModuleList([TCSPLayer(c) for c in p_chans])
        self.txt_C  = TextAdapter(d_text, c_chans)
        self.txt_P  = TextAdapter(d_text, p_chans)
        self.ipool_C= IPooling(d_text, c_chans)
        self.ipool_P= IPooling(d_text, p_chans)
        self.eqC = None; self.eqP = None

    def _apply_tcsp(self, feats, layers, gates):
        out=[]
        for f, l, (g,b) in zip(feats, layers, gates):
            out.append(l(f, g, b))
        return out

    def forward(self, C3, C4, C5, T):
        gatesC = self.txt_C(T)
        C3h,C4h,C5h = self._apply_tcsp([C3,C4,C5], self.tcsp_C, gatesC)
        T1 = self.ipool_C([C3h,C4h,C5h], T)
        P3,P4,P5 = self.base(C3h,C4h,C5h)
        gatesP = self.txt_P(T1)
        P3h,P4h,P5h = self._apply_tcsp([P3,P4,P5], self.tcsp_P, gatesP)
        T2 = self.ipool_P([P3h,P4h,P5h], T1)
        return (P3h,P4h,P5h), T2

    @torch.no_grad()
    def fold_for_vocab(self, T_user: torch.Tensor):
        gatesC = self.txt_C(T_user); self.eqC=[]
        for layer,(g,b) in zip(self.tcsp_C,gatesC): self.eqC.append(layer.fold(g,b))
        gatesP = self.txt_P(T_user); self.eqP=[]
        for layer,(g,b) in zip(self.tcsp_P,gatesP): self.eqP.append(layer.fold(g,b))

    def forward_folded(self, C3,C4,C5):
        assert self.eqC and self.eqP, "call fold_for_vocab() first"
        C_feats=[C3,C4,C5]; Cout=[]
        for f, layer, eq in zip(C_feats, self.tcsp_C, self.eqC):
            y = layer.main(f) + eq(f)
            Cout.append(F.silu(y))
        C3h,C4h,C5h = Cout
        P3,P4,P5 = self.base(C3h,C4h,C5h)
        P_feats=[P3,P4,P5]; Pout=[]
        for f, layer, eq in zip(P_feats, self.tcsp_P, self.eqP):
            y = layer.main(f) + eq(f)
            Pout.append(F.silu(y))
        return tuple(Pout)
