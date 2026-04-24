"""
RepVIT-MICA: MICA (Multivariate Infini Compressive Attention) adapted for RepVIT.

[논문] Potosnak et al., "MICA: Multivariate Infini Compressive Attention
        for Time Series Forecasting", arXiv 2604.06473v1, 2026.

─────────────────────────────────────────────────────────────────
[논문 → Vision 매핑]

원 논문 (time-series):
  입력 : Y ∈ R^{B × C × T},  C = 다변량 채널 수, T = 시간축
  Local : 각 채널 내 temporal tokens에 softmax dot-product attention
            A_local = softmax(QK^T / sqrt(d_k)) V           [Eq.3]
  Global: 모든 C 채널의 K-V를 선형적으로 압축 → cross-channel memory
            M = Σ_c φ(K^(c))^T V^(c)                       [Eq.4]
            z = Σ_c Σ_p φ(K_p^(c))                         [Eq.5]
            A_global = φ(Q)M / (φ(Q)z + ε)                 [Eq.6]
  Gate  : A_mixed = σ(β)⊙A_global + (1-σ(β))⊙A_local      [Eq.7]

본 구현 (vision feature map):
  입력 : X ∈ R^{B × dim × H × W},  dim = feature channels, HW = spatial
  Local : RepVGGDW (3×3 depthwise conv, 기존 RepVIT token mixer)
            ↳ 논문의 softmax attention 대신 경량 local spatial mixing 사용
            ↳ RepVIT 경량성 유지를 위한 의도적 선택
  Global: 모든 HW spatial positions의 K-V를 선형적으로 압축
            M = Σ_{hw} φ(K_{hw})^T V_{hw}   (Eq.4와 수학적 동일 구조)
            z = Σ_{hw} φ(K_{hw})             (Eq.5와 수학적 동일 구조)
            A_global = φ(Q)M / (φ(Q)z + ε)  (Eq.6 동일)
  Gate  : σ(β)⊙A_global + (1-σ(β))⊙A_local  (Eq.7 동일)

─────────────────────────────────────────────────────────────────
[충실도 엄격 평가]

구성요소           | 논문 구현               | 본 구현                  | 충실도
─────────────────────────────────────────────────────────────────
Global linear attn  | Σ_c φ(K^c)^T V^c        | Σ_{hw} φ(K_{hw})^T V_{hw} | ○ 동일 선형 구조
φ 함수             | ELU(x)+1                | ELU(x)+1                  | ○ 동일
β-gate             | σ(β)⊙·+(1-σ(β))⊙·      | 동일                       | ○ 동일
Local attention    | Softmax dot-product      | RepVGGDW (3×3 DW conv)    | △ 구조 다름
Cross-channel M    | C 채널 간 압축           | HW spatial 간 압축         | △ 도메인 변환
M의 의미           | 채널 간 공유 패턴        | Spatial global context     | △ semantics 변환

[핵심 차이점]
1. Local path: 논문은 softmax self-attention (O(HW²)), 본 구현은 DW conv (O(HW))
   → RepVIT 경량성 유지를 위한 설계 선택. "Local-MICA"가 아닌 "MICA-inspired" 수준.
2. Cross-dim 방향: 논문은 C개 채널을 합산, 본 구현은 HW positions를 합산
   → Vision feature map에서 C는 이미 semantically aggregated feature,
     HW는 논문의 "patch/token"에 해당 → 매핑 방향은 논리적으로 타당.
3. 결론: Global linear attention (핵심 기여)은 충실히 구현됨.
   Local부분은 RepVIT 경량성을 유지하기 위해 DW conv로 대체됨.
─────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

from mmdet.utils import get_root_logger
from mmdet.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import _load_checkpoint
from timm.models.layers import SqueezeExcite


# ─────────────────────────────────────────────────────────────────
#  기존 RepVIT 구성 요소 (원본 유지)
# ─────────────────────────────────────────────────────────────────

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups, w.size(0), w.shape[2:],
            stride=self.c.stride, padding=self.c.padding,
            dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device
            ).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert m.groups == m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert m.groups != m.in_channels
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1
        conv_w, conv_b = conv.weight, conv.bias
        conv1_w, conv1_b = conv1.weight, conv1.bias
        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])
        identity = torch.nn.functional.pad(
            torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1,
                       device=conv1_w.device), [1, 1, 1, 1])
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


# ─────────────────────────────────────────────────────────────────
#  [신규] VisionMICAAttention — MICA Global Linear Attention
# ─────────────────────────────────────────────────────────────────

class VisionMICAAttention(nn.Module):
    """
    MICA Global Linear Attention을 vision feature map에 적용.

    [논문 대응]
    - Eq.4: M = Σ_{hw} φ(K_{hw})^T V_{hw}   (spatial positions 압축)
    - Eq.5: z = Σ_{hw} φ(K_{hw})
    - Eq.6: A_global = φ(Q)M / (φ(Q)z + ε)
    - Eq.7: A_mixed = σ(β)⊙A_global + (1-σ(β))⊙A_local
      where A_local = RepVGGDW(x)  (논문은 softmax attn, 여기선 DW conv)

    입력 : x ∈ R^{B × dim × H × W}
    출력 : x ∈ R^{B × dim × H × W}

    파라미터 추가량 (dim=256, num_heads=4):
      QKV projection: 256 × 256 × 3 = 196,608
      out projection: 256 × 256    = 65,536
      β (gate):       num_heads    = 4
      총 ~262K (원본 RepViTBlock 256ch 블록 ~196K 대비 ~134% overhead)

    [복잡도]
    Global  : O(HW × head_dim²) per head → O(HW × dim) 전체
    Local   : O(HW × 9) (3×3 DW conv)
    합계    : O(HW × dim) — 논문과 동일한 선형 복잡도
    """

    def __init__(self, dim: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        # QKV projection (shared weight for efficiency)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

        # Learnable gate β — 논문 Eq.7, shared β across all positions
        # β ∈ R^{1 × num_heads × 1 × 1} (per-head, broadcast over B, HW)
        self.beta = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # Local path: RepVGGDW — 논문의 Local Attention 역할 (경량 대체)
        self.local = RepVGGDW(dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        # β 중앙화 초기화 (논문 Eq.8 참고: β_k ← β_k - E[β_k])
        nn.init.zeros_(self.beta)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """논문 φ(x) = ELU(x) + 1 (Eq.4~6)"""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W  # spatial tokens

        # ── Local path (A_local): 논문의 Local Attention 경량 대체 ──────
        # 논문: softmax(QK^T/sqrt(d_k))V  [Eq.3]
        # 본 구현: RepVGGDW (3×3 DW conv + reparameterization)
        a_local = self.local(x)                   # B × C × H × W

        # ── Global path (A_global): 논문 Eq.4~6 ──────────────────────
        qkv = self.qkv(x)                         # B × 3C × H × W
        Q, K, V = qkv.chunk(3, dim=1)             # each: B × C × H × W

        # reshape: B × C × H × W → B × num_heads × N × head_dim
        def reshape_for_heads(t):
            return t.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
            # → B × num_heads × N × head_dim

        Q = reshape_for_heads(Q)   # B × num_heads × N × head_dim
        K = reshape_for_heads(K)
        V = reshape_for_heads(V)

        phi_Q = self._phi(Q)       # B × num_heads × N × head_dim  (= d_k)
        phi_K = self._phi(K)       # B × num_heads × N × head_dim

        # M = Σ_{hw} φ(K_{hw})^T V_{hw}  [논문 Eq.4]
        # phi_K: B × num_heads × N × d_k
        # V    : B × num_heads × N × d_v  (d_k == d_v = head_dim)
        M = phi_K.transpose(-2, -1) @ V           # B × num_heads × d_k × d_v

        # z = Σ_{hw} φ(K_{hw})  [논문 Eq.5]
        z = phi_K.sum(dim=2, keepdim=True)         # B × num_heads × 1 × d_k

        # A_global = φ(Q)M / (φ(Q)z^T + ε)  [논문 Eq.6]
        num = phi_Q @ M                            # B × num_heads × N × d_v
        denom = (phi_Q @ z.transpose(-2, -1)) + self.eps  # B × num_heads × N × 1
        a_global = num / denom                     # B × num_heads × N × d_v

        # reshape: B × num_heads × N × head_dim → B × C × H × W
        a_global = a_global.permute(0, 1, 3, 2)   # B × num_heads × head_dim × N
        a_global = a_global.reshape(B, C, H, W)
        a_global = self.proj(a_global)

        # ── Mixing Gate: σ(β)⊙A_global + (1-σ(β))⊙A_local  [논문 Eq.7] ──
        # beta: 1 × num_heads × 1 × 1 → broadcast to B × num_heads × H × W
        # A_global/local: B × C × H × W → view as B × num_heads × head_dim × H × W 불가
        # 단순화: β를 채널 차원으로 broadcast
        # head별 β를 head_dim만큼 반복하여 C 차원에 매핑
        gate = torch.sigmoid(self.beta)            # 1 × num_heads × 1 × 1
        gate = gate.repeat(1, self.head_dim, 1, 1) # 1 × C × 1 × 1

        out = gate * a_global + (1.0 - gate) * a_local   # B × C × H × W
        out = self.norm(out)
        return out


# ─────────────────────────────────────────────────────────────────
#  [신규] RepViTBlock_MICA — MICA 선택적 적용 블록
# ─────────────────────────────────────────────────────────────────

class RepViTBlock_MICA(nn.Module):
    """
    기존 RepViTBlock의 SE 위치를 VisionMICAAttention으로 대체한 블록.

    with_mica=True  : SE 대신 VisionMICAAttention 삽입
    with_mica=False : 기존 RepViTBlock과 완전히 동일 (SE 또는 Identity)
    → 선택적 비교 실험 가능

    [삽입 위치 논리]
    SE (SqueezeExcite)와 MICA는 모두 "전역 채널/공간 정보"를 집약하는 역할.
    SE: GAP → FC → sigmoid scaling (position-independent)
    MICA Global: linear attention → query-driven retrieval (position-aware)
    MICA는 SE의 상위 일반화로 볼 수 있어 SE 위치 교체가 자연스러움.
    """

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride,
                 use_se, use_hs, with_mica=False, mica_heads=4):
        super().__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        assert hidden_dim == 2 * inp
        self.with_mica = with_mica

        if stride == 2:
            # 다운샘플링 블록: MICA 적용 안함 (해상도 변환 중 attention 비효율)
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride,
                          (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0),
            )
            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
            ))
            self.with_mica = False  # stride=2에선 강제 비활성

        else:
            assert self.identity
            if with_mica:
                # MICA 모드: token_mixer는 RepVGGDW만 (SE 제거, MICA가 대체)
                self.token_mixer = nn.Sequential(RepVGGDW(inp))
                # VisionMICAAttention이 local(RepVGGDW) + global + gate를 통합
                self.mica = VisionMICAAttention(inp, num_heads=mica_heads)
            else:
                # 기존 모드: RepVGGDW + (SE or Identity)
                self.token_mixer = nn.Sequential(
                    RepVGGDW(inp),
                    SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                )

            self.channel_mixer = Residual(nn.Sequential(
                Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                nn.GELU(),
                Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
            ))

    def forward(self, x):
        if self.with_mica:
            # MICA가 local+global+gate를 통합 처리
            x = self.mica(x)
            return self.channel_mixer(x)
        else:
            return self.channel_mixer(self.token_mixer(x))


# ─────────────────────────────────────────────────────────────────
#  RepViT_MICA backbone
# ─────────────────────────────────────────────────────────────────

class RepViT_MICA(nn.Module):
    """
    RepVIT backbone with optional MICA attention modules.

    mica_indices: MICA를 적용할 block index 목록
                  기본값: Stage3의 SE 블록 중 절반 (index 8, 12, 16)
                  빈 리스트([])로 설정 시 원본 RepVIT와 완전히 동일
    mica_heads  : VisionMICAAttention의 헤드 수
    """

    def __init__(self, cfgs, distillation=False, pretrained=None,
                 init_cfg=None, out_indices=[],
                 mica_indices=None, mica_heads=4):
        super().__init__()
        self.cfgs = cfgs
        self.mica_indices = set(mica_indices) if mica_indices is not None else set()
        self.mica_heads = mica_heads

        # Patch embedding (기존과 동일)
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(
            Conv2d_BN(3, input_channel // 2, 3, 2, 1),
            torch.nn.GELU(),
            Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1),
        )
        layers = [patch_embed]

        # Block 생성: MICA 인덱스에 해당하는 블록은 RepViTBlock_MICA로 대체
        for idx, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            with_mica = (idx in self.mica_indices) and (s == 1)

            # num_heads 결정: head_dim >= 16 보장
            heads = self.mica_heads
            while output_channel // heads < 16 and heads > 1:
                heads //= 2

            layers.append(RepViTBlock_MICA(
                inp=input_channel,
                hidden_dim=exp_size,
                oup=output_channel,
                kernel_size=k,
                stride=s,
                use_se=use_se,
                use_hs=use_hs,
                with_mica=with_mica,
                mica_heads=heads,
            ))
            input_channel = output_channel

        self.features = nn.ModuleList(layers)
        self.init_cfg = init_cfg
        assert self.init_cfg is not None
        self.out_indices = out_indices
        self.init_weights()
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()

    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for {self.__class__.__name__}')
            return
        assert 'checkpoint' in self.init_cfg
        ckpt_path = self.init_cfg['checkpoint'] if self.init_cfg else pretrained
        ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')

        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(f"Miss {missing}")
        logger.info(f"Unexpected {unexpected}")

    def train(self, mode=True):
        super(RepViT_MICA, self).train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.out_indices:
                outs.append(x)
        assert len(outs) == 4
        return outs


# ─────────────────────────────────────────────────────────────────
#  모델 등록: repvit_m1_1_mica
# ─────────────────────────────────────────────────────────────────

@BACKBONES.register_module()
def repvit_m1_1_mica(pretrained=False, distillation=False,
                     init_cfg=None, out_indices=[],
                     mica_indices=None, mica_heads=4, **kwargs):
    """
    RepViT-M1.1 with MICA attention.

    [MICA 삽입 위치 — repvit_m1_1 cfgs 기준 (index 0-base)]
    Block idx  | ch  | SE | stride | MICA 여부 (기본값)
    ──────────────────────────────────────────────────
    0          | 64  |  1 |   1    | ✗ (64ch, head_dim=16 최소)
    1,2        | 64  |  0 |   1    | ✗
    3          | 128 |  0 |   2    | ✗ (다운샘플)
    4          | 128 |  1 |   1    | ✗
    5,6        | 128 |  0 |   1    | ✗
    7          | 256 |  0 |   2    | ✗ (다운샘플)
    8          | 256 |  1 |   1    | ✓ ← MICA (Stage3 SE 블록 1)
    9          | 256 |  0 |   1    | ✗
    10         | 256 |  1 |   1    | ✓ ← MICA (Stage3 SE 블록 2)
    11         | 256 |  0 |   1    | ✗
    12         | 256 |  1 |   1    | ✓ ← MICA (Stage3 SE 블록 3)
    13~20      | 256 |mixed|  1    | ✗
    21         | 512 |  0 |   2    | ✗ (다운샘플)
    22         | 512 |  1 |   1    | ✓ ← MICA (Stage4 SE 블록)
    23         | 512 |  0 |   1    | ✗

    기본 mica_indices = [8, 10, 12, 22]
    → SE 블록 4개를 MICA로 교체 (Stage3 × 3 + Stage4 × 1)
    → 총 블록 24개 중 4개만 MICA (경량성 유지)

    [선택 전략]
    - Stage1/2 (64,128ch): 해상도 H/4, H/8 → spatial token수 많아 attention 비용 높음 → 제외
    - Stage3 (256ch, H/16): FPN P4 feature, semantic richness 높음 → 핵심 삽입 위치
    - Stage4 (512ch, H/32): FPN P5 feature, 가장 작은 spatial → MICA 비용 최소
    """
    if mica_indices is None:
        mica_indices = [8, 10, 12, 22]

    cfgs = [
        # k, t, c,   SE, HS, stride
        [3, 2,  64,  1,  0,  1],  # 0
        [3, 2,  64,  0,  0,  1],  # 1
        [3, 2,  64,  0,  0,  1],  # 2  ← out_index 2 (64ch, H/4)
        [3, 2,  128, 0,  0,  2],  # 3
        [3, 2,  128, 1,  0,  1],  # 4
        [3, 2,  128, 0,  0,  1],  # 5
        [3, 2,  128, 0,  0,  1],  # 6  ← out_index 6 (128ch, H/8)
        [3, 2,  256, 0,  1,  2],  # 7
        [3, 2,  256, 1,  1,  1],  # 8  ← MICA (SE→MICA)
        [3, 2,  256, 0,  1,  1],  # 9
        [3, 2,  256, 1,  1,  1],  # 10 ← MICA
        [3, 2,  256, 0,  1,  1],  # 11
        [3, 2,  256, 1,  1,  1],  # 12 ← MICA
        [3, 2,  256, 0,  1,  1],  # 13
        [3, 2,  256, 1,  1,  1],  # 14
        [3, 2,  256, 0,  1,  1],  # 15
        [3, 2,  256, 1,  1,  1],  # 16
        [3, 2,  256, 0,  1,  1],  # 17
        [3, 2,  256, 1,  1,  1],  # 18
        [3, 2,  256, 0,  1,  1],  # 19
        [3, 2,  256, 0,  1,  1],  # 20 ← out_index 20 (256ch, H/16)
        [3, 2,  512, 0,  1,  2],  # 21
        [3, 2,  512, 1,  1,  1],  # 22 ← MICA (SE→MICA)
        [3, 2,  512, 0,  1,  1],  # 23 ← out_index (512ch, H/32)
    ]
    return RepViT_MICA(
        cfgs=cfgs,
        init_cfg=init_cfg,
        pretrained=pretrained,
        distillation=distillation,
        out_indices=out_indices,
        mica_indices=mica_indices,
        mica_heads=mica_heads,
    )
