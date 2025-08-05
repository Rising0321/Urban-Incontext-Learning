import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    """
    x: (B, N, D)
    shift, scale: (B, D)
    """
    # 添加一个维度，便于与 x 的维度对齐
    scale = torch.tanh(scale).unsqueeze(1)  # 限制 scale 大小，避免爆炸
    shift = shift.unsqueeze(1)
    return x * (1 + scale) + shift


class NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Projections for query, key, value
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, context_):
        B, T, C = x.size()  # Batch, Target sequence length, Embedding size
        context = context_.unsqueeze(0).expand(B, T, C)
        _, S, _ = context.size()  # Source sequence length

        # Compute query, key, value
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.k_proj(context).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)
        v = self.v_proj(context).view(B, S, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, S, hs)

        # Attention weights
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)

        # Attention output
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Final projection
        y = self.out_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.droupout = nn.Dropout(0.1)  # 添加Dropout层以防止过拟合

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.droupout(x)
        return x


class DiffusionBlock(nn.Module):
    def __init__(self, config, res_scale=1.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        )
        self.res_scale = res_scale
        # self.cross_attn = CrossAttention(config)
        # self.ln_3 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        # self.proj = nn.Linear(image_emb.shape[-1], config.n_embd) if image_emb is not None else None
        # self.image_emb = image_emb

    def forward(self, x, c):
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        x = x + self.res_scale * self.attn(modulate(self.ln_1(x), shift_msa, scale_msa))
        x = x + self.res_scale * self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        # x = x + self.res_scale * self.cross_attn(modulate(self.ln_3(x), shift_cross, scale_cross), self.proj(self.image_emb))
        return x


from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # 您的数据维度，例如 Urban Profiling 中区域的数量
    input_dim: int = 256  # 假设您的输入数据是256维的


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: [0, 1, 2, 3, ...]
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def build_mlp(n_embd, z_dim):
    return nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.SiLU(),
        nn.Linear(4 * n_embd, z_dim),
    )


# 模块4: 全新的主模型
class MaskFlowDiffusionGPT(nn.Module):
    def __init__(self, config: GPTConfig, region_emb=None):
        super().__init__()
        self.config = config

        # 1. 输入层: 将原始数据映射到n_embd维度
        self.x_embedder = nn.Linear(1, config.n_embd)

        self.encoder_depth = config.n_layer // 2
        # self.mask_embedder = nn.Parameter(torch.zeros(1, config.input_dim, config.n_embd))  # mask embedding
        # 2. 位置编码:
        self.pos_embed = nn.Parameter(torch.zeros(1, config.input_dim, config.n_embd))

        self.region_emb = region_emb

        # self.pos_proj = nn.Linear(region_emb.shape[-1], config.n_embd)

        self.pos_proj = build_mlp(config.n_embd, region_emb.shape[-1]) if region_emb is not None else None

        # pos_embed_np = get_1d_sincos_pos_embed(config.n_embd, np.arange(config.input_dim))
        # pos_embed_tensor = torch.from_numpy(pos_embed_np).float().unsqueeze(0) # (1, N, D)
        # self.register_buffer('pos_embed', pos_embed_tensor)

        # 3. 扩散时间t的嵌入器
        self.t_embedder = TimestepEmbedder(config.n_embd)

        # 4. Transformer主干: 使用我们新的DiffusionBlock
        res_scale = 1.0 / math.sqrt(config.n_layer)
        self.blocks = nn.ModuleList([DiffusionBlock(config, res_scale=res_scale) for _ in range(config.n_layer)])

        # 5. 输出层: 包含最终的LayerNorm和线性预测头
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )
        self.lm_head = nn.Linear(config.n_embd, 1, bias=True)
        self.mask_head = nn.Linear(config.n_embd, 1, bias=True)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # 一个简单的权重初始化
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # 初始化位置编码
        torch.nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x, t, mask=None):
        """
        前向传播
        :param x: 模型输入, 维度为 (B, N, 1), 其中N是区域数量.
        这个x是在训练循环中由原始数据、噪声和mask混合而成的.
        :param t: 扩散时间步, 维度为 (B,)
        """
        # 1. 准备输入: 数据嵌入 + 位置嵌入

        pos_emb = self.pos_embed  # + self.pos_proj(self.region_emb)

        x = self.x_embedder(x) + pos_emb  # (B, N, n_embd)

        # 2. 准备时间条件
        c = self.t_embedder(t)  # (B, n_embd)

        # 3. 通过Transformer主干
        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (B, N, n_embd)
            if (i + 1) == self.encoder_depth:
                zs = self.pos_proj(x.reshape(-1, self.config.n_embd)).reshape(x.shape[0], x.shape[1],
                                                                              -1)  # (B, N, z_dim)

        # 4. 通过输出层进行预测
        shift, scale = self.adaLN_modulation_final(c).chunk(2, dim=1)
        x = modulate(self.ln_f(x), shift, scale)
        predictions = self.lm_head(x).squeeze(-1)  # (B, N)
        mask_predictions = torch.sigmoid(self.mask_head(x).squeeze(-1))  # (B, N)

        return predictions, zs, mask_predictions
