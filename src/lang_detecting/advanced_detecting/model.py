import math
from collections.abc import Sequence
from itertools import chain, repeat
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from more_itertools import take, windowed
from pydash import chain as c
from pydash import flow
from torch import Tensor

from src.lang_detecting.advanced_detecting.conf import Conf, ExpertConf
from src.lang_detecting.advanced_detecting.model_io_mging import KindToTargets, KindToVocab, Target, Vocab
from src.lang_detecting.advanced_detecting.tokenizer import MultiKindTokenizer


class MaskedLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | Sequence[int], *, eps: float = 1e-5, affine: bool = True, dim: int | Sequence[int] = None):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)
        dim = dim or list(range(-len(self.norm.normalized_shape), 0))
        dim = [dim] if isinstance(dim, int) else list(dim)
        if len(dim) != len(self.norm.normalized_shape):
            raise ValueError(f'Dimensions have to agree: {dim}, {list(self.norm.normalized_shape)}')
        self._dims: list[int] = dim
        self._perm: Optional[list[int]] = None
        self._inv_perm: Optional[list[int]] = None

    @property
    def _over_dims(self) -> list[int]:
        return list(range(-len(self._dims), 0))

    @property
    def _n_norm_dims(self) -> int:
        return len(self._dims)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        if self._perm is None:
            orig_dims = range(x.ndim)
            target_dims = sorted([d % x.ndim for d in self._dims])
            other_dims = sorted(set(orig_dims) - set(target_dims))
            self._perm = other_dims + target_dims
            self._inv_perm = c(list(enumerate(self._perm))).sort(key=lambda t: t[1]).map(lambda t: t[0]).value()
        is_masked = mask is not None
        mask = mask if is_masked else 1
        x = x.permute(self._perm)
        mask = mask.permute(self._perm) if is_masked else 1
        count = self._get_adjusted_count(x, mask) if is_masked else x.numel()
        sum_x = torch.sum(x * mask, dim=self._over_dims, keepdim=True)
        mean = sum_x / count

        var_sum = ((x - mean)**2 * mask).sum(dim=self._over_dims, keepdim=True)
        var = var_sum / count

        x_norm = (x - mean) / torch.sqrt(var + self.norm.eps)
        if self.norm.elementwise_affine:
            x_norm = x_norm * self.norm.weight + self.norm.bias
        x_norm = x_norm.permute(self._inv_perm)
        mask = mask.permute(self._inv_perm) if is_masked else 1
        return x_norm * mask

    def _get_adjusted_count(self, x: Tensor, mask: Tensor) -> Tensor:
        return (torch.ones(x.shape).to(x.device)*mask).sum(dim=self._over_dims, keepdim=True).clamp(min=1)


class ConvBlock(nn.Module):
    def __init__(self,
            channels: Sequence[int],
            kernels: Sequence[int],
            paddings: Sequence[int],
            *,
            act: nn.Module = None,
            scale: float = None,
        ) -> None:
        super().__init__()
        self.convs: nn.ModuleList[nn.Conv1d] = nn.ModuleList([
            nn.Conv1d(
                in_channels=ci,
                out_channels=co,
                kernel_size=k,
                padding=p,
            ) for (ci, co), k, p in zip(windowed(channels, 2), kernels, paddings)
        ])
        if scale is not None:
            with torch.no_grad():
                for conv in self.convs:
                    conv.weight *= scale
        self.act = act or nn.GELU()
        self.conv_norms = nn.ModuleList([
            MaskedLayerNorm((co,), dim=(-2, )) for co in channels[1:]
        ])

    def forward(self, x: Tensor, mask: Tensor = None) -> tuple[Tensor, Tensor]:
        for conv, norm in zip(self.convs, self.conv_norms):
            x = self.act(conv(x))  # B*ch x c_k x l_k
            mask = None if mask is None else F.max_pool1d(mask.float(), conv.kernel_size, conv.stride, conv.padding).int()
            effective_mask = None if mask is None else mask.repeat(x.size(0), 1).unsqueeze(1)
            x = norm(x, effective_mask)
        return x, mask

class Expert(nn.Module):
    def __init__(self,
            vocab: list[Vocab],
            targets: list[Target],
            all_labels: list[Target],
            conf: ExpertConf,
            n_specs: int = 0,
        ):
        super().__init__()
        self.funcs = {
            (LOGSUMEXP:='logsumexp'): torch.logsumexp,
            'relu': torch.relu,
            (SUM:='sum'): torch.sum,
            (SUM_NORM:='sum_norm'): lambda x, *args, **kwargs: self.attn_norm(torch.sum(x, **kwargs), *args),
            (MEAN:='mean'): torch.mean,
            (GATE:='gate'): None,
        }
        n_tokens, n_labels = len(vocab), len(all_labels)
        self.s_chunk = conf.chunking.size
        self.stride = conf.chunking.stride
        self.n_specs = n_specs
        n_emb = conf.n_emb
        self.embed = nn.Embedding(n_tokens, conf.n_emb - n_specs, padding_idx=conf.padding_idx)
        self.emb_dropout = nn.Dropout(p=conf.p_emb_dropout)
        self.pre_conv_norm = MaskedLayerNorm(n_emb, dim=-1)

        channels = (n_emb, *conf.hid_channels)
        n_channels = len(conf.hid_channels)
        kernels = take(n_channels, chain(conf.kernels, repeat(3)))
        paddings = take(n_channels, chain(conf.paddings, repeat(0)))
        self.skip_proj = nn.Conv1d(channels[0], channels[-1], 1)
        self.skip_norm = MaskedLayerNorm(channels[-1], dim=-2)
        nn.init.xavier_uniform_(self.skip_proj.weight)
        with torch.no_grad():
            self.skip_proj.weight *= .1
            self.skip_proj.bias *= .1
            for i in range(n_emb):
                self.skip_proj.weight[i, i, 0] = 1.0
        self.conv = ConvBlock(channels, kernels, paddings, act=nn.GELU(), scale=.1)
        self.conv_dropout = nn.Dropout(p=conf.p_conv_dropout)

        C = channels[-1]
        self.attn = nn.MultiheadAttention(
            embed_dim=C,
            num_heads=1,
            batch_first=True,
            dropout=conf.p_attn_dropout,
        )
        self.attn_norm = MaskedLayerNorm(C, dim=-1)
        self.post_attn_pool_name = SUM
        self.post_attn_pool = self.funcs[self.post_attn_pool_name]
        # self.ffn = nn.Sequential(
        #     nn.Linear(C, 2*C),
        #     nn.GELU(),
        #     nn.Linear(2*C, n_labels),
        # )
        self.ffn = nn.Linear(C, n_labels)
        output_mask = c(all_labels).map(targets.__contains__).map(int).value()
        self.register_buffer('output_mask', torch.tensor(output_mask, dtype=torch.float32))
        self.tokenizer: MultiKindTokenizer = conf.tokenizer

    def forward(self, words: Tensor, specs: Tensor) -> Tensor:
        """
        B - Batch size
        L - Length of the word with boundary padding
        e - Embedding dimension
        ch - Number of chunks
        l_k - Tensor length after k-th convolution [l_0 - Tensor length after chunking]
        c_k - k-th number of channels  [c_0 = e]
        """
        # words: B x L
        # specs: B x L x n_spec
        if words.min() < 0 or words.max() >= self.embed.num_embeddings:
            v_min, v_max, n_embed = words.min().item(), words.max().item(), self.embed.num_embeddings
            raise ValueError(f'{v_min=}, {v_max=}, {n_embed=}')

        x = self.emb_dropout(self.embed(words))  # B x L x e0
        x, mask = self._pad(x, -2, 0, return_mask=True)
        specs = self._pad(specs, -2, 0)
        x = self.cat_specs(x, specs)  # B x L x e1
        x, mask = self.chunk(x), self.chunk(mask)  # B x ch x e1 x l_0
        B, ch, C, L = x.shape
        x = x.reshape(B * ch, C, L)
        skip = self.skip_proj(x)
        effective_mask = mask.unsqueeze(1).repeat(B, skip.size(1), 1)
        skip = self.skip_norm(skip, effective_mask)
        res, mask = self.conv(x, mask)
        x = self.conv_dropout(skip + res)
        *_, C, L = x.shape
        x = x.permute(0, 2, 1).reshape(B, ch, L, C).flatten(1, 2)
        mask = mask.reshape(ch*L)
        effective_mask = mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, C)
        x = self.attn_norm(x, effective_mask)
        attn_out, _ = self.attn(x, x, x)  # B*ch x l_k x c_k
        # attn_out, _ = self.attn(x, x, x, key_padding_mask=mask.repeat(B, 1) == 0)  # B*ch x l_k x c_k
        attn_out = self.attn_norm(attn_out, effective_mask)
        x_attn = x + attn_out
        x = self.post_attn_pool(x_attn, dim=-2)  # B x c_k
        x = self.ffn(x)  # B x o
        # Mask non-expert outputs
        x = x.masked_fill(self.output_mask == 0, -1e9)  # set masked to very negative value
        return x

    def _pad(self, tensor: Tensor, dim: int = -1, pad_val: int = 0, *, return_mask: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        s_unpadded = tensor.shape[dim]
        n_chunks = math.ceil(s_unpadded / self.s_chunk)
        overlap = self.s_chunk - self.stride
        overlap_offset = overlap if n_chunks > 1 else 0
        n_to_pad = self.s_chunk * n_chunks - overlap_offset - s_unpadded
        from_end = -(dim if dim < 0 else dim - len(tensor.shape))
        initial_zeroes = (0, 0) * (from_end - 1)
        tensor = F.pad(tensor, (*initial_zeroes, 0, n_to_pad), value=pad_val)
        if return_mask:
            mask = torch.concat([
                tensor.new_ones(s_unpadded, dtype=torch.int),
                tensor.new_zeros(n_to_pad, dtype=torch.int)
            ], dim=0)
            return tensor, mask
        return tensor

    def cat_specs(self, x: Tensor, specs: Tensor) -> Tensor:
        return torch.cat([x, specs[..., : self.n_specs]], dim=-1)

    def chunk(self, tensor: Tensor) -> Tensor:
        dim = max(-2, -tensor.ndim)  # -2 for x, first
        tensor = tensor.contiguous()
        tensor = tensor.unfold(dimension=dim, size=self.s_chunk, step=self.stride)
        return tensor


class Moe(nn.Module):
    def __init__(self,
            kinds_to_vocabs: KindToVocab,
            kinds_to_targets: KindToTargets,
            kind_to_specs: dict[str, Sequence[Callable]],
            conf: Conf,
        ):
        if kinds_to_vocabs.keys() != kinds_to_targets.keys():
            raise ValueError(f'Incompatible kinds to vocab/targets mapping: {list(kinds_to_targets.keys())} {list(kinds_to_targets.keys())}')
        super().__init__()
        all_targets = c(kinds_to_targets.values()).flatten().apply(flow(set, sorted)).value()  # type: ignore[arg-type]
        self.n_classes = len(all_targets)
        self.experts: nn.ModuleList[Expert] = nn.ModuleList([
            Expert(vocabs, targets, all_targets, conf=conf.expert, n_specs=kind_to_specs.get(kind, 0),)
            for (kind, vocabs), targets in zip(kinds_to_vocabs.items(), kinds_to_targets.values())
        ])

    def forward(self, kinds: Tensor, words: Tensor, specs: Tensor) -> Tensor:
        out = torch.zeros(words.size(0), self.n_classes, device=words.device)
        specs = specs.to_dense()
        for expert_idx, expert in enumerate(self.experts):
            mask = kinds == expert_idx
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]
                out[idx] = expert(
                    words.index_select(0, idx),
                    specs.index_select(0, idx),
                )
        return out
