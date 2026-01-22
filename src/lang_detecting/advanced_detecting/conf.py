from dataclasses import dataclass, field
from typing import Sequence, Optional


@dataclass
class ExpertConf:
    # TODO remove chunk info
    s_chunk: int = 7  # Suits well with the paddings to yield 3 in the last layer
    s_chunk_step: int = s_chunk // 2 + 1
    emb_dim: int = 32
    kernels: Sequence[int] = (3, 3, 3)
    hidden_channels: Sequence[int] = (32, 32)
    paddings: Sequence[int] =  (0, 1, 1)


@dataclass
class Conf:
    expert: ExpertConf = field(default_factory=ExpertConf)
    s_chunk: int = 7  # Suits well with the paddings to yield 3 in the last layer
    s_chunk_step: int = s_chunk // 2 + 1
    lr: float = 1e-4
    weight_decay = 1e-4
    max_batch_size: Optional[int] = None
    accum_grad_bs: int = 10
