from dataclasses import dataclass, field
from typing import Sequence, Optional

@dataclass
class Data:
    input_len_thresh: int = 3
    record_count_thresh: int = 2**4

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
    data: Data = field(default_factory=Data)
    expert: ExpertConf = field(default_factory=ExpertConf)
    epochs: int = 2**8
    lr: float = 1e-4  # 1e-5  # 1e-3
    weight_decay = 1e-4  # 1e-4
    max_batch_size: Optional[int] = 2**12
    accum_grad_bs: int = 2**5
    weights_bias: float = 1.1
