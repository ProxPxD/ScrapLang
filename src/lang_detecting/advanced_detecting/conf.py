from dataclasses import dataclass, field
from typing import OrderedDict, Sequence, Optional

@dataclass
class Augment:
    is_augmenting: bool = True
    pre_augment_size: int = 5
    post_augment_size: int = pre_augment_size

@dataclass
class ValSet:
    size: float = .1
    min_n_label: int = 5

@dataclass
class WordConstrain:
    len_thresh: int = 5
    n_non_uniq: int = 7
    len_to_uniq_ratio: float = 2.0

@dataclass
class Labels:
    all_names: tuple[str] = None  # Autofilled
    used_count: OrderedDict[str, int] = None  # Autofilled

    @property
    def n_all(self) -> Optional[int]:
        return None if self.all_names is None else len(self.all_names)

    @property
    def n_used(self) -> Optional[int]:
        return None if self.used_count is None else len(self.used_names)

    @property
    def used_names(self) -> tuple[str, ...]:
        return tuple(self.used_count.keys())

@dataclass
class Data:
    min_record_n_thresh: int =2**5
    valset: ValSet = field(default_factory=ValSet)
    augment: Augment = field(default_factory=Augment)
    word: WordConstrain = field(default_factory=WordConstrain)
    labels: Labels = field(default_factory=Labels)


@dataclass
class ExpertConf:
    # TODO remove chunk info
    s_chunk: int = 7  # Suits well with the paddings to yield 3 in the last layer
    s_chunk_step: int = s_chunk // 2 + 1
    emb_dim: int = 32
    kernels: Sequence[int] = (3, 3, 3)
    hidden_channels: Sequence[int] = (32, 32)
    paddings: Sequence[int] =  (0, 1, 1)
    leaky_relu_slop: float = 0.1

@dataclass
class Weights:
    prob_tau: float = .5
    entropy_tau: float = .5

    freq_bias: float = 4 # 1.1 # 1.1
    neg_bias: float = 4 # 4  # 3

@dataclass
class Train:
    epochs: int = 2**8  # 2**7
    lr: float = 1e-2  # 1e-5  # 1e-3
    weight_decay = 1e-5  # 1e-4
    max_batch_size: Optional[int] = 2**12
    accum_grad_bs: int = 2**9

@dataclass
class Conf:
    seed: int = 9
    data: Data = field(default_factory=Data)
    expert: ExpertConf = field(default_factory=ExpertConf)
    train: Train = field(default_factory=Train)
    weights: Weights = field(default_factory=Weights)
