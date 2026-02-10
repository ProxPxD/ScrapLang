import math
import warnings
from dataclasses import asdict
from functools import cached_property
from itertools import product
from typing import Callable, Sequence
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn
from clearml.backend_api.session.defs import MissingConfigError
from pandas import DataFrame
from pydash import chain as c, flow
from toolz import valmap
from torch import Tensor

from src.constants import Paths
from src.context import Context
from src.lang_detecting.advanced_detecting.conf import Conf
from src.lang_detecting.advanced_detecting.dataset import BucketChunkDataset
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import KindToMgr, KindToTokensTargets, ModelIOMgr
from src.lang_detecting.advanced_detecting.retry import retry_on
from src.lang_detecting.advanced_detecting.tokenizer import KindToSpecs, MultiKindTokenizer
from src.resouce_managing.valid_data import ValidDataMgr

warnings.filterwarnings('ignore', category=UserWarning, message=r'.*pkg_resources is deprecated.*Setuptools')

EXCEPTION = None
try:
    from tqdm import tqdm
    from torchmetrics import ConfusionMatrix, Metric, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef
    import matplotlib.pyplot as plt
    HAS_TRAINING_SUPERVISION = True
except ImportError as e:
    EXCEPTION = e
    HAS_TRAINING_SUPERVISION = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError as e_t:
    HAS_TENSORBOARD = False

HAS_CLEARML = False
try:
    from clearml import Logger, Task
    HAS_CLEARML = True
    import flatten_dict
    HAS_CLEARML = True
except ImportError as e_c:
    if HAS_CLEARML:
        EXCEPTION = e_c
    else:
        HAS_CLEARML = False

if HAS_TRAINING_SUPERVISION and not HAS_CLEARML and not HAS_TENSORBOARD:
    EXCEPTION = Exception(e_t, e_c)


class AdvancedDetector:
    def __init__(self, context: Context, lang_script: DataFrame, valid_data_mgr: ValidDataMgr, conf: Conf):
        self.context =  context
        self.model_io_mgr = ModelIOMgr()
        self.valid_data_mgr = valid_data_mgr
        self.conf = conf

        kind_to_specs: KindToSpecs = {
            'Latn': [(str.isupper, str.lower)],
            'Cyrl': [(str.isupper, str.lower)],
        }

        kinds_to_tokens_targets: KindToTokensTargets = self.model_io_mgr.extract_kinds_to_vocab_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_targets)
        kinds_to_vocab, kinds_to_targets = KindToMgr.separate_kinds_tos(kinds_to_tokens_targets)
        self.kinds_to_vocab = kinds_to_vocab
        targets = c(kinds_to_targets.values()).flatten().sorted_uniq().value()

        self.tokenizer = MultiKindTokenizer(kinds_to_vocab, targets, kind_to_specs=kind_to_specs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._all_class_names = c(kinds_to_targets.values()).flatten().apply(flow(set, sorted)).value()
        self.moe = Moe(kinds_to_vocab, kinds_to_targets, valmap(len, kind_to_specs), conf=self.conf).to(self.device)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.writer = MagicMock()
        self.task = MagicMock()
        self.metrics = {}
        self.confusion_matrix = None
        self._cm_kind_every: int = 2 ** 5
        self.init_for_training()

    def init_for_training(self) -> None:
        if self.context.dev and not HAS_TRAINING_SUPERVISION or EXCEPTION:
            raise RuntimeError('Dev mode run without tensorboard, torchmetrics, tqdm or matplotlib or either tensorboard or clearml and flatten_dict installed') from EXCEPTION
        if not self.dev_training:
            return
        Paths.DETECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        for file in Paths.DETECTION_LOG_DIR.iterdir():
            if file.name.startswith('events'):
                file.unlink()
        if HAS_CLEARML:
            try:
                Task.set_credentials(
                     api_host='http://127.0.0.1:7003',
                     web_host='http://127.0.0.1:7004',
                     files_host='http://127.0.0.1:7005',
                     key='RA0LL08K8QWF588QOBVB53FMVRIZ6P',
                     secret='aks1mQ-w_7Wwa0-a8nFhOwcDNFYKP8dKZvFa-wMvytzlMJ0UZLiRfQBWlT-4nFRj5Vk',
                )
                for task in Task.get_tasks(project_name='ScrapLang', task_name='Train', tags=['autodelete']):
                    print(f'Deleting old task: {task.name}')
                    task.delete()
                self.task = Task.init(project_name='ScrapLang', task_name='Train', task_type=Task.TaskTypes.training, tags=['autodelete'])

                self.task.connect(flatten_dict.flatten(asdict(self.conf), reducer='dot'))

            except MissingConfigError as e:
                if not HAS_TENSORBOARD:
                    raise RuntimeError('Dev mode run failed to initialize ClearML') from e
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=Paths.DETECTION_LOG_DIR)
        self.metrics: dict[str, Metric] = {
            f'{metric_class.__name__}_{mode}'.lower(): metric_class(task='multiclass', num_classes=self._n_classes, average=mode).to(self.device)
            for metric_class in [Accuracy, Precision, Recall, F1Score]
            for mode in ('macro', )
        }
        self.metrics['matthews_corr_coef'] = MatthewsCorrCoef(task='multiclass', num_classes=self._n_classes).to(self.device)
        self.confusion_matrix = ConfusionMatrix(task='multiclass', num_classes=self._n_classes).to(self.device)

    @property
    def dev_training(self) -> bool:
        return self.context.dev and HAS_TRAINING_SUPERVISION

    @cached_property
    def _n_classes(self) -> int:
        return len(self._all_class_names)

    @cached_property
    def _logger(self) -> Logger:
        return self.task.get_logger()

    def retrain_model(self) -> None:
        try:
            self._retrain_model()
        finally:
            self.task.close()

    def _retrain_model(self) -> None:
        self.task: Task
        dataset = BucketChunkDataset(self.valid_data_mgr.data, tokenizer=self.tokenizer, conf=self.conf, shuffle=True, all_classes=self._all_class_names)
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.lr, weight_decay=self.conf.weight_decay)
        loss_func = nn.BCEWithLogitsLoss(weight=dataset.class_weights.to(self.device), reduction='none')
        self.init_for_training()
        eps = 1e-3
        if self.dev_training:
            comment = []
            comment += [f'{len(self._all_class_names)}-label']
            comment += [class_names_string := ', '.join(self._all_class_names)]
            self.writer.add_text(tag='Classes', text_string=class_names_string)
            comment += [lang_counts := '\n'.join(f'{lang}: {count}' for lang, count in sorted(dataset.class_counts.items(), key=c().get(1), reverse=True))]
            self.writer.add_text(tag='training info', text_string=lang_counts)
            self.task.set_comment('\n'.join(comment))
        for epoch in range(self.conf.epochs):
            self._reset_metrics()
            n_records = 0
            epoch_loss = 0.0
            for batch in tqdm(dataset, desc=f'Epoch {epoch+1}/{self.conf.epochs}'):
                kinds, words, specs, targets = [t.to(self.device) for t in batch]
                n_records += (bs:=words.size(0))
                preds: Tensor = self.moe(kinds, words, specs)
                self._update_metrics(preds, targets)
                loss = loss_func(preds, targets)
                m_neg = targets == 0
                loss_weights = torch.ones_like(preds)
                loss_weights[m_neg] = (preds ** self.conf.neg_bias)[m_neg]
                asym_loss = (loss * loss_weights)
                loss_per_cls = loss.mean(0)
                scale = loss_per_cls.detach().mean() / (loss_per_cls.detach() + eps)
                m_pos = ~m_neg
                final_loss_full = asym_loss * (1 + m_pos * (scale - 1))
                final_loss = final_loss_full.mean()
                final_loss.backward()
                epoch_loss += final_loss.item()

                if n_records >= self.conf.accum_grad_bs:
                    optimizer.step()
                    optimizer.zero_grad()
                    n_records = 0
            if n_records:
                optimizer.step()

            retry_on(self._logger.report_scalar, ConnectionError, 7, 'Loss', mode:='Train', epoch_loss, epoch)
            self._board_metrics(epoch, mode)

    def _manage_metrics(self, func_name: str, *args, **kwargs) -> None:
        for metric in self.metrics.values():
            getattr(metric, func_name)(*args, **kwargs)

    def _reset_metrics(self) -> None:
        if not self.dev_training:
            return
        self._manage_metrics('reset')
        self.confusion_matrix.reset()

    def _update_metrics(self, preds: Tensor, targets: Tensor) -> None:
        if not self.dev_training:
            return
        self._manage_metrics('update', preds, targets)
        pred_labels, true_labels = preds.argmax(dim=1), targets.argmax(dim=1)
        self.confusion_matrix.update(pred_labels, true_labels)

    def _board_metrics(self, step: int, mode: str) -> None:
        if not self.dev_training:
            return
        self.metrics: dict[str, Metric]
        for name, metric in self.metrics.items():
            val = metric.compute().item()
            self.writer.add_scalar(f'{mode}/metric/{name}'.lower(), val, step)
            self._logger.report_scalar(name, mode, val, step)
            retry_on(self._logger.report_scalar, ConnectionError, 7, name, mode, val, step)
        self._board_confusion_matrix(step, mode)

    @cached_property
    def _n_cm_padding(self) -> int:
        return math.floor(math.log10(self.conf.epochs // self._cm_kind_every))

    def _board_confusion_matrix(self, step: int, mode: str):
        if not self.dev_training:
            return
        confusion_matrix = self.confusion_matrix.compute().cpu().numpy()
        if HAS_TENSORBOARD:
            fig = self.plot_confusion_matrix(confusion_matrix, self._all_class_names)
            self.writer.add_figure(f'{mode}/ConfusionMatrix', fig, step)
            plt.close(fig)
        if HAS_CLEARML:
            kwargs = dict(
                title='Confusion Matrix', matrix=confusion_matrix.tolist(), iteration=step+1,
                xlabels=self._all_class_names, ylabels=self._all_class_names, yaxis_reversed=True,
            )
            if step % 2 == 0 or step == self.conf.epochs - 1:
                retry_on(self._logger.report_confusion_matrix, ConnectionError, n_tries=7, **kwargs, series=mode)
            if step == 0 or (step+1) % self._cm_kind_every == 0 or step == self.conf.epochs - 1:
                retry_on(self._logger.report_confusion_matrix, ConnectionError, n_tries=7,
                    **kwargs, series=f'{mode}_{(step+1) // self._cm_kind_every:0>{self._n_cm_padding}}'
                )

    @classmethod
    def plot_confusion_matrix(cls, conf_mat, class_names):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        im = ax.imshow(conf_mat, interpolation="nearest")

        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
            xticklabels=class_names, yticklabels=class_names,
            ylabel='True label', xlabel='Predicted label',
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        thresh = conf_mat.max() / 2
        n_classes = conf_mat.shape[0]
        for i, j in product(range(n_classes), repeat=2):
            ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='white' if conf_mat[i, j] > thresh else 'black')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        fig.tight_layout()
        return fig
