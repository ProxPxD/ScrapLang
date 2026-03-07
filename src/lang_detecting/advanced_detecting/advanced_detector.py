import math
import random
import warnings
from collections import Counter, OrderedDict
from dataclasses import asdict
from functools import cached_property
from itertools import product
from unittest.mock import MagicMock

import numpy as np

from src.lang_detecting.advanced_detecting.data.batcher import Batcher
from src.lang_detecting.advanced_detecting.data.preprocessing import PreprocessorFactory
from src.lang_detecting.advanced_detecting.data.preprocessing.core.consts import TensorBatch
from src.lang_detecting.advanced_detecting.data.splitter import Splitter
from src.lang_detecting.advanced_detecting.data.train_param_calc import TrainParamCalc

np.int = int

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
from src.lang_detecting.advanced_detecting.model import Moe
from src.lang_detecting.advanced_detecting.model_io_mging import KindToMgr, KindToTokensTargets, ModelIOMgr
from src.lang_detecting.advanced_detecting.retry import retry_on
from src.lang_detecting.advanced_detecting.tokenizer import KindToSpecs, MultiKindTokenizer, Tokens
from src.resouce_managing.valid_data import VDC, ValidDataMgr

warnings.filterwarnings('ignore', category=UserWarning, message=r'.*pkg_resources is deprecated.*Setuptools')

EXCEPTION = None
try:
    # noinspection PyUnresolvedReferences
    from tqdm import tqdm
    # noinspection PyUnresolvedReferences
    from torchmetrics import ConfusionMatrix, Metric, Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, SensitivityAtSpecificity
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mlcm import mlcm
    HAS_TRAINING_SUPERVISION = True
except ImportError as e:
    EXCEPTION = e
    HAS_TRAINING_SUPERVISION = False

e_t = e_c = None
try:
    # noinspection PyUnresolvedReferences
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError as e_t:
    HAS_TENSORBOARD = False

HAS_CLEARML = False
try:
    # noinspection PyUnresolvedReferences
    from clearml import Logger, Task
    HAS_CLEARML = True
    # noinspection PyUnresolvedReferences
    import flatten_dict
except ImportError as e_c:
    if HAS_CLEARML:
        EXCEPTION = e_c

if HAS_TRAINING_SUPERVISION and not HAS_CLEARML and not HAS_TENSORBOARD:
    # noinspection PyUnboundLocalVariable
    EXCEPTION = Exception(e_t, e_c)

SERIES_SEQ = (TRAIN:='Train', VAL:='Val')

class AdvancedDetector:
    def __init__(self, context: Context, lang_script: DataFrame, valid_data_mgr: ValidDataMgr, conf: Conf):
        self.context =  context
        self.model_io_mgr = ModelIOMgr()
        self.valid_data_mgr = valid_data_mgr
        self.conf = conf

        # noinspection SpellCheckingInspection
        kind_to_specs: KindToSpecs = {
            'Latn': [(str.isupper, str.lower)],
            'Cyrl': [(str.isupper, str.lower)],
        }

        kinds_to_tokens_targets: KindToTokensTargets = self.model_io_mgr.extract_kinds_to_vocab_classes(lang_script)
        self.model_io_mgr.update_model_io_if_needed(kinds_to_tokens_targets)
        kind_to_vocab, kinds_to_targets = KindToMgr.separate_kinds_tos(kinds_to_tokens_targets)
        kind_to_vocab = self.model_io_mgr.enhance_tokens(kind_to_vocab, [Tokens.BOS])
        self.kinds_to_vocab = kind_to_vocab
        targets = c(kinds_to_targets.values()).flatten().sorted_uniq().value()

        self.tokenizer = MultiKindTokenizer(kind_to_vocab, targets, kind_to_specs=kind_to_specs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessing = PreprocessorFactory(tokenizer=self.tokenizer, conf=self.conf)
        self.splitter = Splitter(self.conf)
        self.batcher = Batcher(self.conf, self.tokenizer)
        self.train_param_calc = TrainParamCalc(self.conf)
        # noinspection PyTypeChecker
        self.conf.data.labels.all_names = c(kinds_to_targets.values()).flatten().apply(flow(set, sorted, tuple)).value()
        self.moe = Moe(kind_to_vocab, kinds_to_targets, valmap(len, kind_to_specs), conf=self.conf).to(self.device)
        self.writer = MagicMock()
        self.task = MagicMock()
        self.metrics = {}
        self._cm_threshes = (.10, .66, .80, .90)
        self._cms: dict[str, dict[float, np.ndarray]] = {}
        self._cm_kind_every: int = 2 ** 5
        self.init_for_training()

    def init_for_training(self) -> None:
        if self.context.dev and not HAS_TRAINING_SUPERVISION or EXCEPTION:
            raise RuntimeError('Dev mode run without tensorboard, torchmetrics, tqdm or matplotlib or mlcm or either tensorboard or clearml and flatten_dict installed') from EXCEPTION
        if not self.dev_training:
            return
        Paths.DETECTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
        for file in Paths.DETECTION_LOG_DIR.iterdir():
            if file.name.startswith('events'):
                file.unlink()
        if HAS_CLEARML:
            try:
                # noinspection SpellCheckingInspection
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
                tags = ['autodelete']
                match self.conf.data.augment.is_augmenting:
                    case True: tags.append('augmented')
                    case False: tags.append('non-augmented')
                match math.log2(self.conf.train.epochs):
                    case exp if exp <= 8: tags.append('lil')
                    case exp if 8 < exp <= 9: tags.append('mid')
                    case exp if 9 < exp <= 10: tags.append('mid-big')
                    case exp if 10 < exp: tags.append('big')
                self.task = Task.init(
                    project_name='ScrapLang', task_name='Train', task_type=Task.TaskTypes.training,
                    tags=tags, reuse_last_task_id=False, auto_connect_arg_parser=False
                )

                self.task.connect(flatten_dict.flatten(asdict(self.conf), reducer='dot'))

            except MissingConfigError as mce:
                if not HAS_TENSORBOARD:
                    raise RuntimeError('Dev mode run failed to initialize ClearML') from mce
        if HAS_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=Paths.DETECTION_LOG_DIR)
        kwargs = dict(task='multilabel', num_labels=self.conf.data.labels.n_all)
        self.metrics: dict[str, dict[str, Metric]] = {}
        for series in SERIES_SEQ:
            # noinspection PyTypeChecker
            self.metrics[series] = {
                f'{metric_class.__name__}': metric_class(**kwargs, average=mode).to(self.device)  # noinspection PyTypeChecker
                for metric_class in [Accuracy, Precision, Recall, F1Score]
                for mode in ('macro',)
            }
            self.metrics[series]['MatthewsCorrCoef'] = MatthewsCorrCoef(**kwargs).to(self.device)
            self._cms[series] = {th: np.zeros((self.conf.data.labels.n_all + 1, self.conf.data.labels.n_all + 1), dtype=int) for th in self._cm_threshes}
        if self.conf.data.labels.all_names:
            class_weights = self.train_param_calc.compute_weights().to(self.device)
            self.train_param_calc.loss_func = nn.BCEWithLogitsLoss(weight=class_weights.to(self.device), reduction='none')

    @property
    def dev_training(self) -> bool:
        return self.context.dev and HAS_TRAINING_SUPERVISION

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
        random.seed(self.conf.seed)

        df: DataFrame = self.preprocessing.init_preprocessor(self.valid_data_mgr.data)
        self.conf.data.labels.used_count = Counter(df[VDC.LANG].explode())
        train_df, val_df = self.splitter.split(self.preprocessing.group(df))
        train_df = self.preprocessing.train_preprocessor(train_df)
        val_df = self.preprocessing.val_preprocessor(val_df)
        batch_all = c().map(self.batcher.batch_data_up)
        train_batches, val_batches = batch_all([train_df, val_df])
        optimizer = torch.optim.AdamW(self.moe.parameters(), lr=self.conf.train.lr, weight_decay=self.conf.train.weight_decay)
        self.init_for_training()
        if self.dev_training:
            comment = []
            comment += [f'{len(self.conf.data.labels.all_names)}-label']
            comment += [class_names_string := ', '.join(self.conf.data.labels.all_names)]
            self.writer.add_text(tag='Classes', text_string=class_names_string)
            comment += [lang_counts := '\n'.join(f'{lang}: {count}' for lang, count in sorted(self.conf.data.labels.used_count.items(), key=c().get(1), reverse=True))]
            self.writer.add_text(tag='training info', text_string=lang_counts)
            self.task.set_comment('\n'.join(comment))
        for epoch in range(self.conf.train.epochs):
            self._reset_metrics()
            train_batches = self.train_param_calc.shuffle_batches(train_batches)
            n_records = 0
            epoch_loss = 0.0
            for batch in tqdm(train_batches, desc=f'Epoch {epoch+1}/{self.conf.train.epochs}'):
                kinds, words, specs, targets = [t.to(self.device) for t in batch]
                n_records += words.size(0)
                logits: Tensor = self.moe(kinds, words, specs)
                # self.moe.eval()
                # with torch.no_grad():
                #     val_logits = self.moe()
                probs = torch.sigmoid(logits)
                self._update_metrics(TRAIN, probs, targets)
                loss = self.train_param_calc.compute_loss(logits, targets)
                loss.backward()
                epoch_loss += loss.item()

                if n_records >= self.conf.train.accum_grad_bs:
                    optimizer.step()
                    optimizer.zero_grad()
                    n_records = 0
            if n_records:
                optimizer.step()
            self._val(val_batches, epoch)
            retry_on(self._logger.report_scalar, ConnectionError, 7, 'Loss', TRAIN, epoch_loss, epoch)
            self._board_metrics(TRAIN, epoch)

    def _val(self, batches: list[TensorBatch], epoch: int) -> None:
        self.moe.eval()
        with torch.no_grad():
            val_loss = .0
            for batch in batches:
                kinds, words, specs, targets = [t.to(self.device) for t in batch]
                logits: Tensor = self.moe(kinds, words, specs)
                probs = torch.sigmoid(logits)
                self._update_metrics(VAL, probs, targets)
                val_loss += self.train_param_calc.compute_loss(logits, targets)
            retry_on(self._logger.report_scalar, ConnectionError, 7, 'Loss', VAL, val_loss, epoch)
            self._board_metrics(VAL, epoch)
        self.moe.train()

    def _manage_metrics(self, func_name: str, series: str, *args, **kwargs) -> None:
        for metric in self.metrics[series].values():
            getattr(metric, func_name)(*args, **kwargs)

    def _reset_metrics(self) -> None:
        if not self.dev_training:
            return
        for series in SERIES_SEQ:
            self._manage_metrics('reset', series)
            for cm in self._cms[series].values():
                cm *= 0

    def _update_metrics(self, series: str, probs: Tensor, targets: Tensor) -> None:
        if not self.dev_training:
            return
        self._manage_metrics('update', series, (probs > .80).long(), targets)
        for thresh, cm in self._cms[series].items():
            np.int = int
            count_matrix, percentage_matrix = mlcm.cm(targets.cpu().numpy(), (probs > thresh).long().cpu().numpy(), print_note=False)
            cm += count_matrix

    def _board_metrics(self, series: str, step: int) -> None:
        if not self.dev_training:
            return
        self.metrics: dict[str, Metric]
        for name, metric in self.metrics[series].items():
            val = metric.compute().item()
            self.writer.add_scalar(f'{series}/metric/{name}'.lower(), val, step)
            self._logger.report_scalar(f'Metric/{name}', series, val, step)
            retry_on(self._logger.report_scalar, ConnectionError, 7, f'Metric/{name}', series, val, step)
        self._board_confusion_matrices(series, step)

    @cached_property
    def _n_cm_padding(self) -> int:
        return math.floor(math.log10(self.conf.train.epochs // self._cm_kind_every))

    def _board_confusion_matrices(self, series: str, step: int) -> None:
        if not self.dev_training:
            return
        if not (step <= 10 or step % 4 == 0 or step == self.conf.train.epochs - 1):
            return
        all_names = self.conf.data.labels.all_names
        kwargs = dict(iteration=step + 1, yaxis_reversed=True)
        # NxN
        for thresh, cm in self._cms[series].items():
            if HAS_CLEARML:
                retry_on(self._logger.report_confusion_matrix, ConnectionError, n_tries=7, **kwargs,
                         title=f'I NxN {thresh:.2f}', series=series, matrix=cm.tolist(),
                         xlabels=[*all_names, '_'], ylabels=[*all_names, '_'])
        # OvR
        true_pos_dict = {}
        for thresh, cm in self._cms[series].items():
            if HAS_CLEARML:
                true_pos_dict[thresh] = true_pos = np.diag(cm)
                false_pos = cm.sum(axis=0) - true_pos
                false_neg = cm.sum(axis=1) - true_pos
                true_false = np.stack([true_pos, false_pos, false_neg], axis=0).T
                retry_on(self._logger.report_confusion_matrix, ConnectionError, n_tries=7, **kwargs,
                         title=f'II OvR {thresh:.2f}', series=series, matrix=true_false.tolist(),
                         xlabels=['True', 'False Pos', 'False Neg'], ylabels=[*all_names, '_'])
        for thresh, cm in self._cms[series].items():
            if HAS_CLEARML:
                # noinspection PyPep8Naming
                C = self.conf.data.labels.n_all
                tp, core = true_pos_dict[thresh].sum(), cm[0:C, 0:C].sum()
                off_diag = core - tp
                last_col, last_row, bottom = cm[0:C, C].sum(), cm[C, 0:C].sum(), cm[C, C].item()
                tf = [
                    [tp, off_diag, last_col],
                    [bottom, last_row, 0]
                ]
                retry_on(self._logger.report_confusion_matrix, ConnectionError, n_tries=7, **kwargs,
                         title=f'III Micro {thresh:.2f}', series=series, matrix=tf,
                         xlabels=['Pred Pos', 'Pred Neg', 'Pred None'], ylabels=['Act Pos', 'Act Neg'])

    @classmethod
    def plot_confusion_matrix(cls, conf_mat, class_names):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        im = ax.imshow(conf_mat, interpolation='nearest')

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
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        fig.tight_layout()
        return fig
