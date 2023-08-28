from typing import Optional
from dataclasses import dataclass
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.config.base import (
    BaseConfig,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.utils import logging
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from megatron.core import parallel_state


@dataclass
class GPTPretrainDatasetConfig(BaseConfig):
    """
    Configuration for the GPT pretraining dataset.

    Attributes:
        data_prefix (Union[List, str, dict]): Path to the data. Can be specified in various formats
            like List, String, or Dictionary. For examples, please refer to the comments in the class.
        index_mapping_dir (Optional[str]): Path to save index mapping .npy files. Default is the same location as data_prefix.
        data_impl (str): Implementation type of the data. Default is "mmap".
        splits_string (str): Split ratios for train, validation, and test data. Default is "900, 50, 50".
        seq_length (str): Sequence length of the model's encoder. Default is "${model.encoder_seq_length}".
        skip_warmup (bool): Whether to skip warmup. Default is True.
        num_workers (int): Number of worker threads for data loading. Default is 2.
        dataloader_type (str): Type of data loader. Options are "single" or "cyclic". Default is "single".
        reset_position_ids (bool): Whether to reset position ids after end-of-document token. Default is False.
        reset_attention_mask (bool): Whether to reset attention mask after end-of-document token. Default is False.
        eod_mask_loss (bool): Whether to mask loss for the end-of-document tokens. Default is False.
        validation_drop_last (bool): Whether to consume last partial validation samples. Set to False if you want to consume them. Default is True.
        no_seqlen_plus_one_input_tokens (bool): Set to True to disable fetching (sequence length + 1) input tokens. Default is False.
        pad_samples_to_global_batch_size (bool): Set to True to pad the last partial batch with -1's to equal global batch size. Default is False.
        shuffle_documents (bool): Set to False to disable documents shuffling. Default is True.
        exchange_indices_distributed (bool): Set to True to exchange indices via torch.distributed instead of filesystem. Default is False.
    """

    data_prefix: str
    index_mapping_dir: Optional[str] = None
    data_impl: str = "mmap"
    splits_string: str = "900, 50, 50"
    seq_length: str = "${model.encoder_seq_length}"
    skip_warmup: bool = True
    num_workers: int = 2
    dataloader_type: str = "single"
    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    eod_mask_loss: bool = False
    validation_drop_last: bool = True
    no_seqlen_plus_one_input_tokens: bool = False
    pad_samples_to_global_batch_size: bool = False
    shuffle_documents: bool = True
    exchange_indices_distributed: bool = False
    
    
class GPTPreTrainingDataset(pl.LightningDataModule):
    def __init__(
        self, 
        config: GPTPretrainDatasetConfig,
        model_config,
    ):
        super(GPTPreTrainingDataset, self).__init__()
        self.config = config
        self.model_config = model_config

    def prepare_data(self):
        # This method is intended for tasks like downloading data, etc.
        pass
    
    def setup(self, stage=None):
        self._train_ds, self._validation_ds, self._test_ds = self.build_train_valid_test_datasets()
        # Move other logic here if needed

    def build_train_valid_test_datasets(self):
        logging.info('Building GPT datasets.')
        if self.trainer.limit_val_batches > 1.0 and isinstance(self.trainer.limit_val_batches, float):
            raise ValueError("limit_val_batches must be an integer or float less than or equal to 1.0.")
        global_batch_size = self.model_config.global_batch_size
        max_train_steps = self.trainer.max_steps
        eval_iters = (max_train_steps // self.trainer.val_check_interval + 1) * self.trainer.limit_val_batches
        test_iters = self.trainer.limit_test_batches

        train_valid_test_num_samples = [
            max_train_steps * global_batch_size,
            eval_iters * global_batch_size,
            test_iters * global_batch_size,
        ]

        if self.trainer.limit_val_batches <= 1.0 and isinstance(self.trainer.limit_val_batches, float):
            train_valid_test_num_samples[
                1
            ] = 1  # This is to make sure we only have one epoch on every validation iteration

        cfg = self.model_config.to_cfg()
        with open_dict(cfg):
            cfg.data = OmegaConf.structured(self.config)
        
        self._train_ds, self._validation_ds, self._test_ds = build_train_valid_test_datasets(
            cfg=cfg,
            trainer=self.trainer,
            data_prefix=self.config.data_prefix,
            data_impl=self.config.data_impl,
            splits_string=self.config.splits_string,
            train_valid_test_num_samples=train_valid_test_num_samples,
            seq_length=self.config.seq_length,
            seed=self.model_config.seed,
            skip_warmup=self.model_config.get('skip_warmup', True),
            tokenizer=self.trainer.model.tokenizer,
        )
        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building GPT datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def build_pretraining_data_loader(self, dataset, consumed_samples, dataset_type=None, drop_last=True, pad_samples_to_global_batch_size=False):
        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=drop_last,
                    global_batch_size=self.cfg.global_batch_size,
                    rampup_batch_size=self.cfg.get('rampup_batch_size', None),
                    pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
        )
    def train_dataloader(self):
        # The existing code to create a train DataLoader using self._train_ds
        consumed_samples = self.trainer.model.compute_consumed_samples(0)
        return self.build_pretraining_data_loader(self._train_ds, consumed_samples)

    def val_dataloader(self):
        # The existing code to create a validation DataLoader using self._validation_ds
        consumed_samples = 0
        return self.build_pretraining_data_loader(self._validation_ds, consumed_samples, "validation", ...)

    def test_dataloader(self):
        # The existing code to create a test DataLoader using self._test_ds
        consumed_samples = 0
        return self.build_pretraining_data_loader(self._test_ds, consumed_samples)
