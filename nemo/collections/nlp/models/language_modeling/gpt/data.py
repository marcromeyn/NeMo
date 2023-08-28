from typing import List, Optional
from dataclasses import dataclass, field
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, open_dict, ListConfig

from nemo.collections.nlp.models.language_modeling.config.base import (
    BaseConfig,
)
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import GPTSFTChatDataset
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
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


@dataclass
class MetricConfig:
    """Configuration for metrics during training, validation, and testing."""
    name: str
    average: Optional[str] = None
    num_classes: Optional[int] = None


@dataclass
class GPTFineTuneDatasetConfig:
    """
    Dataclass for holding dataset configuration.

    Attributes:
        file_names (List[str]): List of paths to JSONL files for the dataset.
        global_batch_size (int): Global batch size for the dataset.
        micro_batch_size (int): Micro batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset or not.
        num_workers (int): Number of workers to use for data loading.
        memmap_workers (int): Number of workers to use for memory mapping.
        pin_memory (bool): Whether to pin memory or not.
        max_seq_length (int): Maximum sequence length.
        min_seq_length (int): Minimum sequence length.
        drop_last (bool): Whether to drop the last incomplete batch.
        concat_sampling_probabilities (Optional[List[float]]): Sampling probabilities for each dataset when strategy='random'.
        context_key (str): Key used for fetching context in the dataset.
        label_key (str): Key used for fetching label in the dataset.
        add_eos (bool): Whether to add End-of-String token.
        add_sep (bool): Whether to add separator token.
        add_bos (bool): Whether to add Beginning-of-String token.
        separate_prompt_and_response_with_newline (bool): Whether to separate prompt and response with a newline.
        truncation_field (str): Field to apply truncation to. Options: ['context', 'answer']
        index_mapping_dir (Optional[str]): Directory to write index mapping files.
        prompt_template (str): Template used for assistant prompt.
        write_predictions_to_file (bool): Whether to write predictions to a file.
        output_file_path_prefix (Optional[str]): Prefix of the file to write predictions to.
        tokens_to_generate (Optional[int]): Number of tokens to generate for evaluation.
        metric (MetricConfig): Metric configuration for evaluation.
    """
    file_names: List[str]
    global_batch_size: int
    micro_batch_size: int
    shuffle: bool
    num_workers: int
    memmap_workers: int
    pin_memory: bool
    max_seq_length: int
    min_seq_length: int
    drop_last: bool
    concat_sampling_probabilities: Optional[List[float]] = None
    context_key: str = 'input'
    label_key: str = 'output'
    add_eos: bool = True
    add_sep: bool = False
    add_bos: bool = False
    separate_prompt_and_response_with_newline: bool = False
    truncation_field: str = 'context'
    index_mapping_dir: Optional[str] = None
    prompt_template: str = "{input} {output}"
    write_predictions_to_file: bool = False
    output_file_path_prefix: Optional[str] = None
    tokens_to_generate: Optional[int] = None
    metric: MetricConfig = field(default_factory=lambda: MetricConfig(name="loss"))



class GPTFineTuneDataset(pl.LightningDataModule):
    def __init__(
        self, 
        train_ds: GPTFineTuneDatasetConfig,
        validation_ds: GPTFineTuneDatasetConfig,
        test_ds: GPTFineTuneDatasetConfig
    ):
        self.train_config = train_ds
        self.val_config = validation_ds
        self.test_config = test_ds
        
    def setup(self, stage=None):
        self._train_ds, self._validation_ds, self._test_ds = self.build_train_valid_test_datasets(stage)

    def build_train_valid_test_datasets(self, stage):
        if stage != 'test':
            logging.info('Building GPT SFT validation datasets.')
            # Wrap this in a list since the general finetuning parent class supports multi-validation.
            self._validation_ds = self._build_dataset(self.val_config, is_train=False)
            logging.info(f'Length of val dataset: {len(self._validation_ds[0])}')

        if stage != 'validate':
            if hasattr(self.cfg.data, 'test_ds'):
                logging.info('Building GPT SFT test datasets.')
                # Wrap this in a list since the general finetuning parent class supports multi-validation.
                self._test_ds = self._build_dataset(self.test_config, is_train=False)
                logging.info(f'Length of test dataset: {len(self._test_ds[0])}')

        if stage == 'validate' or stage == 'test':
            return
        logging.info('Building GPT SFT traing datasets.')
        self._train_ds = self._build_dataset(self.test_config)
        logging.info(f'Length of train dataset: {len(self._train_ds)}')

    def build_data_loader(self, dataset, data_cfg, consumed_samples=0):
        """Buld dataloader given an input dataset."""

        logging.info(f'Building dataloader with consumed samples: {consumed_samples}')
        if isinstance(dataset, BlendableDataset):
            collate_fn = dataset.datasets[0].collate_fn
        else:
            collate_fn = dataset.collate_fn

        batch_sampler = MegatronPretrainingBatchSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=data_cfg.micro_batch_size,
            global_batch_size=data_cfg.global_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size(),
            drop_last=data_cfg.drop_last,
            pad_samples_to_global_batch_size=not data_cfg.drop_last,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
        )

    def train_dataloader(self):
        consumed_samples = self.compute_consumed_samples(0)
        return self.build_data_loader(
            dataset=self._train_ds, 
            data_cfg=self.cfg.data.train_ds, 
            consumed_samples=consumed_samples,
        )

    def val_dataloader(self, datasets, data_cfg):
        dataloaders = []
        for dataset in datasets:
            eval_dl = self.build_data_loader(dataset=dataset, data_cfg=data_cfg, consumed_samples=0,)
            dataloaders.append(eval_dl)
        
        return dataloaders
    
    def _build_dataset(self, data_cfg, is_train=True):
        datasets = []
        # Determine if we are using a single dataset or a list of datasets.
        is_list_config = isinstance(data_cfg.file_names, ListConfig)
        if not is_list_config:
            raise ValueError(f"SFT train/validation datasets must be provided as a list of individual JSONL files.")

        if is_train:
            # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
            # that is of the format [weight1,file_name1,weight2,file_name2,...]
            if data_cfg.concat_sampling_probabilities is None or not isinstance(
                data_cfg.concat_sampling_probabilities, ListConfig
            ):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be a ListConfig with the same number of files in file_names."
                        f"Found: {data_cfg.concat_sampling_probabilities}"
                    )
                )

            if len(data_cfg.get('concat_sampling_probabilities', None)) != len(data_cfg.file_names):
                raise ValueError(
                    (
                        f"concat_sampling_probabilities must be of the same size as file_names.",
                        f"Provided size {len(data_cfg.concat_sampling_probabilities)}, number of datasets {len(data_cfg.file_names)}",
                    )
                )

            data_prefix = []
            for weight, prefix in zip(data_cfg.concat_sampling_probabilities, data_cfg.file_names):
                data_prefix.append(weight)
                data_prefix.append(prefix)

            if self.trainer.max_steps is None or self.trainer.max_steps <= 0:
                raise ValueError(
                    f'Trainer max_steps must be set to a positive integer. Found {self.trainer.max_steps}'
                )
            num_train_samples = [self.trainer.max_steps * data_cfg.global_batch_size]
            _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
            num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
        else:
            num_train_samples_per_dataset = [[None]] * len(data_cfg.file_names)

        # Check dataset max_seq_legnth and max_position_embeddings size
        if (
            self.cfg.get('position_embedding_type', None) in [None, 'learned_absolute']
            and data_cfg.max_seq_length > self.cfg.max_position_embeddings
        ):
            logging.warning(
                f"Set dataset max_seq_length to max_position_embeddings {self.cfg.max_position_embeddings} if using learned_absolute position embedding"
            )
            data_cfg.max_seq_length = self.cfg.max_position_embeddings

        for file_path, num_samples in zip(data_cfg.file_names, num_train_samples_per_dataset):
            if self.cfg.data.get("chat", False):
                dataset_cls = GPTSFTChatDataset
            else:
                dataset_cls = GPTSFTDataset
            dataset = dataset_cls(
                file_path=file_path,
                tokenizer=self.tokenizer,
                max_seq_length=data_cfg.max_seq_length,
                min_seq_length=data_cfg.min_seq_length,
                add_bos=data_cfg.get('add_bos', False),
                add_eos=data_cfg.get('add_eos', True),
                add_sep=data_cfg.get('add_sep', False),
                sep_id=self.sep_id,
                max_num_samples=num_samples[0],
                seed=data_cfg.get('seed', 1234),
                context_key=data_cfg.get('context_key', 'text'),
                label_key=data_cfg.get('label_key', 'answer'),
                separate_prompt_and_response_with_newline=data_cfg.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=self.cfg.get('answer_only_loss', True),
                truncation_field=data_cfg.get('truncation_field', 'context'),
                pad_to_max_length=data_cfg.get('pad_to_max_length', False),
                index_mapping_dir=data_cfg.get('index_mapping_dir', None),
                prompt_template=data_cfg.get('prompt_template', None),
                virtual_tokens=self.virtual_tokens,
                tokens_to_generate=data_cfg.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                memmap_workers=data_cfg.get(
                    'memmap_workers', None
                ),  # used to set num. of workers to create the memmap index files
                hf_dataset=data_cfg.get(
                    'hf_dataset', False
                ),  # Whether to load the json file with the HuggingFace dataset. otherwise, will load the jsonl file with the JSONLMemMapDataset.
            )
            datasets.append(dataset)

        if is_train:
            dataset = BlendableDataset(
                datasets=datasets, weights=data_cfg.concat_sampling_probabilities, size=num_train_samples_after_blend
            )
            return dataset
        else:
            return datasets
