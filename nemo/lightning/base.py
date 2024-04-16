import gc
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, List, Optional, Type, TypeVar

import torch
import torch.distributed
from pytorch_lightning import LightningModule, Trainer
from torch import nn

from nemo import io

NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE = "NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE"
DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
NEMO_CACHE_HOME = Path(os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME))
DEFAULT_NEMO_DATASETS_CACHE = NEMO_CACHE_HOME / "datasets"
NEMO_DATASETS_CACHE = Path(os.getenv("NEMO_DATASETS_CACHE", DEFAULT_NEMO_DATASETS_CACHE))


@dataclass
class ModelParallelism(io.IOMixin):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    sequence_parallel: bool = False

    # Initialization
    perform_initialization: bool = True
    use_cpu_initialization: bool = False

    # Optimizations
    gradient_accumulation_fusion: bool = False
    async_tensor_model_parallel_allreduce: bool = False

    # TODO: What to do with?
    params_dtype: torch.dtype = torch.float32

    # Pipeline Parallel
    pipeline_dtype: Optional[torch.dtype] = None
    grad_scale_func: Optional[Callable] = None
    enable_autocast: bool = False
    autocast_dtype: Optional[torch.dtype] = None
    variable_seq_lengths: bool = False
    num_microbatches_with_partial_activation_checkpoints: Optional[int] = None
    overlap_p2p_comm: bool = False
    batch_p2p_comm: bool = True
    batch_p2p_sync: bool = True
    use_ring_exchange_p2p: bool = False
    deallocate_pipeline_outputs: bool = False
    no_sync_func: Optional[Callable] = None
    grad_sync_func: Optional[Callable] = None
    param_sync_func: Optional[Callable] = None
    
    # def fabric(self, **kwargs) -> Fabric:
    #     from nemo.lightning import FabricMegatronStrategy
        
    #     return Fabric(strategy=FabricMegatronStrategy(self, )
        

    def __post_init__(self):
        """Python dataclass method that is used to modify attributes after initialization.
        See https://docs.python.org/3/library/dataclasses.html#post-init-processing for more details.
        """
        if self.sequence_parallel:
            if self.tensor_model_parallel_size <= 1:
                raise ValueError("Can not use sequence paralllelism without tensor parallelism")
            if self.async_tensor_model_parallel_allreduce:
                # sequence_parallelism already does this async
                self.async_tensor_model_parallel_allreduce = False

        if self.pipeline_model_parallel_size > 1:
            if self.pipeline_dtype is None:
                raise ValueError(
                    "When using pipeline parallelism, pipeline_dtype must be specified"
                )

        if self.autocast_dtype is None:
            self.autocast_dtype = self.params_dtype


@dataclass
class DataConfig:
    seq_length: int
    micro_batch_size: int = 4
    global_batch_size: int = 8
    rampup_batch_size: Optional[List[int]] = None
    train_drop_last: bool = True
    val_drop_last: bool = True
    test_drop_last: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = False

    @property
    def num_microbatches(self) -> int:
        from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        return get_num_microbatches()


ModelT = TypeVar("ModelT", bound=LightningModule)


class ModelConfig(Generic[ModelT], io.IOMixin):
    def model_cls(self) -> Type[ModelT]:
        raise NotImplementedError("Must be implemented by subclass")

    @property
    def model_type(self) -> Type[ModelT]:
        return self.model_cls()
    
    def init(self, *args, data=None, cpu: bool = False, **kwargs) -> ModelT:
        model_cls = self.model_cls()
        if data:
            kwargs.update(data.model_kwargs())

        signature = inspect.signature(model_cls.__init__)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

        model = model_cls(self, *args, **filtered_kwargs)
        
        if not cpu:
            model.cuda(torch.cuda.current_device())
            
        return model


def get_vocab_size(
    config,
    vocab_size: int,
    make_vocab_size_divisible_by: int = 128,
) -> int:
    from nemo.utils import logging

    after = vocab_size
    multiple = make_vocab_size_divisible_by * config.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    logging.info(
        f"Padded vocab_size: {after}, original vocab_size: {vocab_size}, dummy tokens:"
        f" {after - vocab_size}."
    )

    return after


def teardown(trainer: Trainer, model: Optional[nn.Module] = None) -> None:
    # Destroy torch distributed
    if torch.distributed.is_initialized():
        from megatron.core import mpu
        
        mpu.destroy_model_parallel()
        torch.distributed.destroy_process_group()
        
    trainer._teardown()     # noqa: SLF001
    if model is not None:
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
    
    gc.collect()
    torch.cuda.empty_cache()


__all__ = ["DataConfig", "ModelConfig", "get_vocab_size", "ModelT"]
