from contextlib import ExitStack, contextmanager
from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

import torch
from lightning.fabric.accelerators import CPUAccelerator
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.plugins.precision import Precision
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_2_1,
)
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loops.fetchers import _DataFetcher
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing_extensions import override

from nemo.lightning import _strategy_lib
from nemo.lightning.base import DataConfig, get_vocab_size
from nemo.lightning.megatron_parallel import CallbackConnector, MegatronParallel

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig


class FabricMegatronStrategy(DDPStrategy):
    def __init__(
        self,
        parallelism: "ModelParallelConfig",
        data_config: DataConfig,
        # tensor_model_parallel_size: int = 1,
        # pipeline_model_parallel_size: int = 1,
        # virtual_pipeline_model_parallel_size: Optional[int] = None,
        # sequence_parallel: bool = False,
        # data_sampler: Optional[DataSampler] = None,
        accelerator: Optional[Accelerator] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision: Optional[Precision] = None,
        megatron_parallel_cls: Type[nn.Module] = MegatronParallel,
        megatron_callbacks: Optional[CallbackConnector] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["popen", "spawn", "fork", "forkserver"] = "popen",
        no_ddp_communication_hook: bool = True,
        output_data_idx: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision=precision,
            process_group_backend=process_group_backend,
            timeout=timeout,
            start_method=start_method,
            **kwargs,
        )
        self.no_ddp_communication_hook = no_ddp_communication_hook
        self.parallelism = parallelism
        self.data_config = data_config
        self.megatron_parallel_cls = megatron_parallel_cls
        self.megatron_callbacks = CallbackConnector()
        if megatron_callbacks:
            self.megatron_callbacks.add(megatron_callbacks)
        self.output_data_idx = output_data_idx
        
        # used in NVIDIA NGC PyTorch containers
        _strategy_lib.enable_nvidia_optimizations()

    @override
    def _setup_distributed(self) -> None:
        self._set_world_ranks()
        
        assert self.cluster_environment is not None
        _strategy_lib.init_parallel_ranks(
            world_size=self.cluster_environment.world_size(),
            global_rank=self.cluster_environment.global_rank(),
            local_rank=self.cluster_environment.local_rank(),
            parallel_config=self.parallelism,
        )
        
        super()._setup_distributed()
        torch.cuda.set_device(self.cluster_environment.local_rank())
        
        if self.data_config is not None:
            _strategy_lib.initialize_data(self.cluster_environment.global_rank(), self.data_config)
        _strategy_lib.init_model_parallel()

    @override
    def process_dataloader(self, dataloader: DataLoader) -> Iterator:
        loader = _strategy_lib.process_dataloader(dataloader, self.data_config)

        # Code taken from: https://github.com/Lightning-AI/pytorch-lightning/blob/6cbe9ceb560d798892bdae9186291acf9bf5d2e3/src/lightning/pytorch/loops/fit_loop.py#L258-L260
        output = _MegatronDataLoaderIterDataFetcher(
            self.data_config, output_data_idx=self.output_data_idx
        )
        output.setup(CombinedLoader(loader, "max_size_cycle"))
        iter(output)

        return output

    @override
    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        """Pass the optimizer to the precision-plugin if needed & add it as callback."""
        if hasattr(self._precision, "setup_optimizer"):
            optimizer = self._precision.setup_optimizer(optimizer)

        self.megatron_callbacks.add(optimizer)

        return optimizer

    @override
    def setup_module(self, module: Module) -> DistributedDataParallel:
        from megatron.core import mpu
        from nemo.utils import AppState
        
        app_state = AppState()
        megatron_parallel = MegatronParallel(
            module, 
            vp_size=self.parallelism.virtual_pipeline_model_parallel_size,
            cpu=isinstance(self.accelerator, CPUAccelerator)
        )

        if app_state.model_parallel_size is not None:
            self._ddp_kwargs["process_group"] = mpu.get_data_parallel_group()

        dist_data_parallel = super().setup_module(megatron_parallel)
        if self.no_ddp_communication_hook:
            # When using custom gradient accumulation and allreduce, disable
            # DDP communication hook that works on the gradient bucket.
            # Instead, use the custom gradient function and communication hook,
            # which is defined in the master optimizer wrapper.
            dist_data_parallel.require_backward_grad_sync = False
            dist_data_parallel.register_comm_hook(None, noop_hook)

        return dist_data_parallel

    def module_init_context(self, empty_init: Optional[bool] = None) -> ContextManager:
        precision_init_ctx = self.precision.module_init_context()
        module_sharded_ctx = self.megatron_context()
        stack = ExitStack()
        if _TORCH_GREATER_EQUAL_2_1 and empty_init:
            # Materialization happens in `setup`. When modules get wrapped by FSDP, the sequence of operations is:
            # 1) materialize module 2) call `reset_parameters()` 3) shard the module.
            # These operations are applied to each submodule 'bottom up' in the module hierarchy.
            stack.enter_context(torch.device("meta"))
        stack.enter_context(precision_init_ctx)
        stack.enter_context(module_sharded_ctx)

        return stack
    
    def module_to_device(self, module: nn.Module) -> None:
        pass
    
    @override
    def save_checkpoint(
        self,
        path: _PATH,
        state: Dict[str, Union[Module, Optimizer, Any]],
        storage_options: Optional[Any] = None,
        filter_dict: Optional[Dict[str, Callable[[str, Any], bool]]] = None,
    ) -> None:
        """Save model, optimizer, and other state as a checkpoint file.

        Args:
            path: A path to where the file(s) should be saved
            state: A dictionary with contents to be saved. If the dict contains modules or optimizers, their
                state-dict will be retrieved and converted automatically.
            storage_options: Additional options for the ``CheckpointIO`` plugin
            filter: An optional dictionary containing filter callables that return a boolean indicating whether the
                given item should be saved (``True``) or filtered out (``False``). Each filter key should match a
                state key, where its filter will be applied to the ``state_dict`` generated.

        """
        state = self._convert_stateful_objects_in_state(state, filter=(filter_dict or {}))
        self.checkpoint_io.save_checkpoint(checkpoint=state, path=path, storage_options=storage_options)

    @contextmanager
    def megatron_context(self) -> Generator[None, None, None]:
        def monkey_patched(config):
            return {"device": "meta"}

        from megatron.core.transformer.custom_layers import transformer_engine as _te

        original = _te._get_extra_te_kwargs   # noqa: SLF001
        _te._get_extra_te_kwargs = monkey_patched   # noqa: SLF001

        self.parallelism.perform_initialization = False
        self.parallelism.use_cpu_initialization = True

        yield

        _te._get_extra_te_kwargs = original   # noqa: SLF001

    def get_vocab_size(self, vocab_size: int) -> int:
        return get_vocab_size(self.parallelism, vocab_size)


class _MegatronDataLoaderIterDataFetcher(_DataFetcher):
    def __init__(
        self, data_config: DataConfig, *args: Any, output_data_idx: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.data_config = data_config
        self.output_data_idx = output_data_idx
        self._batch: Any = None
        self._batch_idx: int = 0
        self._dataloader_idx: int = 0

    def __iter__(self) -> "_MegatronDataLoaderIterDataFetcher":
        super().__iter__()
        self.iterator_wrapper = iter(
            _DataFetcherWrapper(self, output_data_idx=self.output_data_idx)
        )
        return self

    def __next__(self) -> Iterator["_DataFetcherWrapper"]:  # type: ignore[override]
        if self.done:
            raise StopIteration
        return self.iterator_wrapper

    def reset(self) -> None:
        super().reset()
        self._batch = None
        self._batch_idx = 0
        self._dataloader_idx = 0


class _DataFetcherWrapper(Iterator):
    def __init__(
        self,
        data_fetcher: _MegatronDataLoaderIterDataFetcher,
        output_data_idx: bool = False,
    ) -> None:
        self.data_fetcher = data_fetcher
        self.output_data_idx = output_data_idx

    @property
    def done(self) -> bool:
        return self.data_fetcher.done

    @property
    def fetched(self) -> int:
        return self.data_fetcher.fetched

    @property
    def length(self) -> Optional[int]:
        return self.data_fetcher.length

    @property
    def data_config(self) -> DataConfig:
        return self.data_fetcher.data_config

    def __next__(self):
        fetcher = self.data_fetcher
        if fetcher.done:
            raise StopIteration
        batch, batch_idx, dataloader_idx = super(
            _MegatronDataLoaderIterDataFetcher, fetcher
        ).__next__()
        # save the state so the loops can access it
        fetcher._batch = batch   # noqa: SLF001
        fetcher._batch_idx = batch_idx   # noqa: SLF001
        fetcher._dataloader_idx = dataloader_idx   # noqa: SLF001

        if not self.output_data_idx:
            return batch

        return batch, batch_idx, dataloader_idx
