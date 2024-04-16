import functools
import itertools
import os
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)

import lightning.fabric as fl
import lightning.pytorch as pl
import torch
from lightning.fabric.strategies.parallel import ParallelStrategy
from lightning.fabric.utilities.types import Optimizable
from torch import nn
from torch.utils.data import DataLoader

from nemo import io

NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE = "NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE"


if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from nemo.lightning.megatron_parallel import MegatronParallel
    from nemo.lightning.base import DataConfig


class SharedStateDictProtocol(Protocol):
    def sharded_state_dict(self, prefix=""):
        ...


def init_parallel_ranks(
    world_size: int,
    global_rank: int,
    local_rank: int,
    parallel_config: "ModelParallelConfig",
    seed=1234,
    fp8=False,
) -> None:
    """
    Initializes the parallel ranks for distributed training.

    This function sets up the parallel ranks based on the provided world size, global rank, local rank,
    and parallel configuration. It also sets the seed for random number generation and determines whether
    to use fp8 precision.

    Args:
        world_size (int): The total number of processes participating in the distributed training.
        global_rank (int): The rank of the current process in the distributed training setup.
        local_rank (int): The rank of the current process within its machine.
        parallel_config (ModelParallelConfig): The configuration object containing settings for model parallelism.
        seed (int, optional): The seed for random number generation. Defaults to 1234.
        fp8 (bool, optional): Whether to use fp8 precision for model parameters. Defaults to False.
    """
    from nemo.collections.nlp.modules.common.megatron.megatron_init import (
        initialize_model_parallel_for_nemo,
    )
    from nemo.utils import AppState

    app_state = AppState()

    if os.environ.get(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, "false").lower() == "true":
        init_world_size = (
            app_state.tensor_model_parallel_size * app_state.pipeline_model_parallel_size
        )
        init_global_rank = app_state.global_rank
        init_local_rank = app_state.local_rank
    else:
        init_world_size = world_size
        init_global_rank = global_rank
        init_local_rank = local_rank

    initialize_model_parallel_for_nemo(
        world_size=init_world_size,
        global_rank=init_global_rank,
        local_rank=init_local_rank,
        tensor_model_parallel_size=parallel_config.tensor_model_parallel_size,
        pipeline_model_parallel_size=parallel_config.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=parallel_config.virtual_pipeline_model_parallel_size,
        seed=seed,
        pipeline_model_parallel_split_rank=getattr(
            parallel_config, "pipeline_model_parallel_split_rank", None
        ),
        use_fp8=fp8,
        init_mpi_proc_group=getattr(parallel_config, "ub_tp_comm_overlap", False),
        # apex_transformer_log_level=self.cfg.get('apex_transformer_log_level', 30),
    )


def initialize_data(global_rank: int, config: "DataConfig") -> None:
    """
    Initializes the data for distributed training by setting up the microbatch calculator
    based on the provided global rank and data configuration.

    This function checks if the microbatch calculator has already been initialized. If it has,
    the function validates that the current configuration matches the initialized settings. If the
    calculator has not been initialized, it sets up a new one with the provided configuration.

    Args:
        global_rank (int): The global rank of the current process.
        config (DataConfig): The data configuration object containing settings for global batch size,
            micro batch size, data parallel size, and optional ramp-up batch size.

    Raises
    ------
        Exception: If the microbatch calculator has already been initialized with different settings.

    """
    from nemo.utils import AppState

    app_state = AppState()

    if os.environ.get(NEMO_MEGATRON_MODEL_PARALLEL_APPSTATE_OVERRIDE, "false").lower() == "true":
        init_global_rank = app_state.global_rank
    else:
        init_global_rank = global_rank

    from apex.transformer.microbatches import ConstantNumMicroBatches
    from apex.transformer.pipeline_parallel.utils import (
        _GLOBAL_NUM_MICROBATCHES_CALCULATOR,
        setup_microbatch_calculator,
    )

    if _GLOBAL_NUM_MICROBATCHES_CALCULATOR is None:
        setup_microbatch_calculator(
            rank=init_global_rank,
            global_batch_size=config.global_batch_size,
            micro_batch_size=config.micro_batch_size,
            data_parallel_size=app_state.data_parallel_size,
            rampup_batch_size=config.rampup_batch_size,
        )
    else:
        if isinstance(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, ConstantNumMicroBatches):
            assert (
                _GLOBAL_NUM_MICROBATCHES_CALCULATOR.current_global_batch_size
                == config.global_batch_size
            )
            assert _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size == config.micro_batch_size
            assert (
                _GLOBAL_NUM_MICROBATCHES_CALCULATOR.num_micro_batches
                == config.global_batch_size
                // (config.micro_batch_size * app_state.data_parallel_size)
            )
        else:
            raise Exception("Microbatch calculator already initialized.")


def init_model_parallel(model: Optional[nn.Module] = None) -> None:
    """Initializes Megatron-LM model parallel if using model parallelism."""
    import torch.distributed
    from megatron.core import mpu, parallel_state
    from nemo.utils import AppState

    app_state = AppState()

    # we initialize megatron-lm model parallel and data parallel groups
    # after initializing DDP with PTL.
    if app_state.model_parallel_size is not None:
        # destroy groups in case they have already been created
        # this happens with multiple calls to trainer.test for example
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=app_state.tensor_model_parallel_size,
                pipeline_model_parallel_size=app_state.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=app_state.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=app_state.pipeline_model_parallel_split_rank,
            )

            # assert that fake tp and pp rank match after model parallel init
            assert app_state.tensor_model_parallel_rank == parallel_state.get_tensor_model_parallel_rank()
            assert app_state.pipeline_model_parallel_rank == parallel_state.get_pipeline_model_parallel_rank()

            app_state.tensor_model_parallel_group = parallel_state.get_tensor_model_parallel_group()
            app_state.data_parallel_group = parallel_state.get_data_parallel_group()
            app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
            app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
            app_state.pipeline_model_parallel_group = parallel_state.get_pipeline_model_parallel_group()

            # create MPI process group for UCX-based communication APIs
            if app_state.init_mpi_proc_group:
                torch.distributed.new_group(backend="mpi")

        if model:
            """Set TP group
            Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py#L398
            """
            # Deep iterate but skip self to avoid infinite recursion.
            for index, child in enumerate(model.modules()):
                if index == 0:
                    continue
                if hasattr(child, "set_tensor_parallel_group"):
                    tp_group = mpu.get_tensor_model_parallel_group()
                    child.set_tensor_parallel_group(tp_group)


def process_dataloader(dataloader: DataLoader, data_config: "DataConfig") -> DataLoader:
    """
    Processes the given DataLoader by wrapping it with a MegatronPretrainingSampler if it is not
    already using one. This sampler is responsible for preparing the data for distributed training
    with Megatron-LM.

    Args:
        dataloader (DataLoader): The DataLoader to be processed.
        data_config (DataConfig): An object containing the data configuration, particularly the 
            micro_batch_size, global_batch_size, and rampup_batch_size.

    Returns
    -------
        DataLoader: A DataLoader wrapped with MegatronPretrainingSampler if the original DataLoader 
        was not already using one, otherwise the original DataLoader is returned.
    """
    from megatron.core import parallel_state
    from megatron_ext.sampler import MegatronPretrainingSampler
    
    if isinstance(dataloader.batch_sampler, MegatronPretrainingSampler):
       return dataloader 
    
    batch_sampler = MegatronPretrainingSampler(
        total_samples=len(dataloader.dataset),
        consumed_samples=0,
        micro_batch_size=data_config.micro_batch_size,
        global_batch_size=data_config.global_batch_size,
        rampup_batch_size=data_config.rampup_batch_size,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
        drop_last=getattr(dataloader, "_drop_last", False),
        pad_samples_to_global_batch_size=getattr(
            dataloader, "_pad_samples_to_global_batch_size", False
        ),
    )
    batch_sampler.batch_size = batch_sampler.global_batch_size

    return DataLoader(
        dataloader.dataset,
        batch_sampler=batch_sampler,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        persistent_workers=dataloader.persistent_workers,
        collate_fn=dataloader.collate_fn
    )
    
    
@contextmanager
def megatron_lazy_init_context(config) -> Generator[None, None, None]:
    def monkey_patched(c):
        return {"device": "meta"}

    from megatron.core.transformer.custom_layers import transformer_engine as _te

    original = _te._get_extra_te_kwargs   # noqa: SLF001
    _te._get_extra_te_kwargs = monkey_patched   # noqa: SLF001
    
    _orig_perform_initialization = config.perform_initialization
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.perform_initialization = False
    config.use_cpu_initialization = True

    yield

    _te._get_extra_te_kwargs = original   # noqa: SLF001
    config.perform_initialization = _orig_perform_initialization
    config.use_cpu_initialization = _orig_use_cpu_initialization
    
    
@contextmanager
def megatron_cpu_init_context(config) -> Generator[None, None, None]:
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.use_cpu_initialization = True

    yield

    config.use_cpu_initialization = _orig_use_cpu_initialization
    

ModelT = TypeVar("ModelT", bound=nn.Module)


@overload
def setup_megatron_parallel(
    trainer: pl.Trainer, 
    model: Union[ModelT, Callable[[], ModelT]], 
    cpu: bool = False, **kwargs
) -> "MegatronParallel":
    ...
    

@overload
def setup_megatron_parallel(
    trainer: fl.Fabric, 
    model: Union[ModelT, Callable[[], ModelT]], 
    cpu: bool = False, **kwargs
) -> "MegatronParallel":
    ...


@overload
def setup_megatron_parallel(
    trainer: ParallelStrategy, 
    model: Union[ModelT, Callable[[], ModelT]], 
    cpu: bool = False, **kwargs
) -> "MegatronParallel":
    ...


def setup_megatron_parallel(
    trainer: Union[pl.Trainer, fl.Fabric, ParallelStrategy], 
    model: Union[ModelT, Callable[[], ModelT]], 
    cpu: bool = False, 
    wrap_with_ddp: bool = False,
    **kwargs
) -> "MegatronParallel":
    """
    Sets up the Megatron parallel environment for the given trainer and model.

    This function configures the Megatron parallel environment based on the type of trainer
    provided. It initializes the parallel ranks, sets the CUDA device if not using CPU, and
    builds the Megatron module with the appropriate parallelism and wrapper class.

    Args:
        trainer (Union[pl.Trainer, fl.Fabric, ParallelStrategy]): The trainer object for which to set up
            the Megatron parallel environment.
        model (Union[ModelT, Callable[[], ModelT]]): The model or a callable that returns the model to
            be parallelized.
        cpu (bool): Whether to use CPU for training. Defaults to False.
        wrap_with_ddp (bool): Whether to wrap the model with DistributedDataParallel. Defaults to False.
        **kwargs: Additional keyword arguments.

    Returns
    -------
        MegatronParallel: The Megatron parallelized model.

    Raises
    ------
        ValueError: If the trainer's strategy is not supported.

    Examples
    --------
        >>> # Example with pl.Trainer
        >>> trainer = pl.Trainer(...)
        >>> model = MyModel(...)
        >>> megatron_model = setup_megatron_parallel(trainer, model)

        >>> # Example with fl.Fabric
        >>> fabric = fl.Fabric(...)
        >>> model = MyModel(...)
        >>> megatron_model = setup_megatron_parallel(fabric, model)
    """
    from nemo.lightning import FabricMegatronStrategy, MegatronStrategy

    if isinstance(trainer, pl.Trainer):
        if not hasattr(trainer.strategy, "parallelism"):
            raise ValueError("Only `MegatronStrategy` is supported for now. Found: ", trainer.strategy)
        parallelism = cast(MegatronStrategy, trainer.strategy).parallelism
    elif isinstance(trainer, fl.Fabric):
        if not isinstance(trainer.strategy, FabricMegatronStrategy):
            raise ValueError("Only `FabricMegatronStrategy` is supported for now. Found: ", trainer)
        parallelism = cast(FabricMegatronStrategy, trainer.strategy).parallelism
    else:
        if not isinstance(trainer, FabricMegatronStrategy):
            raise ValueError("Only `FabricMegatronStrategy` is supported for now. Found: ", trainer)
        parallelism = cast(FabricMegatronStrategy, trainer).parallelism

    if not cpu:
        torch.cuda.set_device(trainer.local_rank)

    init_parallel_ranks(
        trainer.world_size, trainer.global_rank, trainer.local_rank, parallelism, **kwargs
    )
    # if isinstance(trainer, pl.Trainer):
    #     wrapper_cls = LightningMegatronParallel
        
    output = _build_megatron_module(
        model,
        parallelism=parallelism,
        cpu=cpu,
        wrap_with_ddp=wrap_with_ddp
    )

    return output


def _build_megatron_module(
    model: Union[ModelT, Callable[[], ModelT]],
    parallelism: "ModelParallelConfig",
    cpu: bool = False,
    wrap_with_ddp: bool = False
) -> "MegatronParallel":
    from apex.transformer.tensor_parallel.layers import (
        set_defaults_if_not_set_tensor_model_parallel_attributes,
    )
    from megatron.core import mpu, parallel_state
    from megatron_ext.megatron_parallel import MegatronParallel
    from nemo.utils import logging

    build_model = functools.partial(init_lightning_module, model=model)
    pipeline: List[nn.Module] = []
    virtual_pipeline_model_parallel_size = parallelism.virtual_pipeline_model_parallel_size
    wrapper_cls = MegatronParallel

    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and virtual_pipeline_model_parallel_size is not None
    ):
        mpu.set_virtual_pipeline_model_parallel_world_size(virtual_pipeline_model_parallel_size)
        for i in range(virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            pipeline.append(build_model(copy=True))
    else:
        pipeline.append(build_model())

    # TODO: Do we still need this?
    # for model_module in output:
    #     model_module.model_type = model_type

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in pipeline:
        for param in model_module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.model_parallel_is_initialized() and mpu.get_data_parallel_rank() == 0:
        msg = (
            f" > number of parameters on (tensor, pipeline) model parallel rank "
            f"({mpu.get_tensor_model_parallel_rank()}, {mpu.get_pipeline_model_parallel_rank()}): "
            f"{_calc_number_of_params(pipeline)}"
        )
        logging.info(msg)

    # GPU allocation.
    if not cpu:
        for model_module in pipeline:
            model_module.cuda(torch.cuda.current_device())

    if wrap_with_ddp:
        i = torch.cuda.current_device()
        pipeline = [
            torch.nn.parallel.distributed.DistributedDataParallel(
                model_module, device_ids=[i], output_device=i, process_group=mpu.get_data_parallel_group(),
            )
            for model_module in pipeline
        ]

    return wrapper_cls(*pipeline)


def init_lightning_module(model: Union[ModelT, Callable[[], ModelT]], copy: bool = False) -> ModelT:
    module: ModelT
    if isinstance(model, nn.Module):
        if copy:
            if isinstance(model, nn.Module):
                module = io.reinit(model)
            else:
                module = model()
        else:
            module = cast(ModelT, model)
    else:
        module = model()

    return module


def _calc_number_of_params(model: List[nn.Module]) -> int:
    assert isinstance(model, list)

    return sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model])


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self,
        init_scale=2.0**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        hysteresis=1,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.optimizer_update_skipped: Optional[bool] = None
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    def _unscale_grads_(self, optimizer, *args):
        if getattr(optimizer, "_custom_amp_unscale_grads", False):
            return optimizer.unscale_grads(*args)
        else:
            return super()._unscale_grads_(optimizer, *args)

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        from megatron.core import parallel_state
        
        retval = None
        found_inf = torch.cuda.FloatTensor(
            [sum(v.item() for v in optimizer_state["found_inf_per_device"].values())]
        )

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf,
            op=torch.distributed.ReduceOp.MAX,
            group=parallel_state.get_model_parallel_group(),
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
            self.optimizer_update_skipped = False
        else:
            self.optimizer_update_skipped = True
        return retval

    def update(self, new_scale=None):
        """
        Updates to native grad scaler update function.
        1. Check inf across model-parallel ranks.
        2. Update hysteresis tracker.
        3. Apply hysteresis to grad scale update.
        """
        from megatron.core import parallel_state
        
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = (
                    "new_scale should be a float or a 1-element torch.cuda.FloatTensor with"
                    " requires_grad=False."
                )
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined,
                op=torch.distributed.ReduceOp.MAX,
                group=parallel_state.get_model_parallel_group(),
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_model_parallel_group(),
                    )
                    found_inf_combined += found_inf

            if found_inf_combined > 0:
                self._hysteresis_tracker -= 1
                if self._hysteresis_tracker <= 0:
                    # When hysteresis becomes zero, follow the native grad scale update rule.
                    # Increase scale and reset growth tracker
                    torch._amp_update_scale_(   # noqa: SLF001
                        _scale,
                        _growth_tracker,
                        found_inf_combined,
                        self._growth_factor,
                        self._backoff_factor,
                        self._growth_interval,
                    )
                else:
                    # Only reset the growth tracker when hysteresis is larger than zero
                    _growth_tracker.fill_(0.0)
            else:
                # When no inf found, follow the native grad scale update rule.
                # Increment growth_tracker, update scale when growth tracker reaches the interval, and
                # reset the hysteresis tracker.
                torch._amp_update_scale_(   # noqa: SLF001
                    _scale,
                    _growth_tracker,
                    found_inf_combined,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
                self._hysteresis_tracker = self.hysteresis

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(
            torch.cuda.amp.grad_scaler._refresh_per_optimizer_state   # noqa: SLF001
        )

    def state_dict(self):
        """
        Add hysteresis_tracker to the native functions' state_dict.
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_hysteresis_tracker": self._hysteresis_tracker,
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Load hysteresis_tracker in addition to the state dict of the native function.
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        if "_hysterisis_tracker" in state_dict:
            self._hysteresis_tracker = state_dict["_hysterisis_tracker"]
        else:
            self._hysteresis_tracker = 1


def enable_nvidia_optimizations() -> None:
    """These optimizations are present in NVIDIA NGC PyTorch Containers."""
    # NVIDIA container version check
    nvidia_torch_version = os.getenv("NVIDIA_PYTORCH_VERSION", None)
    if nvidia_torch_version is not None:
        try:
            NVIDIA_TORCH_MAJOR = int(nvidia_torch_version.split(".")[0])
        except Exception:
            NVIDIA_TORCH_MAJOR = 0
        try:
            NVIDIA_TORCH_MINOR = int(nvidia_torch_version.split(".")[1])
        except Exception:
            NVIDIA_TORCH_MINOR = 0

        # NVFUSER available starting with 21.11
        if NVIDIA_TORCH_MAJOR >= 21 or (NVIDIA_TORCH_MAJOR == 21 and NVIDIA_TORCH_MINOR >= 11):
            # NVFUSER
            torch._C._jit_set_profiling_executor(True)   # noqa: SLF001
            torch._C._jit_set_profiling_mode(True)   # noqa: SLF001
            torch._C._jit_override_can_fuse_on_cpu(False)   # noqa: SLF001
            torch._C._jit_override_can_fuse_on_gpu(False)   # noqa: SLF001
            torch._C._jit_set_texpr_fuser_enabled(False)   # noqa: SLF001
            # torch._C._jit_set_nvfuser_enabled(True)
            torch._C._debug_set_autodiff_subgraph_inlining(False)   # noqa: SLF001
    else:
        # Not a Nvidia container. NVFUSER Dependency check is on users
        pass


def optimizer_sharded_state_dict(
    model: SharedStateDictProtocol, optimizer: Optimizable
) -> Dict[str, torch.Tensor]:
    """
    Sharded state dictionary for an MainParamsOptimizerWrapper.
    Used to save and load the optimizer state when training with distributed_checkpoint.

    Returns
    -------
        dict: The sharded state dictionary for the optimizer
    Raises:
        ValueError: If a parameter ID does not match any model sharded parameter.
    """
    from megatron.core.dist_checkpointing.optimizer import (
        get_param_id_to_sharded_param_map,
        make_sharded_optimizer_tensor,
        optim_state_to_sharding_state,
    )
    from nemo.core.optim import MainParamsOptimizerWrapper
    from nemo.core.optim.optimizers import init_optimizer_states
    
    model_sharded_state_dict = model.sharded_state_dict()

    # remove _extra_state
    model_sharded_state_dict = {
        key: value
        for key, value in model_sharded_state_dict.items()
        if not key.endswith("_extra_state")
    }

    if hasattr(optimizer, "sharded_state_dict"):
        return optimizer.sharded_state_dict(model_sharded_state_dict)
    
    if not isinstance(optimizer, MainParamsOptimizerWrapper):
        # Regular optimizer, e.g. Adam or FusedAdam
        init_optimizer_states(optimizer)
        optimizer_state_dict = optimizer.state_dict()
        id_to_sharded_param_map = get_param_id_to_sharded_param_map(
            model_sharded_state_dict=model_sharded_state_dict,
            optim_params_iter=itertools.chain.from_iterable(g['params'] for g in optimizer.param_groups),
        )
        optim_state_to_sharding_state(optimizer_state_dict, id_to_sharded_param_map)
        return optimizer_state_dict

    optimizer_state_dict: Dict[str, Any] = optimizer.state_dict()

    id_to_sharded_param_map = get_param_id_to_sharded_param_map(
        model_sharded_state_dict=model_sharded_state_dict,
        optim_params_iter=itertools.chain.from_iterable(g for g in optimizer.float16_groups),
    )

    # Convert fp32_from_fp16_params
    assert len(optimizer_state_dict["fp32_from_fp16_params"]) == len(
        optimizer_state_dict["optimizer"]["param_groups"]
    )

    def get_safe(param_id):
        try:
            return id_to_sharded_param_map[param_id]
        except KeyError as e:
            raise ValueError(f"Param id {param_id} does not match any model sharded param") from e

    optimizer_state_dict["fp32_from_fp16_params"] = [
        [
            make_sharded_optimizer_tensor(
                get_safe(param_id), fp32_param, prefix="optimizer.state.fp32_param"
            )
            for param_id, fp32_param in zip(state_group["params"], fp32_group)
        ]
        for fp32_group, state_group in zip(
            optimizer_state_dict["fp32_from_fp16_params"],
            optimizer_state_dict["optimizer"]["param_groups"],
        )
    ]

    # Convert state
    optim_state_to_sharding_state(optimizer_state_dict["optimizer"], id_to_sharded_param_map)

    return optimizer_state_dict
