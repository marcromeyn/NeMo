from typing import Union

from lightning.pytorch import plugins as _pl_plugins
from lightning_fabric.plugins.environments import slurm

from nemo.lightning._strategy_lib import setup_megatron_parallel
from nemo.lightning.base import DataConfig, ModelConfig, ModelParallelism, teardown
from nemo.lightning.fabric.plugins import FabricMegatronMixedPrecision
from nemo.lightning.fabric.strategies import FabricMegatronStrategy
from nemo.lightning.pytorch.core.datamodule import BatchIterator, DataModule, NLPDataModule
from nemo.lightning.pytorch.plugins import (
    MegatronDataSampler,
    MegatronMixedPrecision,
    data_sampler as _data_sampler,
)
from nemo.lightning.pytorch.strategies import MegatronStrategy
from nemo.lightning.pytorch.trainer import Trainer


# We monkey patch because nvidia uses a naming convention for SLURM jobs
def _is_slurm_interactive_mode():
    job_name = slurm.SLURMEnvironment.job_name()
    return job_name is None or job_name.endswith("bash") or job_name.endswith("interactive")

slurm._is_slurm_interactive_mode = _is_slurm_interactive_mode   # noqa: SLF001


_pl_plugins._PLUGIN_INPUT = Union[_pl_plugins._PLUGIN_INPUT, _data_sampler.DataSampler] # noqa: SLF001


__all__ = [
    "BatchIterator",
    "DataConfig",
    "DataModule",
    "FabricMegatronStrategy",
    "FabricMegatronMixedPrecision",
    "ModelConfig",
    "ModelParallelism",
    "MegatronStrategy",
    "MegatronMixedPrecision",
    "MegatronDataSampler",
    "NLPDataModule",
    "setup_megatron_parallel",
    "teardown",
    "Trainer"
]
