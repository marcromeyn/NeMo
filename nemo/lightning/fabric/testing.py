from dataclasses import dataclass
from typing import Optional, cast

import pytest
import torch
from lightning import Fabric
from megatron_ext import ModelParallelism

from nemo.lightning.base import DataConfig
from nemo.lightning.fabric.plugins import FabricMegatronMixedPrecision
from nemo.lightning.fabric.strategies import FabricMegatronStrategy


@dataclass
class MegatronFabric:
    parallelism: ModelParallelism
    data_config: DataConfig
    fabric: Fabric

    @classmethod
    def from_config(
        cls,
        parallelism: ModelParallelism,
        data_config: Optional[DataConfig] = None,
        plugins=None,
        **fabric_kwargs
    ) -> "MegatronFabric":
        _data_config = data_config or DataConfig(512, num_workers=1, global_batch_size=32)
        fabric = Fabric(
            strategy=FabricMegatronStrategy(parallelism, _data_config),
            plugins=plugins,
            **fabric_kwargs
        )

        return cls(parallelism, _data_config, fabric)

    @classmethod
    def launch(
        cls,
        devices: int,
        parallelism: ModelParallelism,
        data_config: Optional[DataConfig] = None,
        plugins=None,
        **fabric_kwargs
    ) -> "MegatronFabric":
        output = cls.from_config(devices, parallelism, data_config, plugins, **fabric_kwargs)
        output.fabric.launch()

        return output

    @property
    def seq_length(self) -> int:
        return self.data_config.seq_length
    
    @property
    def strategy(self) -> FabricMegatronStrategy:
        return cast(FabricMegatronStrategy, self.fabric.strategy)


@pytest.fixture(autouse=True)
def tp2_config() -> ModelParallelism:
    return ModelParallelism(
        tensor_model_parallel_size=2,
    )

@pytest.fixture(autouse=True)
def tp1_config() -> ModelParallelism:
    return ModelParallelism(
        tensor_model_parallel_size=1,
    )


@pytest.fixture(autouse=True)
def pp2_config() -> ModelParallelism:
    return ModelParallelism(
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )


@pytest.fixture(autouse=True)
def vpp2_config() -> ModelParallelism:
    return ModelParallelism(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )


@pytest.fixture(autouse=True)
def fabric_tp2(tp2_config) -> MegatronFabric:
    if not torch.cuda.device_count() >= 2:
        pytest.skip("Requires at least 2 GPUs")

    plugins = [FabricMegatronMixedPrecision("bf16-mixed")]

    return MegatronFabric.from_config(parallelism=tp2_config, plugins=plugins)


@pytest.fixture(autouse=True)
def fabric_pp2(pp2_config) -> MegatronFabric:
    if not torch.cuda.device_count() >= 2:
        pytest.skip("Requires at least 2 GPUs")

    return MegatronFabric.from_config(parallelism=pp2_config)
