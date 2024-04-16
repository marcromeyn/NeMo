from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar

import lightning.fabric as fl
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from nemo.lightning.base import DataConfig

DataConfigT = TypeVar("DataConfigT", bound=DataConfig)
BatchIterator = Any


if TYPE_CHECKING:
    from nemo.core.classes.dataset import Dataset
    
    
class MegatronDataMixin:
    rampup_batch_size: Optional[List[int]] = None
    
    def compute_consumed_samples(self, steps_since_resume=0) -> int:
        from nemo.utils import AppState

        from nemo.lightning.pytorch.strategies import MegatronStrategy

        if not isinstance(self.trainer.strategy, MegatronStrategy):
            return 0

        app_state = AppState()

        if self.config.rampup_batch_size is not None:
            from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

            current_global_batch_size = getattr(
                _GLOBAL_NUM_MICROBATCHES_CALCULATOR, "current_global_batch_size", 1
            )
            consumed_samples = (
                self.prev_consumed_samples + self.if_first_step * current_global_batch_size
            )
        else:
            consumed_samples = (
                self.init_consumed_samples
                + steps_since_resume
                * app_state.data_parallel_size
                * self.config.micro_batch_size
                * self.config.num_microbatches
            )

        return int(consumed_samples)

    # Megatron callbacks
    def on_megatron_step_start(self, trainer: pl.Trainer) -> None:
        # do validation and save the checkpoint when gbs is changed
        if (
            self.config.rampup_batch_size is not None
            and self.prev_global_batch_size != self.current_global_batch_size
            and self.prev_global_batch_size
        ):
            trainer.should_stop = True

    def on_megatron_step_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import apex.transformer.pipeline_parallel.utils

        if self.config.rampup_batch_size is None:
            return

        self.prev_global_batch_size = self.current_global_batch_size

        # TODO: Add consumed samples
        consumed_samples = self.compute_consumed_samples(
            trainer.global_step + 1 - self.init_global_step
        )

        self.prev_consumed_samples = consumed_samples

        num_microbatch_calculator = (
            apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR   # noqa: SLF001
        )

        num_microbatch_calculator.update(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        current_global_batch_size = num_microbatch_calculator.current_global_batch_size
        pl_module.log(
            "global_batch_size",
            current_global_batch_size,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        self.if_first_step = 1


class DataModule(pl.LightningDataModule, Generic[DataConfigT]):
    def __init__(
        self,
        config: DataConfigT,
    ) -> None:
        super().__init__()
        self.config: DataConfigT = config
        self.init_consumed_samples: int = 0
        self.prev_consumed_samples = 0
        self.if_first_step = 0
        self.prev_global_batch_size = None

    def setup(self, stage: str):
        if stage == "fit":
            self.init_global_step = self.trainer.global_step
            
    def fabric_setup(
        self, 
        fabric: fl.Fabric,
        num_train_samples: int,
        num_val_samples: int,
        num_test_samples: int,
    ) -> None:
        with fabric.rank_zero_first():
            self.prepare_data()

        self.setup("fit")
        
        callbacks = getattr(fabric.strategy, "megatron_callbacks", None)
        if callbacks:
            callbacks.add(self)

    def to_dataloader(
        self, dataset: "Dataset", drop_last: bool = True, pad_samples_to_global_batch_size=False
    ) -> DataLoader:
        output = DataLoader(
            dataset,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            collate_fn=dataset.collate_fn
        )

        output._drop_last = drop_last   # noqa: SLF001
        output._pad_samples_to_global_batch_size = pad_samples_to_global_batch_size   # noqa: SLF001

        return output

    def compute_consumed_samples(self, steps_since_resume=0) -> int:
        from nemo.utils import AppState

        from nemo.lightning.pytorch.strategies import MegatronStrategy

        if not isinstance(self.trainer.strategy, MegatronStrategy):
            return 0

        app_state = AppState()

        if self.config.rampup_batch_size is not None:
            from apex.transformer.pipeline_parallel.utils import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

            current_global_batch_size = getattr(
                _GLOBAL_NUM_MICROBATCHES_CALCULATOR, "current_global_batch_size", 1
            )
            consumed_samples = (
                self.prev_consumed_samples + self.if_first_step * current_global_batch_size
            )
        else:
            consumed_samples = (
                self.init_consumed_samples
                + steps_since_resume
                * app_state.data_parallel_size
                * self.config.micro_batch_size
                * self.config.num_microbatches
            )

        return int(consumed_samples)

    # Megatron callbacks
    def on_megatron_step_start(self, trainer: pl.Trainer) -> None:
        # do validation and save the checkpoint when gbs is changed
        if (
            self.config.rampup_batch_size is not None
            and self.prev_global_batch_size != self.current_global_batch_size
            and self.prev_global_batch_size
        ):
            trainer.should_stop = True

    def on_megatron_step_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        import apex.transformer.pipeline_parallel.utils

        if self.config.rampup_batch_size is None:
            return

        self.prev_global_batch_size = self.current_global_batch_size

        # TODO: Add consumed samples
        consumed_samples = self.compute_consumed_samples(
            trainer.global_step + 1 - self.init_global_step
        )

        self.prev_consumed_samples = consumed_samples

        num_microbatch_calculator = (
            apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR   # noqa: SLF001
        )

        num_microbatch_calculator.update(
            consumed_samples=consumed_samples,
            consistency_check=False,
        )
        current_global_batch_size = num_microbatch_calculator.current_global_batch_size
        pl_module.log(
            "global_batch_size",
            current_global_batch_size,
            prog_bar=True,
            rank_zero_only=True,
            batch_size=1,
        )
        self.if_first_step = 1

    def model_kwargs(self) -> Dict[str, Any]:
        return {}

    @property
    def current_global_batch_size(self) -> int:
        import apex.transformer.pipeline_parallel.utils

        num_microbatch_calculator = (
            apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR   # noqa: SLF001
        )
        current_global_batch_size = num_microbatch_calculator.current_global_batch_size

        return current_global_batch_size

    def train_dataloader(self) -> DataLoader:
        return self.to_dataloader(self._train_ds, drop_last=self.config.train_drop_last)

    def val_dataloader(self) -> DataLoader:
        return self.to_dataloader(self._validation_ds, drop_last=self.config.val_drop_last)

    def test_dataloader(self) -> DataLoader:
        return self.to_dataloader(self._test_ds, drop_last=self.config.test_drop_last)



class NLPDataModule(DataModule[DataConfigT], Generic[DataConfigT]):
    def __init__(self, config: DataConfigT, tokenizer) -> None:
        super().__init__(config)
        self.tokenizer = tokenizer

    def model_kwargs(self) -> Dict[str, Any]:
        return {"tokenizer": self.tokenizer}
    
    def setup(self, stage: str = ""):
        pass


__all__ = ["BatchIterator", "DataModule", "NLPDataModule"]
