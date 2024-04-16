import gc
import os
from typing import Any

import pytorch_lightning as L
from pytorch_lightning.utilities.types import STEP_OUTPUT


class GarbageCollection(L.Callback):
    def __init__(self, gc_interval: int = 0) -> None:
        super().__init__()
        self.gc_interval = gc_interval
        self.gc_in_validation = bool(int(os.getenv("NEMO_MANUAL_GC_IN_VALIDATION", '1')))
        assert (
            self.gc_interval >= 0
        ), "gc_interval should be an integer value larger than or equal to 0."
        # If gc_interval > 0, memory garbage collection is manually controlled.
        # The automatic garbage collector sould be disabled before training starts.
        if self.gc_interval > 0:
            gc.disable()
            self.validation_global_step = 1

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.gc_interval > 0 and (trainer.global_step % self.gc_interval == 0):
            gc.collect()

    def on_validation_batch_end(self, pl_module: L.LightningModule) -> None:
        if self.gc_interval > 0 and self.gc_in_validation:
            if self.validation_global_step % self.gc_interval == 0:
                gc.collect()
            pl_module.validation_global_step += 1

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.gc_interval > 0 and self.gc_in_validation:
            gc.collect()

    def on_validation_end(self) -> None:
        if self.gc_interval > 0 and self.gc_in_validation:
            gc.collect()
