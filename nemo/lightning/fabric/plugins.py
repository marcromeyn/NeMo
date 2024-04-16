from contextlib import contextmanager
from typing import Any, Generator, Literal, Optional, TypeVar, Union

import torch
from lightning.fabric.plugins.precision import MixedPrecision
from lightning.fabric.utilities.types import Optimizable
from torch import nn
from torch.optim import Optimizer

from nemo.lightning._strategy_lib import GradScaler

AnyT = TypeVar("AnyT")


class FabricMegatronMixedPrecision(MixedPrecision):
    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"] = "16-mixed",
        amp_02: bool = True,
        device="cuda",
        scaler: Optional[Union[torch.cuda.amp.GradScaler, str]] = None,
    ) -> None:
        if precision == "bf16-mixed":
            scaler = None
        else:
            scaler = GradScaler(
                init_scale=2**32,
                growth_interval=1000,
                hysteresis=2,
            )

        super().__init__(precision, device, scaler)
        dtype = None
        # MixedPrecisionPlugin class in PTL >= 2.0 takes only "16-mixed" or "bf16-mixed" for precision arg
        if precision == "16-mixed":
            dtype = torch.float16

            def float16_convertor(val):
                return val.half()

        elif precision == "bf16-mixed":
            dtype = torch.bfloat16

            def float16_convertor(val):
                return val.bfloat16()

        torch.set_autocast_gpu_dtype(dtype)
        self.float16_convertor = float16_convertor
        self.amp_02 = amp_02

    def convert_input(self, data: AnyT) -> AnyT:
        """Convert model inputs (forward) to the floating point precision type of this plugin.

        Note: MegatronStrategy will take care of only doing this when:
            mpu.is_pipeline_first_stage()

        """
        from megatron.core.transformer.module import fp32_to_float16
        
        return fp32_to_float16(data, self.float16_convertor)

    def convert_output(self, data: AnyT) -> AnyT:
        """Convert outputs to the floating point precision type expected after model's forward.

        Note: MegatronStrategy will take care of only doing this when:
            mpu.is_pipeline_first_stage()

        """
        from megatron.core.transformer.module import float16_to_fp32
        
        return float16_to_fp32(data)

    def setup_optimizer(self, optimizer: Optimizer) -> Optimizer:
        from nemo.core.optim import MainParamsOptimizerWrapper
        
        return MainParamsOptimizerWrapper(
            optimizer,
            # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_base_model.py#L496
            fp32_grad_accum=True,
            contiguous_grad_bucket=True,
        )

    def convert_module(self, module: nn.Module) -> nn.Module:
        """Convert the module parameters to the precision type this plugin handles.

        This is optional and depends on the precision limitations during optimization.

        """
        if self.precision == "bf16-mixed":
            return module.bfloat16()
        if self.precision == "16-mixed":
            return module.half()

        return module

    def optimizer_step(
        self,
        optimizer: Optimizable,
        **kwargs: Any,
    ) -> None:
        from nemo.core.optim import MainParamsOptimizerWrapper
        
        assert isinstance(
            optimizer, MainParamsOptimizerWrapper
        ), "MegatronHalfPrecisionPlugin supports only the optimizer with master parameters"

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"

            # skip scaler logic, as bfloat16 does not require scaler
            return super().optimizer_step(optimizer, **kwargs)

        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()

        # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
        step_output = self.scaler.step(optimizer, **kwargs)
        self.scaler.update()

        return step_output

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """No explicit precision casting. Inputs are supposed to be manually casted."""
        try:
            yield
        finally:
            pass
