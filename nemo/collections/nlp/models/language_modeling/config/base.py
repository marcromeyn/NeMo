from typing import Any, Optional, List
from dataclasses import dataclass, field


class BaseConfig:
    # Mimic OmegaConf API for backwards compatibility
    def get(self, attribute_name: str, default=None):
        """Retrieves the value of the specified attribute or returns the default value if not found."""
        return getattr(self, attribute_name, default)
    
    def __contains__(self, key) -> bool:
        return key in self.__dict__.values()
    
    def _set_flag(self, name: str, value: bool):
        pass
    
    def _get_node_flag(self, name: str):
        return True


@dataclass
class AMPConfig(BaseConfig):
    """Configuration for AMP (Automatic Mixed Precision) settings.

    Attributes:
        megatron_amp_O2 (bool): 
            Enable O2-level automatic mixed precision using main parameters. Default is False.
        grad_allreduce_chunk_size_mb (int): 
            Chunk size in MB for gradient all-reduce. Default is 125.
        native_amp_init_scale (int): 
            Initial scale for native AMP. Default is 2 ** 32.
        native_amp_growth_interval (int): 
            Growth interval for native AMP. Default is 1000.
        hysteresis (int): 
            Gradient scale hysteresis. Default is 2.
        fp32_residual_connection (bool): 
            Move residual connections to fp32. Default is False.
        fp16_lm_cross_entropy (bool): 
            Move the cross entropy unreduced loss calculation for LM head to fp16. Default is False.
    """
    
    # Megatron O2-style half-precision
    megatron_amp_O2: bool = False
    grad_allreduce_chunk_size_mb: int = 125

    # Mixed precision
    native_amp_init_scale: int = 4294967296  # 2 ** 32
    native_amp_growth_interval: int = 1000
    hysteresis: int = 2 
    fp32_residual_connection: bool = False
    fp16_lm_cross_entropy: bool = False


@dataclass
class FusionConfig(BaseConfig):
    """Configuration for various fusion optimizations.

    Attributes:
        grad_div_ar_fusion (bool): 
            Fuse grad division into torch.distributed.all_reduce. 
            Only used with O2 and no pipeline parallelism. Default is True.
        gradient_accumulation_fusion (bool): 
            Fuse weight gradient accumulation to GEMMs. 
            Only used with pipeline parallelism and O2. Default is False.
        bias_activation_fusion (bool): 
            Use a kernel that fuses the bias addition from weight matrices with the 
            subsequent activation function. Default is True.
        bias_dropout_add_fusion (bool): 
            Use a kernel that fuses the bias addition, dropout and residual 
            connection addition. Default is True.
        masked_softmax_fusion (bool): 
            Use a kernel that fuses the attention softmax with its mask. 
            Default is True.
        get_attention_mask_from_fusion (bool): 
            When using fused softmax, it will create the attention mask so we 
            won't copy it to the pipeline stages. Default is True.
    """
    
    grad_div_ar_fusion: bool = True
    gradient_accumulation_fusion: bool = False
    bias_activation_fusion: bool = True
    bias_dropout_add_fusion: bool = True
    masked_softmax_fusion: bool = True
    get_attention_mask_from_fusion: bool = True


@dataclass
class ActivationCheckpointingConfig(BaseConfig):
    """Configuration for activation checkpointing strategies.

    Attributes:
        activations_checkpoint_granularity (Optional[str]): 'selective' or 'full'. Default is None.
        activations_checkpoint_method (Optional[str]): 'uniform', 'block'. Default is None.
        activations_checkpoint_num_layers (Optional[int]): Number of transformer layers to checkpoint. Default is None.
        num_micro_batches_with_partial_activation_checkpoints (Optional[int]): Number of micro-batches with partial activation checkpoints. Default is None.
        activations_checkpoint_layers_per_pipeline (Optional[int]): Number of Transformer layers to skip checkpointing at later pipeline stages. Default is None.
    """
    
    ## Activation Checkpointing
    # NeMo Megatron supports 'selective' activation checkpointing where only the memory intensive part of attention is checkpointed.
    # These memory intensive activations are also less compute intensive which makes activation checkpointing more efficient for LLMs (20B+).
    # See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.
    # 'full' will checkpoint the entire transformer layer.
    activations_checkpoint_granularity: Optional[str] = None  # 'selective' or 'full'
    activations_checkpoint_method: Optional[str] = None  # 'uniform', 'block'
    # 'uniform' divides the total number of transformer layers and checkpoints the input activation
    # of each chunk at the specified granularity. When used with 'selective', 'uniform' checkpoints all attention blocks in the model.
    # 'block' checkpoints the specified number of layers per pipeline stage at the specified granularity
    activations_checkpoint_num_layers: Optional[int] = None
    # when using 'uniform' this creates groups of transformer layers to checkpoint. Usually set to 1. Increase to save more memory.
    # when using 'block' this this will checkpoint the first activations_checkpoint_num_layers per pipeline stage.
    num_micro_batches_with_partial_activation_checkpoints: Optional[int] = None
    # This feature is valid only when used with pipeline-model-parallelism.
    # When an integer value is provided, it sets the number of micro-batches where only a partial number of Transformer layers get checkpointed
    # and recomputed within a window of micro-batches. The rest of micro-batches in the window checkpoint all Transformer layers. The size of window is
    # set by the maximum outstanding micro-batch backpropagations, which varies at different pipeline stages. The number of partial layers to checkpoint
    # per micro-batch is set by 'activations_checkpoint_num_layers' with 'activations_checkpoint_method' of 'block'.
    # This feature enables using activation checkpoint at a fraction of micro-batches up to the point of full GPU memory usage.
    activations_checkpoint_layers_per_pipeline: Optional[int] = None
    # This feature is valid only when used with pipeline-model-parallelism.
    # When an integer value (rounded down when float is given) is provided, it sets the number of Transformer layers to skip checkpointing at later
    # pipeline stages. For example, 'activations_checkpoint_layers_per_pipeline' of 3 makes pipeline stage 1 to checkpoint 3 layers less than
    # stage 0 and stage 2 to checkpoint 6 layers less stage 0, and so on. This is possible because later pipeline stage
    # uses less GPU memory with fewer outstanding micro-batch backpropagations. Used with 'num_micro_batches_with_partial_activation_checkpoints',
    # this feature removes most of activation checkpoints at the last pipeline stage, which is the critical execution path.


@dataclass
class TransformerEngineConfig(BaseConfig):
    """Configuration for the Transformer Engine.

    Attributes:
        transformer_engine (bool): Enable Transformer Engine. Default is False.
        fp8 (bool): Enables fp8 in TransformerLayer forward. Default is False.
        fp8_e4m3 (bool): Sets fp8_format = recipe.Format.E4M3. Default is False.
        fp8_hybrid (bool): Sets fp8_format = recipe.Format.HYBRID. Default is False.
        fp8_margin (int): Scaling margin. Default is 0.
        fp8_interval (int): Scaling update interval. Default is 1.
        fp8_amax_history_len (int): Number of steps for which amax history is recorded per tensor. Default is 1.
        fp8_amax_compute_algo (str): 'most_recent' or 'max'. Algorithm for computing amax from history. Default is "most_recent".
        reduce_amax (bool): Perform reduction to sync amax tensors across GPUs after every iteration. Default is True.
        use_emha (bool): Use fused multi-head attention for large sequence-length. Default is False.
        ub_tp_comm_overlap (bool): Use userbuffer backend to overlap tensor-parallel communications with computes. Default is False.
        ub_tp_comm_overlap_cfg (Optional[Any]): A YAML file with userbuffer communicator configurations. Default is None.
    """
    
    ## Transformer Engine
    transformer_engine: bool = False
    fp8: bool = False
    fp8_e4m3: bool = False
    fp8_hybrid: bool = False
    fp8_margin: int = 0
    fp8_interval: int = 1
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    reduce_amax: bool = True
    use_emha: bool = False
    ub_tp_comm_overlap: bool = False
    ub_tp_comm_overlap_cfg: Optional[Any] = None


@dataclass
class NSysProfilingConfig(BaseConfig):
    """Configuration for NSys profiling.

    Attributes:
        enabled (bool): Enable profiling. Default is False.
        start_step (int): Global batch to start profiling. Default is 10.
        end_step (int): Global batch to end profiling. Default is 10.
        ranks (List[int]): Global rank IDs to profile. Default is [0].
        gen_shape (bool): Generate model and kernel details including input shapes. 
            Default is False.
    """
    
    enabled: bool = False
    start_step: int = 10
    end_step: int = 10
    ranks: List[int] = field(default_factory=lambda: [0])
    gen_shape: bool = False

    
@dataclass
class SchedulerConfig(BaseConfig):
    """
    Configuration for the learning rate scheduler.

    Attributes:
        name (str): The name of the scheduler. Example: 'CosineAnnealing'.
        warmup_steps (int): Number of steps for the warm-up phase of the scheduler.
        constant_steps (int): Number of steps for which the learning rate is held constant.
        min_lr (float): Minimum learning rate that can be reached by the scheduler.
    """

    name: str = 'CosineAnnealing'
    warmup_steps: int = 500
    constant_steps: int = 50000
    min_lr: float = 2e-5


@dataclass
class OptimizationConfig(BaseConfig):
    """
    Configuration for the optimizer used in the training process.

    Attributes:
        name (str): The name of the optimizer. Example: 'fused_adam'.
        lr (float): The learning rate for the optimizer.
        weight_decay (float): Weight decay factor for regularization in the optimizer.
        betas (List[float]): Coefficients used for computing running averages of gradient
            and its square. Example: [0.9, 0.98].
        sched (SchedulerConfig): Configuration for the learning rate scheduler.
    """

    name: str = 'fused_adam'
    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    sched: SchedulerConfig = field(default_factory=lambda: SchedulerConfig())
