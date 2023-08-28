from typing import List, Optional
from dataclasses import dataclass, asdict

from omegaconf import OmegaConf, DictConfig
from nemo.collections.nlp.models.language_modeling.config.base import (
    ActivationCheckpointingConfig,
    AMPConfig,
    BaseConfig,
    FusionConfig,
    NSysProfilingConfig,
    OptimizationConfig,
    TransformerEngineConfig
)


@dataclass
class ParallelismConfig(BaseConfig):
    """Configuration for parallelism including micro and global batch sizes, and model parallelism.

    Attributes:
        micro_batch_size (int): Micro batch size, limited by GPU memory.
        global_batch_size (int): Global batch size, uses more micro batches to reach this size.
        rampup_batch_size (Optional[List[int]]): Specifies the batch size ramp-up. Should be a list of 3 values: [<start_batch_size>, <batch_size_increment>, <rampup_samples>]. Defaults to None.
        tensor_model_parallel_size (int): Intra-layer model parallelism. Defaults to 1.
        pipeline_model_parallel_size (int): Inter-layer model parallelism. Defaults to 1.
        virtual_pipeline_model_parallel_size (Optional[int]): Interleaved pipeline. Defaults to None.
        sequence_parallel (bool): 
            Makes tensor parallelism more memory efficient for LLMs (20B+) by parallelizing layer 
            norms and dropout sequentially
            See Reducing Activation Recomputation in Large Transformer Models: 
            https://arxiv.org/abs/2205.05198 for more details.
    """

    micro_batch_size: int = 4
    global_batch_size: int = 8
    rampup_batch_size: Optional[List[int]] = None
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    sequence_parallel: bool = False


@dataclass
class GPTTokenizerConfig(BaseConfig):
    """Configuration for the Tokenizer used in the GPT.

    Attributes:
        library (str): The library used for tokenization. Determines the underlying implementation.
            Default is 'megatron'.
        type (str): Type of tokenizer, which defines the algorithm to be used for tokenizing text.
            Example: 'GPT2BPETokenizer'. Default is 'GPT2BPETokenizer'.
        model (Optional[str]): Path to a pre-trained model file for the tokenizer, if any.
            Default is None.
        vocab_file (Optional[str]): Path to a file containing vocabulary, used for mapping tokens to
            integers. Default is None.
        merge_file (Optional[str]): Path to a file containing merge rules for Byte-Pair Encoding (BPE)
            algorithms. Default is None.
        delimiter (Optional[str]): Delimiter used for special tokenization needs, such as tabular data.
            Default is None. Note: Only used for tabular tokenizer.
        sentencepiece_legacy (bool): If set to True, enables adding special tokens to sentencepiece
            tokenizers. This is useful for compatibility with older models that expected this behavior.
            Default is False.
    """

    library: str = 'megatron'
    type: str = 'GPT2BPETokenizer'
    model: Optional[str] = None
    vocab_file: Optional[str] = None
    merge_file: Optional[str] = None
    delimiter: Optional[str] = None  # only used for tabular tokenizer
    sentencepiece_legacy: bool = False  # Legacy=True allows you to add special tokens to sentencepiece tokenizers.


@dataclass
class GPTMiscConfig:
    """Miscellaneous configuration settings for GPT.

    Attributes:
        seed (Optional[int]): Seed for random number generation. Default is 1234.
        resume_from_checkpoint (Optional[str]): Manually set the checkpoint file to load from. Default is None.
        use_cpu_initialization (bool): Init weights on the CPU (slow for large models). Default is False.
        onnx_safe (bool): Use work-arounds for known problems with Torch ONNX exporter. Default is False.
        apex_transformer_log_level (int): Python logging level. Default is 30.
        gradient_as_bucket_view (bool): Allocate gradients in a contiguous bucket to save memory. Default is True.
        sync_batch_comm (bool): Enable stream synchronization after each p2p communication between pipeline stages. Default is False.
    """
    
    seed: Optional[int] = 1234
    resume_from_checkpoint: Optional[str] = None
    use_cpu_initialization: bool = False
    onnx_safe: bool = False
    apex_transformer_log_level: int = 30
    gradient_as_bucket_view: bool = True
    sync_batch_comm: bool = False
    gc_interval: int = 0



_V1_FLATTENED_CONFIGS = {
    "amp": AMPConfig, 
    "activation_checkpoint": ActivationCheckpointingConfig, 
    "fusion": FusionConfig, 
    "misc": GPTMiscConfig, 
    "te": TransformerEngineConfig,
    "parallel": ParallelismConfig
}
_V1_ATTRIBUTES = {
    key: name
    for name, class_name in _V1_FLATTENED_CONFIGS.items()
    for key in class_name.__dataclass_fields__.keys()
}


@dataclass
class GPTConfig(BaseConfig):
    """Configuration for the GPT model.

    Attributes:
        encoder_seq_length (int): Sequence length for the encoder. This defines the maximum number
            of tokens that can be processed by the encoder. Default is 512.
        max_position_embeddings (Union[int, str]): Maximum number of positional embeddings. This
            can be set to a specific number or a string expression, such as "${.encoder_seq_length}".
            Default value is derived from the encoder_seq_length.
        num_layers (int): Number of layers in the transformer architecture. Affects model complexity.
            Default is 12.
        hidden_size (int): Dimensionality of the hidden layers in the transformer. Default is 768.
        ffn_hidden_size (int): Hidden size for the Feed Forward Network, usually 4 times the hidden_size.
            Default is 3072.
        num_attention_heads (int): Number of attention heads in the multi-head attention mechanism.
            Default is 12.
        init_method_std (float): Standard deviation of the zero-mean normal distribution used for weight
            initialization. Default is 0.02.
        use_scaled_init_method (bool): If True, uses scaled residuals initialization method. Default is True.
        hidden_dropout (float): Dropout probability applied to hidden state within the transformer.
            Helps in regularization. Default is 0.1.
        attention_dropout (float): Dropout probability applied to attention mechanism. Default is 0.1.
        ffn_dropout (float): Dropout probability applied in the feed-forward layer. Default is 0.0.
        kv_channels (Optional[int]): Dimension of projection weights in multi-head attention. Set to
            hidden_size // num_attention_heads if None. Default is None.
        apply_query_key_layer_scaling (bool): If True, scales Q * K^T by 1 / layer-number. Default is False.
        normalization (str): Type of normalization layer to use. Options are 'layernorm', 'rmsnorm'.
            Default is 'layernorm'.
        layernorm_epsilon (float): Small value added to denominator in layer normalization. Default is 1e-5.
        do_layer_norm_weight_decay (bool): If True, applies weight decay on all layer normalization parameters.
            Default is False.
        make_vocab_size_divisible_by (int): Pads the vocabulary size to be divisible by this value for
            computational efficiency. Default is 128.
        pre_process (bool): If True, includes an embedding layer at the start. Default is True.
        post_process (bool): If True, includes a pooler layer at the end. Default is True.
        persist_layer_norm (bool): If True, uses persistent fused layer norm kernel. Default is True.
        bias (bool): If True, includes bias terms in all weight matrices. Default is True.
        activation (str): Activation function type. Options include various variations of GELU.
            Default is 'gelu'.
        headscale (bool): If True, learns extra parameters that scale the output of each self-attention
            head. Default is False.
        transformer_block_type (str): Type of transformer block. Options include 'pre_ln', 'post_ln',
            'normformer'. Default is 'pre_ln'.
        openai_gelu (bool): If True, uses OpenAI's version of GELU instead of the default. Default is False.
        normalize_attention_scores (bool): If True, scales Q * K^T by 1 / sqrt(hidden_size_per_head).
            Generally set to True for compatibility. Default is True.
        position_embedding_type (str): Type of position embedding. Options include 'learned_absolute',
            'rope', 'alibi', among others. Default is 'learned_absolute'.
        rotary_percentage (float): Multiplier for per head dimension if using rope position embeddings.
            Default is 1.0.
        attention_type (str): Attention mechanism type. Default is 'multihead'.
        share_embeddings_and_output_weights (bool): If True, shares weights between embedding and output layers.
            Default is True.
        overlap_p2p_comm (bool): If True, overlaps peer-to-peer communication with computations. Only valid when
            `virtual_pipeline_model_parallel_size` is larger than 1. Default is False.
        batch_p2p_comm (bool): If True, batches consecutive inter-peer send/receive operations. Only valid when
            `virtual_pipeline_model_parallel_size` is larger than 1. Default is True.
    """
    
    # data: GPTPretrainDatasetConfig
    tokenizer: GPTTokenizerConfig = GPTTokenizerConfig()
    nsys_profile: NSysProfilingConfig = NSysProfilingConfig()
    optim: OptimizationConfig = OptimizationConfig()
    
    # For V1 these configs can also be accessed in a "flat" way
    amp: AMPConfig = AMPConfig()
    activation_checkpoint: ActivationCheckpointingConfig = ActivationCheckpointingConfig()
    fusion: FusionConfig = FusionConfig()
    misc: GPTMiscConfig = GPTMiscConfig()
    te: TransformerEngineConfig = TransformerEngineConfig()
    parallel: ParallelismConfig = ParallelismConfig()
    
    encoder_seq_length: int = 512
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_attention_heads: int = 12
    init_method_std: float = 0.02
    use_scaled_init_method: bool = True
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.0
    kv_channels: Optional[int] = None
    apply_query_key_layer_scaling: bool = False
    normalization: str = 'layernorm'
    layernorm_epsilon: float = 1e-5
    do_layer_norm_weight_decay: bool = False
    make_vocab_size_divisible_by: int = 128
    pre_process: bool = True
    post_process: bool = True
    persist_layer_norm: bool = True
    bias: bool = True
    activation: str = 'gelu'
    headscale: bool = False
    transformer_block_type: str = 'pre_ln'
    openai_gelu: bool = False
    normalize_attention_scores: bool = True
    position_embedding_type: str = 'learned_absolute'
    rotary_percentage: float = 1.0
    attention_type: str = 'multihead'
    share_embeddings_and_output_weights: bool = True
    overlap_p2p_comm: bool = False
    batch_p2p_comm: bool = True
    
    mcore_gpt: bool = False
    seq_len_interpolation_factor: Optional[float] = None
    num_query_groups: Optional[int] = None
    use_flash_attention: bool = False
    
    # From sft
    restore_from_path: Optional[str] = None
    save_nemo_on_validation_end: bool = False
    answer_only_loss: bool = False
    fp32_grad_accum: bool = False
    contiguous_grad_bucket: bool = False
    async_grad_allreduce: bool = False
    merges_file: Optional[str] = None
    vocab_file: Optional[str] = None
    target: Optional[str] = None
    precision: Optional[str] = None
    
    @classmethod
    def from_flattened_cfg(cls, cfg: OmegaConf) -> "GPTConfig":
        """
        Constructs the dataclass from a flattened OmegaConf object.

        Args:
            cfg (OmegaConf): The flattened configuration object.

        Returns:
            GPTConfig: The constructed dataclass.
        """
        # Create a copy of the cfg to avoid modifying the original
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)

        # Handle special v1 flattened attributes
        v1_config_objects = {
            name: class_name() for name, class_name in _V1_FLATTENED_CONFIGS.items()
        }
        _to_remove = ["data"]
        for key, value in cfg_dict.items():
            attr_class = _V1_ATTRIBUTES.get(key)
            if attr_class:
                v1_config_objects[attr_class].__setattr__(key, value)
                _to_remove.append(key)
                
        for key in _to_remove:
            cfg_dict.pop(key, None)

        # Extract other attributes, providing defaults if not found
        # data = GPTPretrainDatasetConfig(**cfg_dict.pop("data", {}))
        tokenizer = GPTTokenizerConfig(**cfg_dict.pop("tokenizer", {}))
        nsys_profile = NSysProfilingConfig(**cfg_dict.pop("nsys_profile", {}))
        optim = OptimizationConfig(**cfg_dict.pop("optim", {}))
        
        cfg_dict.pop("max_position_embeddings", None)

        # Un-pack the remaining attributes to instantiate the GPTConfig class
        return cls(
            # data=data, 
            tokenizer=tokenizer, 
            nsys_profile=nsys_profile, 
            optim=optim,
            **v1_config_objects,
            **cfg_dict
        )
        
    # def __post_init__(self):
    #     self.data.seq_length = str(self.encoder_seq_length)
    
    @property
    def max_position_embeddings(self) -> int:
        return self.encoder_seq_length
    
    def to_cfg(self, precision=None) -> DictConfig:
        output_dict = asdict(self)
        output_dict["max_position_embeddings"] = self.max_position_embeddings
        
        for key in _V1_FLATTENED_CONFIGS:
            del output_dict[key]
            for nested_key, val in asdict(getattr(self, key)).items():
                output_dict[nested_key] = val
                
        if precision:
            output_dict["precision"] = precision
            
        return OmegaConf.create(output_dict)
    
    def __getattr__(self, name: str):
        """
        Allows access to the attributes that were flattened in v1 in a flattened way.

        Args:
            name (str): The name of the attribute being accessed.

        Returns:
            Any: The value of the attribute.
        """
        # Check if the attribute name is in the flattened names set
        if name in _V1_ATTRIBUTES:
            attr_name = _V1_ATTRIBUTES[name]
            return getattr(getattr(self, attr_name), name)

        # Check if the attribute is part of the dataclass attributes
        if name in self.__dict__:
            return getattr(self, name)

        # If attribute not found, raise an AttributeError
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
