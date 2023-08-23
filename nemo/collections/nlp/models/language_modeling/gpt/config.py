from typing import Any, Optional, Union, List, Dict, Tuple
from dataclasses import dataclass, is_dataclass, field

from omegaconf import OmegaConf
from nemo.collections.nlp.models.language_modeling.config.base import (
    ActivationCheckpointingConfig,
    AMPConfig,
    FusionConfig,
    NSysProfilingConfig,
    OptimizationConfig,
    TransformerEngineConfig
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)


@dataclass
class GPTPretrainDatasetConfig:
    """
    Configuration for the GPT pretraining dataset.

    Attributes:
        data_prefix (Union[List, str, dict]): Path to the data. Can be specified in various formats
            like List, String, or Dictionary. For examples, please refer to the comments in the class.
        index_mapping_dir (Optional[str]): Path to save index mapping .npy files. Default is the same location as data_prefix.
        data_impl (str): Implementation type of the data. Default is "mmap".
        splits_string (str): Split ratios for train, validation, and test data. Default is "900, 50, 50".
        seq_length (str): Sequence length of the model's encoder. Default is "${model.encoder_seq_length}".
        skip_warmup (bool): Whether to skip warmup. Default is True.
        num_workers (int): Number of worker threads for data loading. Default is 2.
        dataloader_type (str): Type of data loader. Options are "single" or "cyclic". Default is "single".
        reset_position_ids (bool): Whether to reset position ids after end-of-document token. Default is False.
        reset_attention_mask (bool): Whether to reset attention mask after end-of-document token. Default is False.
        eod_mask_loss (bool): Whether to mask loss for the end-of-document tokens. Default is False.
        validation_drop_last (bool): Whether to consume last partial validation samples. Set to False if you want to consume them. Default is True.
        no_seqlen_plus_one_input_tokens (bool): Set to True to disable fetching (sequence length + 1) input tokens. Default is False.
        pad_samples_to_global_batch_size (bool): Set to True to pad the last partial batch with -1's to equal global batch size. Default is False.
        shuffle_documents (bool): Set to False to disable documents shuffling. Default is True.
        exchange_indices_distributed (bool): Set to True to exchange indices via torch.distributed instead of filesystem. Default is False.
    """

    data_prefix: str
    index_mapping_dir: Optional[str] = None
    data_impl: str = "mmap"
    splits_string: str = "900, 50, 50"
    seq_length: str = "${model.encoder_seq_length}"
    skip_warmup: bool = True
    num_workers: int = 2
    dataloader_type: str = "single"
    reset_position_ids: bool = False
    reset_attention_mask: bool = False
    eod_mask_loss: bool = False
    validation_drop_last: bool = True
    no_seqlen_plus_one_input_tokens: bool = False
    pad_samples_to_global_batch_size: bool = False
    shuffle_documents: bool = True
    exchange_indices_distributed: bool = False


@dataclass
class GPTTokenizerConfig:
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



_V1_FLATTENED_CONFIGS = [
    AMPConfig, 
    ActivationCheckpointingConfig, 
    FusionConfig, 
    GPTMiscConfig, 
    TransformerEngineConfig
]
_V1_ATTRIBUTES = {
    key: class_name.__name__.lower() 
    for class_name in _V1_FLATTENED_CONFIGS 
    for key in class_name.__dataclass_fields__.keys()
}


@dataclass
class GPTConfig:
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
    
    data: GPTPretrainDatasetConfig
    tokenizer: GPTTokenizerConfig = GPTTokenizerConfig()
    nsys_profile: NSysProfilingConfig = NSysProfilingConfig()
    optim: OptimizationConfig = OptimizationConfig()
    
    # For V1 these configs can also be accessed in a "flat" way
    amp: AMPConfig = AMPConfig()
    activation: ActivationCheckpointingConfig = ActivationCheckpointingConfig()
    fusion: FusionConfig = FusionConfig()
    misc: GPTMiscConfig = GPTMiscConfig()
    te: TransformerEngineConfig = TransformerEngineConfig()
    
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
        cfg_copy = OmegaConf.copy(cfg)

        # Handle special v1 flattened attributes
        v1_config_objects = {
            class_name.__name__.lower(): class_name() for class_name in _V1_FLATTENED_CONFIGS
        }
        for key, value in cfg_copy.items():
            attr_class = _V1_ATTRIBUTES.get(key)
            if attr_class:
                v1_config_objects[attr_class].__setattr__(key, value)
                del cfg_copy[key]

        # Extract other attributes, providing defaults if not found
        data = cfg_copy.pop("data", GPTPretrainDatasetConfig())
        tokenizer = cfg_copy.pop("tokenizer", GPTTokenizerConfig())
        nsys_profile = cfg_copy.pop("nsys_profile", NSysProfilingConfig())
        optim = cfg_copy.pop("optim", OptimizationConfig())

        # Un-pack the remaining attributes to instantiate the GPTConfig class
        return cls(
            data=data, 
            tokenizer=tokenizer, 
            nsys_profile=nsys_profile, 
            optim=optim,
            **v1_config_objects, 
            **cfg_copy
        )
        
    def __post_init__(self):
        self.data.seq_length = str(self.encoder_seq_length)
        
    def get(self, attribute_name: str, default=None):
        """Retrieves the value of the specified attribute or returns the default value if not found."""
        return getattr(self, attribute_name, default)
    
    @property
    def max_position_embeddings(self) -> int:
        return self.encoder_seq_length
    
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

        # If attribute not found, call the base class's __getattr__ method
        return super().__getattr__(name)
