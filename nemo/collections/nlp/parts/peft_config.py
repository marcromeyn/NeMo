# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass

from omegaconf import OmegaConf, open_dict
from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import (
    AdapterName,
    LoraKQVAdapterConfig,
    MLPInfusedAdapterConfig,
    InfusedAdapterConfig,
    PromptEncoderAdapterConfig,
    ParallelLinearAdapterConfig,
)
from nemo.utils import logging, model_utils
from nemo.core.classes.mixins.adapter_mixins import (
    AdapterConfig,
    AdapterModuleMixin,
    _prepare_default_adapter_config,
)

if TYPE_CHECKING:
    from nemo.collections.nlp.models.nlp_model import NLPModel
    

class PEFTConfig:
    _registry = {}  # Class registry to keep track of subclasses
    _cfgs = {}  # Class registry to keep track of subclasses
    
    def __init__(self, config):
        self.config = config
    
    def apply(self, model: NLPModel):
        configs = self.build(model)
        if not getattr(self.config, "layer_selection", None):
            layer_selection = list(range(1, model.cfg.num_layers + 1))
        else:
            layer_selection = self.config.layer_selection
        for peft_name, peft_cfg in configs.items():
            # hasattr(self, "model") means is GPT and not T5
            if hasattr(model, "model") and not isinstance(peft_cfg, PromptEncoderAdapterConfig):
                if layer_selection is not None:
                    logging.info(
                        f"Layer selection {layer_selection} is enabled for the current model ("
                        f"{model.__class__.__name__} + {peft_name})"
                    )
                for layer in model.model.language_model.encoder.layers:
                    if layer.layer_number in layer_selection:
                        for _, module in layer.named_modules():
                            _check_and_add_adapter(module, peft_name, peft_cfg)
            else:
                # Non GPT models, as well as GPT+PTuning do not support layer selection
                if layer_selection is not None:
                    logging.warning(
                        "Layer selection is specified, but it is not supported for either "
                        f"{model.__class__.__name__} or {peft_name})"
                    )
                for _, module in model.named_modules():
                    _check_and_add_adapter(module, peft_name, peft_cfg)

            # Update the model.cfg with information about the new adapter from cfg
            module_name, adapter_name = model.resolve_adapter_module_name_(peft_name)
            with open_dict(model.cfg):
                # Construct the minimum config required to be updated by adapter implementations
                if 'adapters' not in model.cfg:
                    model.cfg.adapters = OmegaConf.create({})

                model.cfg.adapters = _prepare_default_adapter_config(
                    global_key=model.adapter_global_cfg_key,
                    meta_key=model.adapter_metadata_cfg_key,
                    cfg=model.cfg.adapters,
                )

                # Inject the module name in the adapter metadata cfg
                gcfg = model.adapter_global_cfg_key
                mcfg = model.adapter_metadata_cfg_key
                model.cfg.adapters[gcfg][mcfg]['modules'][adapter_name] = module_name

                model.cfg.adapters[adapter_name] = OmegaConf.create(peft_cfg)
    
    @classmethod
    def register(cls, key: str, config):
        """
        Register a subclass in the registry
        """
        def decorator(subclass):
            cls._registry[key] = subclass
            cls._cfgs[key] = config
            return subclass
        return decorator

    @classmethod
    def from_cfg(cls, cfg):
        """
        Create a new instance of a registered subclass
        """
        name = cfg.peft_scheme
        if name not in cls._registry:
            raise ValueError(f"No subclass registered under name {name}")
        if name not in cls._cfgs:
            raise ValueError(f"No subclass registered under name {name}")
        
        config = cls._cfgs[name](**cfg[name])
        return cls._registry[name](config)
    
    
def _check_and_add_adapter(module, peft_name, peft_cfg):
    if isinstance(module, AdapterModuleMixin):
        if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
            module.add_adapter(name=peft_name, cfg=peft_cfg)
        
        
@dataclass
class LoraPEFTConfig:
    adapter_dim: int
    adapter_dropout: float
    layer_selection: Optional[List[int]] = None  # None will apply adapters to all layers
    weight_tying: bool = False
    position_embedding_strategy: Optional[str] = None  # Used only when weight_tying is True
    
    column_init_method: str = "normal"  # Options: 'xavier', 'zero', 'normal'
    row_init_method: str = "zero"  # Options: 'xavier', 'zero', 'normal'


@PEFTConfig.register("lora", LoraPEFTConfig)
class Lora(PEFTConfig):
    def build(self, model: NLPModel) -> Dict[str, AdapterConfig]:
        cfg = model.cfg
        if cfg.get("kv_channels", None) is None:
            assert (
                cfg.hidden_size % cfg.num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = cfg.hidden_size // cfg.num_attention_heads
        else:
            kv_channels = cfg.kv_channels
        projection_size = kv_channels * cfg.num_attention_heads

        adapter_cfg = LoraKQVAdapterConfig(
            in_features=cfg.hidden_size,
            out_features=3 * projection_size,
            dim=self.config.adapter_dim,
            norm_position=None,
            norm_type=None,
            activation="identity",
            column_init_method=self.config.column_init_method,
            row_init_method=self.config.row_init_method,
            gather_output=False,
            dropout=self.config.adapter_dropout,
        )
        
        return {
            AdapterName.LORA_KQV_ADAPTER: adapter_cfg,
        }
        
        
@dataclass
class IA3PEFTConfig:
    in_features: int
    layer_selection: Optional[List[int]] = None  # None will apply adapters to all layers


@PEFTConfig.register("ia3", IA3PEFTConfig)
class IA3(PEFTConfig):
    def build(self, model: NLPModel) -> Dict[str, AdapterConfig]:
        mlp_infused_adapter_cfg = MLPInfusedAdapterConfig(
            in_features=self.config.in_features,
        )
        infused_adapter_cfg = InfusedAdapterConfig(
            in_features=self.config.in_features
        )
        
        return {
            AdapterName.KEY_INFUSED: infused_adapter_cfg,
            AdapterName.VALUE_INFUSED: infused_adapter_cfg,
            AdapterName.MLP_INFUSED: mlp_infused_adapter_cfg,
        }
        
        
@dataclass
class PtuningPEFTConfig:
    virtual_tokens: int
    bottleneck_dim: int
    embedding_dim: int
    init_std: float
    hidden_size: int
    layer_selection: Optional[List[int]] = None  # None will apply adapters to all layers


@PEFTConfig.register("ptuning", PtuningPEFTConfig)
class Ptuning(PEFTConfig):
    def build(self, model: NLPModel) -> Dict[str, AdapterConfig]:
        adapter_cfg = PromptEncoderAdapterConfig(
            self.config.virtual_tokens,
            self.config.bottleneck_dim,
            self.config.embedding_dim,
            self.config.init_std,
            self.config.hidden_size
        )
        
        return {AdapterName.PTUNING_ADAPTER: adapter_cfg}
    
    
@dataclass
class AdapterPEFTConfig:
    adapter_dim: int
    adapter_dropout: float
    norm_position: str = "pre"
    norm_type: str = "mixedfusedlayernorm"
    column_init_method: str = "xavier"
    row_init_method: str = "zero"
    layer_selection: Optional[List[int]] = None  # None will apply adapters to all layers
    
    # Seem un-used
    type: str = 'parallel_adapter' # this should be either 'parallel_adapter' or 'linear_adapter'
    weight_tying: bool = False
    position_embedding_strategy: Optional[str] = None # used only when weight_tying is True


@PEFTConfig.register("adapter", AdapterPEFTConfig)
class Adapter(PEFTConfig):
    def build(self, model: NLPModel) -> Dict[str, AdapterConfig]:
        adapter_cfg = ParallelLinearAdapterConfig(
            in_features=model.cfg.hidden_size,
            out_features=model.cfg.hidden_size,
            dim=self.config.adapter_dim,
            norm_position=self.config.norm_position,
            norm_type=self.config.norm_type,
            column_init_method=self.config.column_init_method,
            row_init_method=self.config.row_init_method,
            dropout=self.config.adapter_dropout
        )
        
        return {
            AdapterName.PRE_ATTN_ADAPTER: adapter_cfg,
            AdapterName.POST_ATTN_ADAPTER: adapter_cfg
        }
