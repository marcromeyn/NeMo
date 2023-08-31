# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Union

from omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.modules.common.megatron.adapters.parallel_adapters import PromptEncoderAdapterConfig
from nemo.collections.nlp.parts.peft_config import PEFTConfig
from nemo.core.classes.mixins.adapter_mixins import (
    AdapterModelPTMixin,
    AdapterModuleMixin,
    _prepare_default_adapter_config,
)
from nemo.utils import logging, model_utils


class NLPAdapterModelMixin(AdapterModelPTMixin):
    """ NLP Adapter Mixin that can augment any Encoder module with Adapter module support.
    # Todo rewrite doc string
    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.

    An Adapter module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Adapter module is zero-initialized, so that the residual connection to the adapter
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `adapter_layer`: A torch.nn.ModuleDict(), whose keys are the names of the adapter (globally unique),
                and values are the Adapter nn.Module().
        -   `adapter_cfg`: A OmegaConf DictConfig object that holds the config of the adapters that are initialized.
        -   `adapter_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.adapters.global_cfg.*`.

    **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Adapter config information to `self.cfg.adapters`.
    """

    def __init__(self, *args, **kwargs):
        self.use_peft = False
        self.setup_complete = False
        super().__init__(*args, **kwargs)
        if hasattr(self, "enc_dec_model"):
            self.model_prefix = "enc_dec_model."  # for T5
        else:
            self.model_prefix = "model.module." if self.cfg.megatron_amp_O2 else "model."

    def _get_all_keys(self,):
        """
        Returns all the keys in the model
        """
        k = [n for n, p in self.named_parameters()]
        return set(k)

    def add_adapter(self, peft_cfgs: Union[PEFTConfig, List[PEFTConfig]]):
        """
        High level API to add one or more adapter modules to the model, and freeze the base weights

        Args:
            peft_cfgs: One or more PEFTConfig objects that specify the PEFT method configuration
        """

        def _check_and_add_adapter(module, peft_name, peft_cfg):
            if isinstance(module, AdapterModuleMixin):
                if model_utils.import_class_by_path(peft_cfg._target_) in module.get_accepted_adapter_types():
                    module.add_adapter(name=peft_name, cfg=peft_cfg)

        if not isinstance(peft_cfgs, List):
            peft_cfgs = [peft_cfgs]

        self.base_keys = self._get_all_keys()
        self.freeze()
        logging.info(f"Before adding PEFT params:\n{self.summarize()}")

        for cfg in peft_cfgs:
            layer_selection = cfg.layer_selection
            for peft_name, peft_cfg in cfg.get_config_dict().items():
                # hasattr(self, "model") means is GPT and not T5
                if hasattr(self, "model") and not isinstance(peft_cfg, PromptEncoderAdapterConfig):
                    if layer_selection is not None:
                        logging.info(
                            f"Layer selection {layer_selection} is enabled for the current model ("
                            f"{self.__class__.__name__} + {peft_name})"
                        )
                    for layer in self.model.language_model.encoder.layers:
                        if layer.layer_number in layer_selection:
                            for _, module in layer.named_modules():
                                _check_and_add_adapter(module, peft_name, peft_cfg)
                else:
                    # Non GPT models, as well as GPT+PTuning do not support layer selection
                    if layer_selection is not None:
                        logging.warning(
                            "Layer selection is specified, but it is not supported for either "
                            f"{self.__class__.__name__} or {peft_name})"
                        )
                    for _, module in self.named_modules():
                        _check_and_add_adapter(module, peft_name, peft_cfg)

                # Update the model.cfg with information about the new adapter from cfg
                module_name, adapter_name = self.resolve_adapter_module_name_(peft_name)
                with open_dict(self.cfg):
                    # Construct the minimum config required to be updated by adapter implementations
                    if 'adapters' not in self.cfg:
                        self.cfg.adapters = OmegaConf.create({})

                    self.cfg.adapters = _prepare_default_adapter_config(
                        global_key=self.adapter_global_cfg_key,
                        meta_key=self.adapter_metadata_cfg_key,
                        cfg=self.cfg.adapters,
                    )

                    # Inject the module name in the adapter metadata cfg
                    gcfg = self.adapter_global_cfg_key
                    mcfg = self.adapter_metadata_cfg_key
                    self.cfg.adapters[gcfg][mcfg]['modules'][adapter_name] = module_name

                    self.cfg.adapters[adapter_name] = OmegaConf.create(peft_cfg)

        logging.info(f"After adding PEFT params:\n{self.summarize()}")
        self.adapter_keys = self._get_all_keys() - self.base_keys
        self.use_peft = True

    def get_peft_state_dict(self):
        """
        Gets the keys associated with the adapters only.
        """
        state_dict = self.model.state_dict(prefix=self.model_prefix)
        peft_state_dict = {}
        for k in self.adapter_keys:
            # state_dict keys needs to be in non-O2 format and will be corrected in PEFTSaveRestoreConnector if O2=True
            new_k = k.replace("model.module.", "model.", 1)
            peft_state_dict[new_k] = state_dict[k]
        return peft_state_dict


    def state_dict(self, destination=None, prefix=None, keep_vars=False):
        if self.use_peft and self.setup_complete:
            # Once setup is complete we no longer need to track the frozen part of the model. Only there adapter state dict keeps changing so state_dict only track these.
            return self.get_peft_state_dict()
        else:
            # we want all the params with the same keys as calling self.state_dict()
            # but we can't call self.state_dict() here as it would be a recursive call.
            # so we call self.model.state_dict(prefix="model.") which will return all the keys and params same as calling self.state_dict()
            return self.model.state_dict(prefix=self.model_prefix)

    def load_state_dict(self, state_dict, strict: bool = True):
        if self.use_peft and self.setup_complete:
            # at this stage only adapter params will appear in the state_dict arg
            # so we only update those while the rest of the model is frozen.
            # setting strict=False will ignore the missing keys (which are not being updated anyway)
            # explicitly check if state_dict.keys matches all the expected self.adapter_keys since we don't have the
            # safety in strict=True anymore.
            assert set(state_dict.keys()) == self.adapter_keys
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict, strict=True)
