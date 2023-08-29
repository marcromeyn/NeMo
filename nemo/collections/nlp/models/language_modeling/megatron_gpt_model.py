# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import itertools
import inspect
import queue
import warnings
from dataclasses import fields
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Union, TYPE_CHECKING

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.gpt_dataset import build_train_valid_test_datasets
from nemo.collections.nlp.models.language_modeling.megatron.gpt_model import GPTModel
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    get_all_params_for_weight_decay_optimization,
    get_ltor_masks_and_position_ids,
    get_params_for_weight_decay_optimization,
)
from nemo.collections.nlp.modules.common.text_generation_utils import (
    generate,
    get_computeprob_response,
    get_default_length_params,
    get_default_sampling_params,
    megatron_gpt_generate,
)
from nemo.collections.nlp.modules.common.transformer.text_generation import (
    LengthParam,
    OutputType,
    SamplingParam,
    TextGeneration,
)
from nemo.collections.nlp.parts.utils_funcs import activation_to_func, get_last_rank
from nemo.core.classes import Exportable
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils import logging

try:
    import apex.transformer.pipeline_parallel.utils
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import init_method_normal, scaled_init_method_normal

    # TODO @tmoon: Use once available in Megatron-LM
    # from megatron.core.pipeline_parallel.schedules import DataIteratorList

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    TransformerConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    import transformer_engine
    from transformer_engine.pytorch import module as te_module

    HAVE_TE = True

except (ImportError, ModuleNotFoundError):
    HAVE_TE = False
    
if TYPE_CHECKING:
    from nemo.collections.nlp.models.language_modeling.gpt.config import GPTConfig
    

class MegatronGPTExportableModel(torch.nn.Module, Exportable):
    """
    Megatron GPT Wrapper for ONNX export
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.fp8_enabled = model.cfg.get('fp8', False)
        self.fp8_recipe = None
        if self.fp8_enabled and HAVE_TE:
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=0, interval=1, fp8_format=transformer_engine.common.recipe.Format.E4M3
            )

        self.dtype = None
        if model.cfg['precision'] == 'bf16':
            self.dtype = torch.bfloat16
        elif int(model.cfg['precision']) == 32:
            self.dtype = torch.float
        elif int(model.cfg['precision']) == 16:
            self.dtype = torch.float16
        else:
            raise ValueError(f"precision: {model.cfg['precision']} is not supported.")

    def forward(self, tokens, position_ids, attention_mask):
        if self.fp8_enabled and HAVE_TE:
            with transformer_engine.pytorch.onnx_export(self.fp8_enabled), transformer_engine.pytorch.fp8_autocast(
                enabled=self.fp8_enabled, fp8_recipe=self.fp8_recipe
            ), torch.no_grad(), torch.inference_mode(), torch.autocast(
                'cuda', dtype=self.dtype
            ), warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning, module=r'.*')
                assert tokens.shape == position_ids.shape
                assert attention_mask.shape[2] == attention_mask.shape[3] == tokens.shape[1] == position_ids.shape[1]
                output_tensor = self.model.forward(
                    tokens=tokens.cuda(),
                    text_position_ids=position_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    labels=None,
                )
        else:
            with torch.no_grad(), torch.inference_mode(), torch.autocast(
                'cuda', dtype=self.dtype
            ), warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning, module=r'.*')
                assert tokens.shape == position_ids.shape
                assert attention_mask.shape[2] == attention_mask.shape[3] == tokens.shape[1] == position_ids.shape[1]
                output_tensor = self.model.forward(
                    tokens=tokens.cuda(),
                    text_position_ids=position_ids.cuda(),
                    attention_mask=attention_mask.cuda(),
                    labels=None,
                )

        return output_tensor

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def input_example(self, max_batch=1, max_dim=768, seq_len=6):
        ids = [self.model.tokenizer.text_to_ids(text) for text in ["how is the weather on           Sunday"]]
        id_tensors = [torch.unsqueeze(torch.LongTensor(id_list), dim=0) for id_list in ids]
        masks_and_position_ids = [
            get_ltor_masks_and_position_ids(id_tensor, self.model.tokenizer.eos_id, False, False, False)
            for id_tensor in id_tensors
        ]
        for tokens, attn_mask_and_pos_ids in zip(id_tensors, masks_and_position_ids):
            attn_mask, _, pos_ids = attn_mask_and_pos_ids
            return tokens, pos_ids, attn_mask

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "position_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('D', 'D', 'T', 'T'), ChannelType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"logits": NeuralType(('B', 'T', 'D'), ChannelType())}

    @property
    def input_names(self) -> List[str]:
        return ['input_ids', 'position_ids', 'attention_mask']

    @property
    def output_names(self) -> List[str]:
        return ['logits']


class MegatronGPTModel(MegatronBaseModel, TextGeneration):
    """
    Megatron GPT pretraining
    """

    def __init__(self, cfg: GPTConfig, trainer: Trainer):
        if not HAVE_APEX:
            raise ImportError(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

        if not HAVE_MEGATRON_CORE:
            raise ImportError(
                "megatron-core was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        # this prevents base constructor from initializing tokenizer
        self.tokenizer = None
        
        from nemo.collections.nlp.models.language_modeling.gpt.config import GPTConfig
        
        if isinstance(cfg, DictConfig):
            cfg = GPTConfig.from_flattened_cfg(cfg)
        
        if isinstance(cfg, GPTConfig):
            self.config = cfg
            cfg = self.config.to_cfg(trainer.precision)
        else:
            raise ValueError(f"Please provide a DictConfig or GPTConfig, got: {cfg}")
        
        super().__init__(cfg, trainer=trainer, no_lm_init=True)

        self._validate_trainer()

        # build the transformer config
        self.transformer_config = self.build_transformer_config()

        self.megatron_amp_o2 = self.config.megatron_amp_O2

        self.mcore_gpt = self.config.mcore_gpt

        self.rampup_batch_size = self.config.rampup_batch_size
        if self.rampup_batch_size:
            self.prev_consumed_samples = 0
            self.if_first_step = 0
            self.prev_global_batch_size = None

        if not self.megatron_amp_o2 and self.config.virtual_pipeline_model_parallel_size:
            raise ValueError('Virtual pipeline model parallel is only supported when using megatron_amp_O2')

        # build_model returns a list of modules which are used for interleaved pipeline parallelism
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                on_cpu=True,
                virtual_pipeline_model_parallel_size=self.config.virtual_pipeline_model_parallel_size,
            )
        else:
            self.model = build_model(
                model_provider_func=self.model_provider_func,
                wrap_with_ddp=False,
                virtual_pipeline_model_parallel_size=self.config.virtual_pipeline_model_parallel_size,
            )

        # if we're not using interleaved, then self.model is a module.
        if self.config.virtual_pipeline_model_parallel_size is None:
            self.model = self.model[0]

        if self.megatron_amp_o2:

            if not self.with_distributed_adam:
                # Pre-allocate the model on GPU to have master parameters allocated on the same device with matching data type
                if isinstance(self.model, list):
                    for module in self.model:
                        module.cuda(torch.cuda.current_device())
                else:
                    self.model.cuda(torch.cuda.current_device())

            self._wrap_model_for_O2()

        if self.trainer.precision in ['bf16', 'bf16-mixed']:
            self.autocast_dtype = torch.bfloat16
        elif self.trainer.precision in [32, '32', '32-true']:
            self.autocast_dtype = torch.float
        elif self.trainer.precision in [16, '16', '16-mixed']:
            self.autocast_dtype = torch.half
        else:
            raise ValueError('precision must be in ["32-true", "16-mixed", "bf16-mixed"]')

        self.enable_autocast = (
            True if (not self.megatron_amp_o2) and (self.autocast_dtype in [torch.float16, torch.bfloat16]) else False
        )

        self.transformer_engine = self.config.transformer_engine

        # configuration used for inference
        self._inference_config = None

        # Convert the global-batch-based profile index to micro-batch index
        if hasattr(self, '_nsys_profile_enabled'):
            mp_size = self.config.tensor_model_parallel_size * self.config.pipeline_model_parallel_size
            data_parallel_world_size = trainer.world_size // mp_size
            grad_accum_steps = self.config.global_batch_size // (self.config.micro_batch_size * data_parallel_world_size)
            self._nsys_profile_start_step *= grad_accum_steps
            self._nsys_profile_end_step *= grad_accum_steps

        self.get_attention_mask_from_fusion = self.config.get_attention_mask_from_fusion
        self.initialize_ub = self.config.ub_tp_comm_overlap

    def get_gpt_module_list(self):
        if isinstance(self.model, list):
            return [
                model.module if isinstance(model, (Float16Module, MCoreFloat16Module)) else model
                for model in self.model
            ]
        elif isinstance(self.model, (Float16Module, MCoreFloat16Module)):
            return [self.model.module]
        else:
            return [self.model]

    def set_inference_config(self, inference_config):
        self._inference_config = inference_config

    def get_inference_config(self):
        return self._inference_config

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        if self.mcore_gpt:
            model = MCoreGPTModel(
                config=self.transformer_config,
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                max_sequence_length=self.config.encoder_seq_length,
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self.config.share_embeddings_and_output_weights,
                position_embedding_type=self.config.position_embedding_type,
                rotary_percent=self.config.rotary_percentage,
            )
        else:
            assert (
                self.cfg.get('num_query_groups', None) is None
            ), "Group Query Attention is only supported in Megatron Core. Set 'mcore_gpt' to use GQA."
            
            _params = dict(
                config=self.model_parallel_config,
                vocab_size=self.cfg.get('override_vocab_size', self.padded_vocab_size),
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                megatron_legacy=self.cfg.get('megatron_legacy', False),
                precision=self.cfg.precision
            )
            
            param_names = inspect.signature(GPTModel.__init__).parameters.keys()
            for key in param_names:
                if key not in _params and key != "self":
                    _params[key] = getattr(self.config, key)

            model = GPTModel(**_params)
        return model

    def setup_optimizer_param_groups(self):
        """ModelPT override. Optimizer will get self._optimizer_param_groups"""
        if self.cfg.get('do_layer_norm_weight_decay', False):
            if isinstance(self.model, list):
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization(self.model)
            else:
                self._optimizer_param_groups = get_all_params_for_weight_decay_optimization([self.model])

        else:
            self._optimizer_param_groups = get_params_for_weight_decay_optimization(self.model)

    def configure_optimizers(self):

        if self.with_distributed_adam:

            # Disable overlapped grad sync for embedding grad when
            # pipeline parallelism is enabled
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                modules = self.get_gpt_module_list()
                if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                    if len(modules) > 1:
                        module = modules[0]  # only the first virtual rank has the embeddings
                    else:
                        module = modules[0]
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        param = (
                            module.shared_embedding_or_output_weight()
                            if self.mcore_gpt
                            else module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_o2
                        param._disable_overlap_grad_sync = True
                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    if len(modules) > 1:
                        module = modules[-1]  # only the last virtual rank has the embeddings
                    else:
                        module = modules[0]
                    if self.cfg.get('share_embeddings_and_output_weights', True):
                        param = (
                            module.shared_embedding_or_output_weight()
                            if self.mcore_gpt
                            else module.word_embeddings_weight()
                        )
                        param._disable_greedy_grad_copy = not self.megatron_amp_o2
                        param._disable_overlap_grad_sync = True

            # Disable overlapped grad sync for layer norm grads when
            # sequence parallelism is enabled
            for param in self.parameters():
                if getattr(param, 'sequence_parallel', False):
                    param._disable_greedy_grad_copy = not self.megatron_amp_o2
                    param._disable_overlap_grad_sync = True

            # Initialize parameter buckets for overlapped grad and param syncs
            # Note: Params with disabled overlapping are put in the
            # last param bucket
            buckets = []
            if self.cfg.get('virtual_pipeline_model_parallel_size', None) is not None:
                # Initialize a bucket for each virtual pipeline stage
                for module in self.model:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    stage_bucket = []
                    layers = module.decoder.layers if self.mcore_gpt else module.language_model.encoder.layers
                    for layer in layers:
                        stage_bucket.extend(
                            p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)
                        )
                    buckets.append(stage_bucket)
            else:
                # Initialize a bucket for each Transformer layer
                modules = self.model if isinstance(self.model, list) else [self.model]
                for module in modules:
                    if isinstance(module, (Float16Module, MCoreFloat16Module)):
                        module = module.module
                    layers = module.decoder.layers if self.mcore_gpt else module.language_model.encoder.layers
                    for layer in layers:
                        buckets.append(
                            [p for p in layer.parameters() if not getattr(p, '_disable_overlap_grad_sync', False)]
                        )
            buckets.reverse()
            used_params = set()
            for bucket in buckets:
                used_params.update(bucket)
            buckets[-1].extend(p for p in self.parameters() if p not in used_params)
            self.distributed_adam_buckets = buckets

        return super().configure_optimizers()

    def forward(self, tokens, text_position_ids, attention_mask, labels):
        output_tensor = self.model(tokens, text_position_ids, attention_mask, labels=labels)
        return output_tensor

    def fwd_bwd_step(self, dataloader_iter, batch_idx, forward_only):

        # handle asynchronous grad reduction
        no_sync_func = None
        grad_sync_func = None
        param_sync_func = None
        if not forward_only and self.with_distributed_adam:
            no_sync_func = partial(self._optimizer.no_sync, greedy_grad_copy=self.megatron_amp_o2,)
            grad_sync_func = self.reduce_overlap_gradients
            param_sync_func = self.sync_overlap_parameters

        # pipeline schedules will get these from self.model.config
        for module in self.get_gpt_module_list():
            module.config.no_sync_func = no_sync_func
            module.config.grad_sync_func = grad_sync_func
            module.config.param_sync_func = param_sync_func

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        # TODO @akhattar: add num_micro_batches_with_partial_activation_checkpoints when ready
        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(forward_only),
            data_iterator=self._make_data_iterator_list(dataloader_iter),
            model=self.model,
            num_microbatches=get_num_microbatches(),
            forward_only=forward_only,
            seq_length=self.config.encoder_seq_length,
            micro_batch_size=self.config.micro_batch_size,
        )

        # only the last stages of the pipeline return losses
        if losses_reduced_per_micro_batch:
            if (not forward_only) or self.cfg.data.get('validation_drop_last', True):
                # average loss across micro batches
                loss_tensors_list = [loss_reduced['avg'] for loss_reduced in losses_reduced_per_micro_batch]
                loss_tensor = torch.concat(loss_tensors_list)
                loss_mean = loss_tensor.mean()
            else:
                # Get the total loss since micro batches sizes are not uniform
                loss_sum_tensors_list = [
                    loss_sum['loss_sum_and_ub_size']
                    for loss_sum in losses_reduced_per_micro_batch
                    if loss_sum['loss_sum_and_ub_size'][1] > 0
                ]
                loss_sum = (
                    torch.vstack(loss_sum_tensors_list).sum(axis=0)
                    if len(loss_sum_tensors_list) > 0
                    else torch.tensor([0.0, 0.0]).cuda()
                )
                return loss_sum
        else:
            # we're not on the last pipeline stage so no losses
            if forward_only:
                loss_mean = []
            else:
                loss_mean = torch.tensor(0.0).cuda()

        return loss_mean

    def initialize_ub_func(self):
        ub_cfgs = self.config.ub_tp_comm_overlap_cfg
        if ub_cfgs is None:
            warnings.warn(
                "Couldn't find TP config. Please check the path correctness. Initializing TP comm overlap with the default config."
            )

        input_shape = [
            self.config.encoder_seq_length * self.config.micro_batch_size,
            self.config.hidden_size,
        ]

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=self.config.tensor_model_parallel_size,
            use_fp8=self.config.fp8,
            ub_cfgs=ub_cfgs,
        )
        self.initialize_ub = False

    def training_step(self, dataloader_iter, batch_idx):
        """
            We pass the dataloader iterator function to the micro-batch scheduler.
            The input batch to each micro-batch is fetched using the dataloader function
            in the micro-batch fwd function.
        """
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if self.rampup_batch_size:
            num_microbatch_calculator = apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR
            current_global_batch_size = num_microbatch_calculator.current_global_batch_size
            # do validation and save the checkpoint when gbs is changed
            if self.prev_global_batch_size != current_global_batch_size and self.prev_global_batch_size:
                self.trainer.should_stop = True

        # we zero grads here because we also call backward in the megatron-core fwd/bwd functions
        self._optimizer.zero_grad()

        if self.with_distributed_adam:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, (Float16Module, MCoreFloat16Module)):
                    module = module.module
                if not self.mcore_gpt:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        loss_mean = self.fwd_bwd_step(dataloader_iter, batch_idx, False)

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.config.tensor_model_parallel_size > 1 and self.config.sequence_parallel:
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_o2:
            # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
            if self.config.pipeline_model_parallel_size > 1 or self.config.sequence_parallel:
                # main grads are stored in the MainParamsOptimizer wrapper
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.config.pipeline_model_parallel_size > 1 and self.cfg.get(
            'share_embeddings_and_output_weights', True
        ):
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        ## logging
        # we can only log on one rank if it is rank zero so we broadcast from last rank
        # we can avoid this broadcast by updating the PTL log function to accept specific ranks
        torch.distributed.broadcast(loss_mean, get_last_rank())

        # (@adithyare) we need to check for the _scaler attribute to enable pp>1 for adapter training
        if self.config.precision == 16 and hasattr(self.trainer.precision_plugin.scaler, "_scale"):
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)

        self.log('reduced_train_loss', loss_mean, prog_bar=True, rank_zero_only=True, batch_size=1)
        lr = self._optimizer.param_groups[0]['lr']
        self.log('lr', lr, rank_zero_only=True, batch_size=1)
        self.log(
            'global_step', self.trainer.global_step, prog_bar=True, rank_zero_only=True, batch_size=1,
        )

        consumed_samples = self._compute_consumed_samples_after_training_step()
        # TODO: make sure compute_consumed_samples works for pipeline parallelism
        self.log(
            'consumed_samples', consumed_samples, prog_bar=True, rank_zero_only=True, batch_size=1,
        )

        if self.rampup_batch_size:
            self.prev_global_batch_size = current_global_batch_size
            self.prev_consumed_samples = consumed_samples
            num_microbatch_calculator.update(
                consumed_samples=consumed_samples, consistency_check=False,
            )
            current_global_batch_size = num_microbatch_calculator.current_global_batch_size
            self.log('global_batch_size', current_global_batch_size, prog_bar=True, rank_zero_only=True, batch_size=1)
            self.if_first_step = 1

        return loss_mean

    def backward(self, *args, **kwargs):
        """ LightningModule hook to do backward.
            We want this to do nothing since we run backward in the fwd/bwd functions from megatron-core.
            No need to call it here.
        """
        return

    def optimizer_zero_grad(self, *args, **kwargs):
        """ LightningModule hook to zero grad.
            We want this to do nothing as we are zeroing grads during the training_step.
        """
        return

    def _append_sequence_parallel_module_grads(self, module, grads):
        """ Helper method for allreduce_sequence_parallel_gradients"""

        for param in module.parameters():
            sequence_parallel_param = getattr(param, 'sequence_parallel', False) or getattr(
                param, 'sequence_parallel_enabled', False
            )
            # (@adithyare) adapter training now extends MegatronGPTModel
            # so we have to add this check here to ensure we do not
            # perform all_reduce when grad is None.
            # grad can be None when performing PeFT training.
            if sequence_parallel_param and param.requires_grad:
                if self.megatron_amp_o2:
                    grad = param.main_grad
                else:
                    grad = param.grad
                grads.append(grad.data)

    def allreduce_sequence_parallel_gradients(self):
        """ All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.
            Modified from megatron-lm:
            https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
        """

        grads = []
        if isinstance(self.model, list):
            for module in self.model:
                self._append_sequence_parallel_module_grads(module, grads)
        else:
            self._append_sequence_parallel_module_grads(self.model, grads)

        coalesced = torch._utils._flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    def allreduce_first_last_embeddings(self):

        # Modified from megatron-lm: https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/training.py#L407
        # All-reduce word_embeddings' grad across first and last stages to ensure
        # that word_embeddings parameters stay in sync.
        # This should only run for models that support pipelined model parallelism
        # (BERT and GPT-2).
        if parallel_state.get_pipeline_model_parallel_world_size() > 1 and (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            module_list = self.get_gpt_module_list()
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                module = module_list[0]  # only the first virtual rank has the embeddings
            elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                module = module_list[-1]  # only the last virtual rank has the embeddings
            share_embeddings = (
                module.share_embeddings_and_output_weights if self.mcore_gpt else module.share_token_embeddings
            )
            if share_embeddings:
                word_embeddings_weight = (
                    module.shared_embedding_or_output_weight() if self.mcore_gpt else module.word_embeddings_weight()
                )
                # (@adithyare) adapter training now extends MegatronGPTModel so we have to add this check here to ensure we do not perform all_reduce when grad is None.
                # grad can be None when performing PeFT training.
                if word_embeddings_weight.requires_grad:
                    if self.megatron_amp_o2:
                        # O2 recipe stores a "main" copy of weights and grads
                        grad = word_embeddings_weight.main_grad
                    else:
                        grad = word_embeddings_weight.grad
                    torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())

    def _make_data_iterator_list(self, data_iterator: Iterator) -> List[Iterator]:
        """ Convert data iterator into form expected by Megatron

            With interleaved pipeline parallelism, Megatron expects a
            list of one data iterator per model chunk. Each model
            chunk independently gets data from its data iterator, so
            we need to interact with the data iterator multiple times
            for each microbatch step. Instead of incorporating this
            logic into the data loader, we cache the iterator's output
            to the first model chunk and reuse it in the other model
            chunks.
        """

        if not isinstance(self.model, list) or len(self.model) == 1:
            return data_iterator  # TODO @tmoon: Remove
            # TODO @tmoon: Use once available in Megatron-LM
            # return DataIteratorList([data_iterator])

        class CachingIterator:
            """Iterator wrapper that caches values"""

            class Proxy:
                """Returns values from caching iterator wrapper

                Assumed to never advance past the caching iterator.
                """

                def __init__(self):
                    self.cache = queue.Queue()

                def __iter__(self):
                    return self

                def __next__(self):
                    return self.cache.get_nowait()

            def __init__(self, iterator: Iterator):
                self.iterator = iterator
                self.proxies = []

            def make_proxy(self):
                self.proxies.append(CachingIterator.Proxy())
                return self.proxies[-1]

            def __iter__(self):
                return self

            def __next__(self):
                val = next(self.iterator)
                for proxy in self.proxies:
                    proxy.cache.put(val)
                return val

        # Make list of iterator wrappers
        iters = [CachingIterator(data_iterator)]
        while len(iters) < len(self.model):
            iters.append(iters[0].make_proxy())
        return iters  # TODO @tmoon: Remove
        # TODO @tmoon: Use once available in Megatron-LM
        # return DataIteratorList(iters)

    def get_forward_output_and_loss_func(self, validation_step=False):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):

            # Get data batch
            batch = next(dataloader_iter)

            # Transfer needed data to GPU
            required_keys = set()
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                required_keys.update(batch.keys())
            else:
                required_keys.add('attention_mask')
                if parallel_state.is_pipeline_first_stage():
                    required_keys.update(('tokens', 'position_ids'))
                if parallel_state.is_pipeline_last_stage():
                    required_keys.update(('labels', 'loss_mask'))
            if self.get_attention_mask_from_fusion:
                required_keys.remove('attention_mask')
            batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in batch.items()}

            # Model forward pass
            forward_args = {
                'input_ids': batch['tokens'],
                'position_ids': batch['position_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels'],
                'loss_mask': batch['loss_mask'],
            }
            if not self.mcore_gpt:
                forward_args['checkpoint_activations_all_layers'] = checkpoint_activations_all_layers
            output_tensor = model(**forward_args)

            def loss_func(output_tensor):
                # Loss for a micro-batch (ub)
                loss_for_ub = self.loss_func(batch['loss_mask'], output_tensor)
                if validation_step and not self.cfg.data.get('validation_drop_last', True):
                    num_valid_tokens_in_ub = batch['loss_mask'].sum()
                    if loss_for_ub.isnan():
                        assert batch['loss_mask'].count_nonzero() == 0, 'Got NaN loss with non-empty input'
                        loss_sum_for_ub = torch.zeros_like(num_valid_tokens_in_ub)
                    else:
                        loss_sum_for_ub = num_valid_tokens_in_ub * loss_for_ub

                    loss_sum_and_ub_size_all_gpu = torch.cat(
                        [
                            loss_sum_for_ub.clone().detach().view(1),
                            torch.tensor([num_valid_tokens_in_ub]).cuda().clone().detach(),
                        ]
                    )
                    # Could potentially reduce num_valid_samples_in_microbatch and use that to aggregate instead of len(self._validation_ds)
                    torch.distributed.all_reduce(
                        loss_sum_and_ub_size_all_gpu, group=parallel_state.get_data_parallel_group()
                    )
                    return loss_for_ub, {'loss_sum_and_ub_size': loss_sum_and_ub_size_all_gpu}
                else:
                    reduced_loss = average_losses_across_data_parallel_group([loss_for_ub])
                    return loss_for_ub, {'avg': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def get_forward_output_only_func(self):
        def fwd_output_only_func(dataloader_iter, model):
            batch = next(dataloader_iter)
            extra_arg = {}
            if len(batch) == 3:
                batch = [x.cuda() for x in batch]
                tokens, attention_mask, position_ids = batch
                attention_mask = attention_mask[0:1]
            else:
                (
                    tokens,
                    attention_mask,
                    position_ids,
                    set_inference_key_value_memory,
                    inference_max_sequence_len,
                ) = batch
                tokens = tokens.cuda()
                position_ids = position_ids.cuda()
                if attention_mask is not None:
                    attention_mask = attention_mask.cuda()
                    attention_mask = attention_mask[0:1]
                extra_arg['set_inference_key_value_memory'] = set_inference_key_value_memory[0].item()
                extra_arg['inference_max_sequence_len'] = inference_max_sequence_len[0].item()
            output_tensor = model(tokens, position_ids, attention_mask, **extra_arg)

            def id_func(output_tensor):
                return output_tensor, {'logits': output_tensor}

            return output_tensor, id_func

        return fwd_output_only_func

    def validation_step(self, dataloader_iter, batch_idx):
        """
            Our dataloaders produce a micro-batch and then we fetch
            a number of microbatches depending on the global batch size and model parallel size
            from the dataloader to produce a list of microbatches.
            The list of microbatches is then piped through the pipeline using megatron-core fwd/bwd functions.
        """
        # Prefetch the dataloader_iter before fwd_bwd func to avoid PP rank 2 from waiting indefinitely when PP rank 1 reaches the end of dataloader_iter
        dataloader_iter, done = self._prefetch(dataloader_iter)
        if done:
            return
        mode = 'test' if self.trainer.testing else 'val'
        # Initialize userbuffer communicators.
        if self.initialize_ub:
            self.initialize_ub_func()

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.eval()

        loss = self.fwd_bwd_step(dataloader_iter, batch_idx, True)

        if isinstance(self.model, list):
            for model_module in self.model:
                model_module.train()
        self.validation_step_outputs.append(loss) if mode == 'val' else self.test_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        if parallel_state.is_pipeline_last_stage():
            # only the last pipeline parallel stages return loss with their batch size
            if self.cfg.data.get('validation_drop_last', True):
                averaged_loss = torch.stack(self.validation_step_outputs).mean()
            else:
                # Compute the avg loss by total_loss across all samples / total number of samples
                total_loss_and_total_samples = torch.vstack(self.validation_step_outputs).sum(axis=0)
                avg_loss = total_loss_and_total_samples[0] / total_loss_and_total_samples[1]
                averaged_loss = avg_loss.type(torch.float32).cuda()
        else:
            averaged_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        # we can only log on one rank if it is rank zero so we broadcast from last rank
        torch.distributed.broadcast(averaged_loss, get_last_rank())

        self.log('val_loss', averaged_loss, prog_bar=True, rank_zero_only=True, batch_size=1)
        self.validation_step_outputs.clear()  # free memory

        return averaged_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        averaged_loss = average_losses_across_data_parallel_group(self.test_step_outputs)
        logging.info(f'test_loss: {averaged_loss[0]}')
        self.test_step_outputs.clear()  # free memory

    def loss_func(self, loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        # TODO: add nemo version here
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()  # sequence level nll
        return loss

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """
        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert(
            self.model
        )

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if self.rampup_batch_size:
            optimizer = self.cfg.optim.get('name', None)
            assert (
                optimizer == 'fused_adam'
            ), f'{optimizer} optimizer is not supported yet with rampup batch size. Please, use fused_adam optimizer instead.'

            num_microbatch_calculator = apex.transformer.pipeline_parallel.utils._GLOBAL_NUM_MICROBATCHES_CALCULATOR
            num_microbatch_calculator.update(self.init_consumed_samples, consistency_check=False)
            self.prev_consumed_samples = self.init_consumed_samples

        if stage == 'predict':
            return

        if stage == 'fit':
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                if self.cfg.get('share_embeddings_and_output_weights', True):
                    for index, module in enumerate(self.get_gpt_module_list()):
                        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                            parallel_state.set_virtual_pipeline_model_parallel_rank(index)
                        sync_embeddings = (
                            module.initialize_last_stage_with_word_embeddings
                            if self.mcore_gpt
                            else module.sync_initial_word_embeddings
                        )
                        sync_embeddings()
                    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                        parallel_state.set_virtual_pipeline_model_parallel_rank(0)

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_gpt', False):
            self.setup_transformer_engine_tp_groups()

    def generate(
        self,
        inputs: Union[List[str], torch.Tensor, List[dict]],
        length_params: LengthParam,
        sampling_params: SamplingParam = None,
    ) -> OutputType:

        # check whether the DDP is initialized
        if parallel_state.is_unitialized():

            def dummy():
                return

            if self.trainer.strategy.launcher is not None:
                self.trainer.strategy.launcher.launch(dummy, trainer=self.trainer)
            self.trainer.strategy.setup_environment()

            if self.cfg.get('transformer_engine', False):
                self.setup_transformer_engine_tp_groups()

        # set the default sampling params if it is None.
        # default do greedy sampling
        if sampling_params is None:
            sampling_params = get_default_sampling_params()

        # set the default length params if it is None.
        # default do greedy sampling
        if length_params is None:
            length_params = get_default_length_params()

        return megatron_gpt_generate(self.cuda(), inputs, self.tokenizer, length_params, sampling_params)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        inference_config = self.get_inference_config()
        if inference_config is None:
            return None
        else:
            # need to overwrite some configuration, make it immutable
            inference_config = inference_config.copy()
            compute_logprob = inference_config['compute_logprob']
            if compute_logprob:
                inference_config['inputs'] = batch
                inference_config['tokens_to_generate'] = 1
                inference_config['all_probs'] = True
                inference_config["add_BOS"] = False
                inference_config['greedy'] = True
                response = generate(self, **inference_config)
                compute_prob_response = get_computeprob_response(self.tokenizer, response, batch)
                return compute_prob_response
            else:
                inference_config['inputs'] = batch
                return generate(self, **inference_config)

    def list_available_models(self):
        return None

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """ PTL hook: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#transfer-batch-to-device
            When using pipeline parallelism, we need the global batch to remain on the CPU,
            since the memory overhead will be too high when using a large number of microbatches.
            Microbatches are transferred from CPU to GPU inside the pipeline.
        """
        return batch

    def _validate_trainer(self):
        """ Certain trainer configurations can break training.
            Here we try to catch them and raise an error.
        """
        if self.trainer.accumulate_grad_batches > 1:
            raise ValueError(
                f'Gradient accumulation is done within training_step. trainer.accumulate_grad_batches must equal 1'
            )

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="megatron_gpt_345m",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/megatron_gpt_345m/versions/1/files/megatron_gpt_345m.nemo",
                description="345M parameter GPT generative Megatron model.",
            )
        )
        return result

    def setup_training_data(self, cfg):
        pass
    
    def setup_validation_data(self, cfg):
        pass
    
    def setup_transformer_engine_tp_groups(self):
        """ This should be called after model parallel groups have been initialized
            and only needs to be called when using Transformer Engine.
        """

        for module in self.get_gpt_module_list():
            """Set TP group
               Copied from: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py#L398
            """
            # Deep iterate but skip self to avoid infinite recursion.
            for index, child in enumerate(module.modules()):
                if index == 0:
                    continue
                if hasattr(child, "set_tensor_parallel_group"):
                    tp_group = parallel_state.get_tensor_model_parallel_group()
                    child.set_tensor_parallel_group(tp_group)

    def on_save_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-save-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                checkpoint[f'model{i}'] = self.model[i].module.state_dict_for_save_checkpoint()
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def on_load_checkpoint(self, checkpoint) -> None:
        """LightningModule hook:
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#on-load-checkpoint
        """
        if isinstance(self.model, list):
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                self.model[i].module.load_state_dict(checkpoint[f'model{i}'], strict=True)
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    def parameters(self):
        if isinstance(self.model, list):
            return itertools.chain.from_iterable(module.parameters() for module in self.model)
        else:
            return self.model.parameters()

    @property
    def mgpt_wrapper(self):
        return MegatronGPTExportableModel(self)

    def list_export_subnets(self):
        return ['mgpt_wrapper']

    def _reset_activation_checkpointing_args(self):
        """ Disables activation checkpointing completely and saves the values so that
            _restore_activation_checkpointing_args can restore them later. This function must always be
            called before _restore_activation_checkpointing_args.
        """
        # Store values to restore them later.
        self.last_activations_checkpoint_granularity = self.cfg.activations_checkpoint_granularity
        self.last_activations_checkpoint_method = self.cfg.activations_checkpoint_method
        self.last_activations_checkpoint_num_layers = self.cfg.activations_checkpoint_num_layers
        self.last_activations_checkpoint_layers_per_pipeline = self.cfg.activations_checkpoint_layers_per_pipeline

        # Reset config values. Needed for calling generate.
        self.cfg.activations_checkpoint_granularity = None
        self.cfg.activations_checkpoint_method = None
        self.cfg.activations_checkpoint_num_layers = None
        self.cfg.activations_checkpoint_layers_per_pipeline = None

        # Reset model parameters.
        for module in self.get_gpt_module_list():
            # TODO: @eharper how to update this for mcore
            module.language_model.encoder.activations_checkpoint_granularity = None
            module.language_model.encoder.activations_checkpoint_method = None
            module.language_model.encoder.activations_checkpoint_num_layers = None
            module.language_model.encoder.activations_checkpoint_layers_per_pipeline = None

    def _restore_activation_checkpointing_args(self):
        """ Restores the activation checkpointing parameters using the values saved by
            _reset_activation_checkpointing_args. This function must never be called before
            _reset_activation_checkpointing_args.
        """
        # Restore config values.
        # TODO: @eharper how to update this for mcore
        self.cfg.activations_checkpoint_granularity = self.last_activations_checkpoint_granularity
        self.cfg.activations_checkpoint_method = self.last_activations_checkpoint_method
        self.cfg.activations_checkpoint_num_layers = self.last_activations_checkpoint_num_layers
        self.cfg.activations_checkpoint_layers_per_pipeline = self.last_activations_checkpoint_layers_per_pipeline

        # Restore model parameters.
        for module in self.get_gpt_module_list():
            module.language_model.encoder.activations_checkpoint_granularity = (
                self.last_activations_checkpoint_granularity
            )
            module.language_model.encoder.activations_checkpoint_method = self.last_activations_checkpoint_method
            module.language_model.encoder.activations_checkpoint_num_layers = (
                self.last_activations_checkpoint_num_layers
            )
            module.language_model.encoder.activations_checkpoint_layers_per_pipeline = (
                self.last_activations_checkpoint_layers_per_pipeline
            )

    def _reset_sequence_parallelism_args(self):
        """ Disables sequence parallelism completely and saves the values so that
            _restore_sequence_parallelism_args can restore them later. This function must always be
            called before _restore_sequence_parallelism_args.
        """
        # Store values to restore them later.
        self.last_sequence_parallel = self.cfg.sequence_parallel

        # Reset config values. Needed for calling generate.
        self.cfg.sequence_parallel = False
        self.model_parallel_config.sequence_parallel = False
        self.transformer_config.sequence_parallel = False

        # Reset model parameters.
        for module in self.get_gpt_module_list():
            for mod in module.modules():
                if hasattr(mod, "sequence_parallel"):
                    mod.sequence_parallel = False

    def _restore_sequence_parallelism_args(self):
        """ Restores the sequence parallelism parameters using the values saved by
            _reset_sequence_parallelism_args. This function must never be called before
            _reset_sequence_parallelism_args.
        """
        # Restore config values.
        self.cfg.sequence_parallel = self.last_sequence_parallel
        self.model_parallel_config.sequence_parallel = self.last_sequence_parallel
        self.transformer_config.sequence_parallel = self.last_sequence_parallel

        # Restore model parameters.
        for module in self.get_gpt_module_list():
            for mod in module.modules():
                if hasattr(mod, "sequence_parallel"):
                    mod.sequence_parallel = self.last_sequence_parallel

    def build_transformer_config(self) -> TransformerConfig:
        """ Builds the megatron core gpt transformer config for the model.
            For attributes in the nemo model config that are the same 
            as the megatron core TransformerConfig, we will use the value from the nemo model config.
            For attributes in TransformerConfig that are not in the nemo model config, we add custom logic.
        """

        # create a dictionary copy of the model config
        cfg = OmegaConf.to_container(self.cfg, resolve=True)

        # create a dict to store the transformer config arguments
        transformer_config_dict = {}

        # get model parallel configs from the base class
        model_parallel_config = self.build_model_parallel_config()

        add_bias_linear = self.cfg.get('bias', True)

        activation = self.cfg.get('activation', 'gelu')
        # TODO: need to check which activation functions are supported in mcore
        activation_func = activation_to_func(activation)

        init_method_std = self.cfg.get('init_method_std', 0.02)
        # default used in mcore
        init_method = init_method_normal(init_method_std)

        output_layer_init_method = init_method
        num_layers = self.cfg.get('num_layers', 1)
        use_scaled_init_method = self.cfg.get('use_scaled_init_method', True)
        if use_scaled_init_method:
            output_layer_init_method = scaled_init_method_normal(init_method_std, num_layers=num_layers)

        attention_softmax_in_fp32 = False  # not currently used in NeMo unless apply_query_key_layer_scaling is True
        apply_query_key_layer_scaling = self.cfg.get('apply_query_key_layer_scaling', False)
        if apply_query_key_layer_scaling:
            attention_softmax_in_fp32 = True

        bias_activation_fusion = self.cfg.get('bias_activation_fusion', True)
        bias_gelu_fusion = True if bias_activation_fusion else False

        bias_dropout_fusion = self.cfg.get('bias_dropout_add_fusion', True)

        # TODO: need to check if recompute APIs are matching up properly
        recompute_granularity = self.cfg.get('activations_checkpoint_granularity', None)
        recompute_method = self.cfg.get('activations_checkpoint_method', None)
        recompute_num_layers = self.cfg.get('activations_checkpoint_num_layers', None)

        # any configs that are not in the nemo model config will be added here
        config_mapping = {
            'apply_residual_connection_post_layernorm': False,  # we don't use this in NeMo
            'layernorm_zero_centered_gamma': False,  # not currently used in NeMo
            'add_bias_linear': add_bias_linear,
            'gated_linear_unit': False,  # TODO: is this used in NeMo?
            'activation_func': activation_func,
            'init_method': init_method,
            'output_layer_init_method': output_layer_init_method,
            'attention_softmax_in_fp32': attention_softmax_in_fp32,
            'bias_gelu_fusion': bias_gelu_fusion,
            'bias_dropout_fusion': bias_dropout_fusion,
            'recompute_granularity': recompute_granularity,
            'recompute_method': recompute_method,
            'recompute_num_layers': recompute_num_layers,
            'distribute_saved_activations': False,  # not currently used in NeMo
        }

        # populate the transformer config dict
        for field in fields(TransformerConfig):
            # model config has priority
            if field.name in cfg:
                transformer_config_dict[field.name] = cfg[field.name]
            # then model parallel config
            elif field in fields(model_parallel_config):
                transformer_config_dict[field.name] = getattr(model_parallel_config, field.name)
            # then config mapping
            elif field.name in config_mapping:
                transformer_config_dict[field.name] = config_mapping[field.name]
            else:
                logging.warning(
                    f"The model: {self} does not have field.name: {field.name} in its cfg. "
                    f"Add this key to cfg or config_mapping to make to make it configurable."
                )

        transformer_config = TransformerConfig(**transformer_config_dict)

        return transformer_config

    def _wrap_model_for_O2(self):
        """ Wraps self.model in a float16 wrapper if the model is using megatron amp O2.
            Args:
                model: The model to wrap. Can be a list of modules or a single module.
            Returns:
                The wrapped model. Returns a list of wrapped modules or a single wrapped module.
        """
        Float16Wrapper = MCoreFloat16Module if self.mcore_gpt else Float16Module

        nemo_args = {
            'config': self.model_parallel_config,
            'precision': self.cfg.precision,
            'share_token_embeddings': self.cfg.get('share_embeddings_and_output_weights', True),
        }
        mcore_args = {
            'config': self.transformer_config,
        }

        args = mcore_args if self.mcore_gpt else nemo_args

        # Model wrapper to convert both model and inputs to half precision
        if isinstance(self.model, list):
            converted_model = []
            for module in self.model:
                args['module'] = module
                converted_model.append(Float16Wrapper(**args))
            self.model = converted_model
        else:
            args['module'] = self.model
            self.model = Float16Wrapper(**args)

        args.pop('module')
