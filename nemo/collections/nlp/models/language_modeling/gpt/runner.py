import os
from typing import Tuple, Union

from omegaconf import OmegaConf, DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from nemo.collections.nlp.models.language_modeling.gpt.config import GPTConfig
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
    NLPSaveRestoreConnector,
    PEFTSaveRestoreConnector,
)
from nemo.utils.exp_manager import exp_manager as exp_manager_fn


def gpt_pre_training(
    config: Union[OmegaConf, GPTConfig],
    trainer: Union[OmegaConf, Trainer],
    exp_manager: OmegaConf # TODO: Turn this into config-class
):
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    
    if isinstance(config, DictConfig):
        cfg = config
        gpt_config = GPTConfig.from_flattened_cfg(config)
    else:
        raise NotImplementedError()    
    
    if isinstance(trainer, DictConfig):
        trainer = Trainer(
            plugins=default_trainer_plugins(gpt_config, trainer.precision), 
            strategy=default_trainer_strategy(gpt_config), 
            **trainer
        )
        
    exp_manager_fn(trainer, exp_manager)
    
    # TODO: Where to add this snippet when model_config can be passed to MegatronGPTModel?
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    
    model = MegatronGPTModel(cfg, trainer)

    trainer.fit(model)
    
    return model, trainer


def gpt_peft(
    config: Union[OmegaConf, GPTConfig],
    trainer: Union[OmegaConf, Trainer],
    exp_manager: OmegaConf # TODO: Turn this into config-class
):
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    
    if isinstance(config, DictConfig):
        cfg = config
        # gpt_config = GPTConfig.from_flattened_cfg(config)
    else:
        raise NotImplementedError()    
    
    if isinstance(trainer, DictConfig):
        cfg_trainer = trainer
        trainer = Trainer(
            plugins=default_trainer_plugins(cfg, trainer.precision), 
            strategy=default_trainer_strategy(cfg), 
            **trainer
        )
        
    exp_manager_fn(trainer, exp_manager)
    
    if not cfg.get("restore_from_path", False):
        raise RuntimeError("PEFT training needs a trained base model present.")
    
    if cfg.resume_from_checkpoint is not None:
        trainer.ckpt_path = cfg.resume_from_checkpoint
        
    with open_dict(cfg):
        cfg.precision = cfg_trainer.precision
        
    base_model_save_restore_connector = NLPSaveRestoreConnector()
    if os.path.isdir(cfg.restore_from_path):
        base_model_save_restore_connector.model_extracted_dir = cfg.restore_from_path
    base_model_cfg = MegatronGPTModel.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        return_config=True,
        save_restore_connector=base_model_save_restore_connector,
    )
    base_model_cfg = _modify_config(base_model_cfg, cfg, add_cfg_to_tree=False)
    save_restore_connector = PEFTSaveRestoreConnector(
        peft_model_nemo_path=cfg.peft.restore_from_path, peft_model_ckpt_path=trainer.ckpt_path
    )
    if os.path.isdir(cfg.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.restore_from_path
    peft_cls = _get_peft_scheme(cfg)
    model = peft_cls.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        override_config_path=base_model_cfg,
        save_restore_connector=save_restore_connector,
    )
    
    trainer.fit(model)
    
    return model, trainer
        
        
def default_trainer_strategy(config: GPTConfig) -> NLPDDPStrategy:
    # Replace with your desired logic for GPT
    return NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=config.get('gradient_as_bucket_view', True),
        find_unused_parameters=False,
    )
    

def default_trainer_plugins(config: GPTConfig, precision: Union[str, int]) -> list:
    megatron_amp_o2 = config.get('megatron_amp_O2', False)
    with_distributed_adam = config.optim.name == 'distributed_fused_adam'

    plugins = []
    if precision in [16, '16', 'bf16', '16-mixed', 'bf16-mixed']:
        scaler = None
        if precision in [16, '16', '16-mixed']:
            scaler = GradScaler(
                init_scale=config.get('native_amp_init_scale', 2 ** 32),
                growth_interval=config.get('native_amp_growth_interval', 1000),
                hysteresis=config.get('hysteresis', 2),
            )
            plugin_precision = '16-mixed'
        else:
            plugin_precision = 'bf16-mixed'

        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=plugin_precision, device='cuda', scaler=scaler))

    if config.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    return plugins



def _get_peft_scheme(cfg):
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import (
        MegatronGPTAdapterModel,
        MegatronGPTAdapterModelWeightTying,
        MegatronGPTAdapterPTuningModel,
        MegatronGPTIA3Model,
        MegatronGPTLoRAModel,
        MegatronGPTLoRAModelWeightTying,
        MegatronGPTPTuningModel,
    )
    
    if cfg.peft.peft_scheme == "adapter":
        if cfg.peft.adapter_tuning.weight_tying:
            peft_cls = MegatronGPTAdapterModelWeightTying
        else:
            peft_cls = MegatronGPTAdapterModel
    elif cfg.peft.peft_scheme == "ia3":
        peft_cls = MegatronGPTIA3Model
    elif cfg.peft.peft_scheme == "ptuning":
        peft_cls = MegatronGPTPTuningModel
    elif cfg.peft.peft_scheme == "adapter_and_ptuning":
        peft_cls = MegatronGPTAdapterPTuningModel
    elif cfg.peft.peft_scheme == "lora":
        if cfg.peft.lora_tuning.weight_tying:
            peft_cls = MegatronGPTLoRAModelWeightTying
        else:
            peft_cls = MegatronGPTLoRAModel
    else:
        raise RuntimeError("Invalid Peft scheme")
    return peft_cls


def _modify_config(gpt_cfg, cfg, add_cfg_to_tree=False):
    """
    This function modifies the original gpt pre-training config (gpt_cfg) with attributes from the finetuning config (cfg).
    The `add_cfg_to_tree` arg adds `cfg` to the top of the yaml tree which is needed for all `hparams.yaml` files when passed as an arg to `load_from_checkpoint()`.
    """
    OmegaConf.set_struct(gpt_cfg, True)
    OmegaConf.resolve(cfg)
    with open_dict(gpt_cfg):
        gpt_cfg.megatron_amp_O2 = cfg.get('megatron_amp_O2', False)
        gpt_cfg.micro_batch_size = cfg.data.train_ds.micro_batch_size
        gpt_cfg.global_batch_size = cfg.data.train_ds.global_batch_size
        gpt_cfg.sequence_parallel = cfg.get("sequence_parallel", False)
        gpt_cfg.activations_checkpoint_granularity = cfg.get("activations_checkpoint_granularity", None)
        gpt_cfg.activations_checkpoint_num_layers = cfg.get("activations_checkpoint_num_layers", None)
        gpt_cfg.activations_checkpoint_method = cfg.get("activations_checkpoint_method", None)
        gpt_cfg.activations_checkpoint_layers_per_pipeline = cfg.get(
            "activations_checkpoint_layers_per_pipeline", None
        )
        gpt_cfg.data = cfg.data
        gpt_cfg.optim = cfg.optim
        gpt_cfg.precision = cfg.precision
        gpt_cfg.answer_only_loss = cfg.answer_only_loss
        gpt_cfg.restore_from_path = cfg.restore_from_path
        gpt_cfg.resume_from_checkpoint = cfg.resume_from_checkpoint
        gpt_cfg.save_nemo_on_validation_end = cfg.save_nemo_on_validation_end
        gpt_cfg.gradient_as_bucket_view = cfg.gradient_as_bucket_view
        gpt_cfg.hidden_dropout = cfg.get('hidden_dropout', 0.0)
        gpt_cfg.attention_dropout = cfg.get('attention_dropout', 0.0)
        gpt_cfg.ffn_dropout = cfg.ffn_dropout
        gpt_cfg.peft = cfg.peft
        peft_cls = _get_peft_scheme(cfg)
        gpt_cfg.target = f"{peft_cls.__module__}.{peft_cls.__name__}"

        # This is needed when modifying a hparam file directly to load `.ckpt` files.
        # This is not needed to modify the cfg in `.nemo` files.
        if add_cfg_to_tree:
            OmegaConf.resolve(gpt_cfg)
            gpt_cfg.cfg = gpt_cfg

    return gpt_cfg