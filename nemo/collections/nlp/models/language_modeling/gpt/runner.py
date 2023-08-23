from typing import Tuple, Union

from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from nemo.collections.nlp.models.language_modeling.gpt.config import GPTConfig
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.utils.exp_manager import exp_manager as exp_manager_fn
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel


def gpt_pre_training(
    config: Union[OmegaConf, GPTConfig],
    trainer: Union[OmegaConf, Trainer],
    exp_manager: OmegaConf # TODO: Turn this into config-class
) -> Tuple[MegatronGPTModel, Trainer]:
    if isinstance(config, OmegaConf):
        config = GPTConfig.from_flattened_cfg(config)
    
    if isinstance(trainer, OmegaConf):
        trainer = Trainer(
            plugins=default_trainer_plugins(config), 
            strategy=default_trainer_strategy(config), 
            **trainer
        )
        
    exp_manager_fn(trainer, exp_manager)
    
    # TODO: Where to add this snippet?
    """
    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    """
    
    model = MegatronGPTModel(config, trainer)

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
    with_distributed_adam = config.optim.get('name') == 'distributed_fused_adam'

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

    if self.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    return plugins