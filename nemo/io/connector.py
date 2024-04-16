import os
import shutil
from pathlib import Path, PosixPath, WindowsPath
from typing import Generic, Optional, TypeVar

import lightning as L

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == 'nt':
    BasePath = WindowsPath
else:
    BasePath = PosixPath
    
    
SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")

    
class Connector(BasePath, Generic[SourceT, TargetT]):
    default_path = None
    
    def init(self) -> TargetT:
        raise NotImplementedError()
    
    def apply(self, output_path: Path) -> Path:
        raise NotImplementedError()

    def __new__(cls, *args, **kwargs):
        if cls.default_path is not None and not args and 'path' not in kwargs:
            # If default_path is set and no arguments are provided, use default_path as the argument
            return super().__new__(cls, cls.default_path)
        
        return super().__new__(cls, *args, **kwargs)
    
    def __call__(
        self, 
        output_path: Optional[Path] = None, 
        overwrite: bool = False
    ) -> Path:
        _output_path = output_path or self.local_path()
        
        if overwrite and _output_path.exists():
            shutil.rmtree(_output_path)
        
        if not _output_path.exists():
            to_return = self.apply(_output_path)
            _output_path = to_return or _output_path

        return _output_path

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        if base_path:
            _base = base_path
        else:
            from nemo_ext.lightning.base import NEMO_CACHE_HOME

            _base = Path(NEMO_CACHE_HOME)

        return _base / str(self).replace("://", "/")

    def is_in_cache(self, base_path: Optional[Path] = None) -> bool:
        return self.local_path(base_path=base_path).exists()


# TODO: Rename this to CheckpointConnector?
class ModelConnector(Connector, Generic[SourceT, TargetT]):
    def nemo_setup(
        self, 
        model: L.LightningModule, 
        trainer: Optional[L.Trainer] = None
    ) -> L.Trainer:
        from nemo_ext.lightning import MegatronStrategy, Trainer
        
        _trainer = trainer or Trainer(devices=1, accelerator="cpu", strategy=MegatronStrategy())

        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()

        if not model.state_dict():
            _trainer.strategy.lazy_init = True
            with _trainer.init_module():
                model.configure_model()

        return _trainer

    def nemo_save(self, output_path: Path, trainer: L.Trainer):
        trainer.strategy.setup(trainer)
        trainer.save_checkpoint(output_path)

    def nemo_load(self, path: Path, trainer: Optional[L.Trainer] = None, cpu: bool = True):
        from nemo_ext.io.api import load_ckpt
        from nemo_ext.lightning import MegatronStrategy, Trainer, _strategy_lib
        
        model = load_ckpt(path).model
        _trainer = trainer or Trainer(devices=1, accelerator="cpu" if cpu else "gpu", strategy=MegatronStrategy())
        
        _trainer.strategy.connect(model)
        _trainer.strategy.setup_environment()
        # TODO: Fix cpu initialization
        if not model.state_dict():
            if cpu:
                # TODO: Make this more generic
                with _strategy_lib.megatron_cpu_init_context(model.config):
                    model.configure_model()
            else:                
                model.configure_model()
        
        _trainer.strategy.setup(_trainer)
        _trainer.strategy.load_checkpoint(path)

        return model, _trainer
