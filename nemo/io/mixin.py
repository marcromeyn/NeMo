import functools
import inspect
import pickle
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar

import fiddle as fdl
from typing_extensions import Self

from nemo.io.capture import IOProtocol
from nemo.io.connector import ModelConnector

ConnT = TypeVar('ConnT', bound=ModelConnector)


class IOMixin:
    __io__ = fdl.Config[Self]
    
    def __new__(cls, *args, **kwargs):
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            cfg_kwargs = self.io_transform_args(original_init, *args, **kwargs)
            self.__io__ = self.io_init(**cfg_kwargs)
            original_init(self, *args, **kwargs)
        
        cls.__init__ = wrapped_init
        output = object().__new__(cls)
        
        return output
    
    def io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
        sig = inspect.signature(init_fn)
        bound_args = sig.bind_partial(self, *args, **kwargs)
        bound_args.apply_defaults()
        config_kwargs = {k: v for k, v in bound_args.arguments.items() if k != "self"}

        to_del = []
        for key in config_kwargs:
            if isinstance(config_kwargs[key], IOProtocol):
                config_kwargs[key] = config_kwargs[key].__io__
            if is_dataclass(self):
                # Check if the arg is a factory (dataclasses.field)
                if config_kwargs[key].__class__.__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
                    to_del.append(key)

        for key in to_del:
            del config_kwargs[key]
        
        return config_kwargs

    def io_init(self, **kwargs) -> fdl.Config[Self]:
        return fdl.Config(type(self), **kwargs)
    
    def io_dump(self, output: Path):
        config_path = Path(output) / "io.pkl"
        with open(config_path, "wb") as f:
            pickle.dump(self.__io__, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    
class ConnectorMixin:
    _IMPORTERS: Dict[str, Type[ModelConnector]] = {}
    _EXPORTERS: Dict[str, Type[ModelConnector]] = {}
    
    @classmethod
    def import_from(cls, path: str) -> Self:
        output = cls._get_connector(path).init()
        output.ckpt_path = output.import_ckpt_path(path)

        return output
        
    @classmethod
    def register_importer(
        cls,
        ext: str, 
        default_path: Optional[str] = None
    ) -> Callable[[Type[ConnT]], Type[ConnT]]:
        def decorator(connector: Type[ConnT]) -> Type[ConnT]:
            cls._IMPORTERS[ext] = connector
            if default_path:
                connector.default_path = default_path
            return connector
        
        return decorator

    @classmethod
    def register_exporter(
        cls, 
        ext: str, 
        default_path: Optional[str] = None
    ) -> Callable[[Type[ConnT]], Type[ConnT]]:
        def decorator(connector: Type[ConnT]) -> Type[ConnT]:
            cls._EXPORTERS[ext] = connector
            if default_path:
                connector.default_path = default_path
            return connector
        
        return decorator
    
    @classmethod
    def importer(cls, path: str) -> ModelConnector:
        return cls._get_connector(path, importer=True)
    
    @classmethod
    def exporter(cls, ext: str, path: str) -> ModelConnector:
        return cls._get_connector(ext, path, importer=False)
    
    def import_ckpt(
        self, 
        path: str, 
        overwrite: bool = False,
        base_path: Optional[Path] = None
    ) -> Path:
        connector = self._get_connector(path)
        ckpt_path: Path = connector.local_path(base_path=base_path)
        ckpt_path = connector(ckpt_path, overwrite=overwrite)
        
        return ckpt_path
    
    @classmethod
    def _get_connector(cls, ext, path=None, importer=True) -> ModelConnector:
        _path = None
        if "://" in ext:
            ext, _path = ext.split("://")
        else:
            _path = path
        
        connector = cls._IMPORTERS.get(ext) if importer else cls._EXPORTERS.get(ext)
        if not connector:
            raise ValueError(f"No connector found for extension '{ext}'")
        
        if not _path:
            if not connector.default_path:
                raise ValueError(
                    f"No default path specified for extension '{ext}'. ",
                    "Please provide a path"
                )
            
            return connector()

        return connector(_path)
