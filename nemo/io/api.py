import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar

import fiddle as fdl
import lightning as L

from nemo.io.mixin import ConnectorMixin, ConnT, ModelConnector
from nemo.io.pl import TrainerCheckpoint

CkptType = TypeVar("CkptType")


def load(
    path: Path, 
    output_type: Type[CkptType] = Any
) -> CkptType:
    del output_type     # Just for type-hint
    
    _path = Path(path)
    if hasattr(_path, 'is_dir') and _path.is_dir():
        _path = Path(_path) / "io.pkl"
    elif hasattr(_path, 'isdir') and _path.isdir:
        _path = Path(_path) / "io.pkl"
    
    if not _path.is_file():
        raise FileNotFoundError(f"No such file: '{_path}'")

    with open(_path, "rb") as f:
        config = pickle.load(f)

    return fdl.build(config)


def load_ckpt(path: Path) -> TrainerCheckpoint:
    return load(path, output_type=TrainerCheckpoint)


def model_importer(
    target: Type[ConnectorMixin],
    ext: str,
    default_path: Optional[str] = None
) -> Callable[[Type[ConnT]], Type[ConnT]]:
    return target.register_importer(ext, default_path=default_path)


def model_exporter(
    target: Type[ConnectorMixin],
    ext: str,
    default_path: Optional[str] = None
) -> Callable[[Type[ConnT]], Type[ConnT]]:
    return target.register_exporter(ext, default_path=default_path)


def import_ckpt(
    model: L.LightningModule, 
    source: str, 
    output_path: Optional[Path] = None, 
    overwrite: bool = False
) -> Path:
    importer: ModelConnector = model.importer(source)
    return importer(overwrite=overwrite, output_path=output_path)


def load_connector_from_trainer_ckpt(
    path: Path,
    target: str
) -> ModelConnector:
    return load_ckpt(path).model.exporter(target, path)


def export_ckpt(
    path: Path,
    target: str,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    load_connector: Callable[[Path, str], ModelConnector] = load_connector_from_trainer_ckpt
) -> Path:
    exporter: ModelConnector = load_connector(path, target)
    _output_path = output_path or Path(path) / target

    return exporter(overwrite=overwrite, output_path=_output_path)
