from unittest.mock import Mock

from nemo.lightning import DataModule


class TestDataModule:
    def setup_method(self, method):
        self.config = Mock()
        self.config.num_workers = 2
        self.config.pin_memory = True
        self.config.persistent_workers = True
        self.dataset = Mock()
        self.datamodule = DataModule(self.config)

    def test_init(self):
        assert self.datamodule.config == self.config
        assert self.datamodule.init_consumed_samples == 0
        assert self.datamodule.prev_consumed_samples == 0
        assert self.datamodule.if_first_step == 0
        assert self.datamodule.prev_global_batch_size is None

    def test_to_dataloader(self):
        dataloader = self.datamodule.to_dataloader(self.dataset)
        assert dataloader.num_workers == self.config.num_workers
        assert dataloader.pin_memory == self.config.pin_memory
        assert dataloader.persistent_workers == self.config.persistent_workers
        assert dataloader._drop_last   # noqa: SLF001
        assert not dataloader._pad_samples_to_global_batch_size   # noqa: SLF001

    def test_model_kwargs(self):
        assert self.datamodule.model_kwargs() == {}
