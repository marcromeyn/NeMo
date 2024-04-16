from nemo import (
    io,
    lightning as nl,
)


class TestTrainer:
    def test_reinit(self):
        trainer = nl.Trainer(devices=1, accelerator="cpu", strategy=nl.MegatronStrategy())
        copy_trainer = io.reinit(trainer)
        
        assert copy_trainer.__io__.devices == 1
        assert copy_trainer.accelerator.__class__.__name__ == "CPUAccelerator"
        assert isinstance(copy_trainer.strategy, nl.MegatronStrategy)
