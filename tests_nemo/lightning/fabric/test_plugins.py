import pytest
import torch
from nemo.lightning import MegatronMixedPrecision


class TestMegatronMixedPrecision:
    @pytest.mark.parametrize(
        "precision, amp_02, device, scaler, expected_scaler_type",
        [
            ("16-mixed", False, "cuda", None, torch.cuda.amp.GradScaler),
            ("16-mixed", True, "cpu", None, torch.cuda.amp.GradScaler),
            ("16-mixed", True, "cuda", "test", None),
            ("16-mixed", True, "cuda", torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler),
            ("16-mixed", True, "cuda", None, torch.cuda.amp.GradScaler),
        ],
    )
    def test_valid_init(self, precision, amp_02, device, scaler, expected_scaler_type):
        plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
        assert plugin.precision == precision
        assert plugin.amp_02 == amp_02
        assert plugin.device == device
        if expected_scaler_type:
            assert isinstance(plugin.scaler, expected_scaler_type)
    
    
    # def test_amp_02_false(self):
    #     precision = "16-mixed"
    #     amp_02 = False
    #     device = "cuda"
    #     scaler = None
    #     plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
    #     assert plugin.precision == precision
    #     assert plugin.amp_02 == amp_02
    #     assert plugin.device == device
    #     assert isinstance(plugin.scaler, torch.cuda.amp.GradScaler)

    # def test_device_cpu(self):
    #     precision = "16-mixed"
    #     amp_02 = True
    #     device = "cpu"
    #     scaler = None
    #     plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
    #     assert plugin.precision == precision
    #     assert plugin.amp_02 == amp_02
    #     assert plugin.device == device
    #     assert isinstance(plugin.scaler, torch.cuda.amp.GradScaler)

    # def test_scaler_str(self):
    #     precision = "16-mixed"
    #     amp_02 = True
    #     device = "cuda"
    #     scaler = "test"
    #     plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
    #     assert plugin.precision == precision
    #     assert plugin.amp_02 == amp_02
    #     assert plugin.device == device
    #     assert plugin.scaler == scaler

    # def test_scaler_gradscaler(self):
    #     precision = "16-mixed"
    #     amp_02 = True
    #     device = "cuda"
    #     scaler = torch.cuda.amp.GradScaler()
    #     plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
    #     assert plugin.precision == precision
    #     assert plugin.amp_02 == amp_02
    #     assert plugin.device == device
    #     assert plugin.scaler == scaler
    # def test_init(self):
    #     precision = "16-mixed"
    #     amp_02 = True
    #     device = "cuda"
    #     scaler = None
    #     plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
    #     assert plugin.precision == precision
    #     assert plugin.amp_02 == amp_02
    #     assert plugin.device == device
    #     assert isinstance(plugin.scaler, torch.cuda.amp.GradScaler)

    # def test_init_bf16_mixed(self):
    #     precision = "bf16-mixed"
    #     amp_02 = True
    #     device = "cuda"
    #     scaler = None
    #     plugin = MegatronMixedPrecision(precision, amp_02, device, scaler)
    #     assert plugin.precision == precision
    #     assert plugin.amp_02 == amp_02
    #     assert plugin.device == device
    #     assert plugin.scaler == None

    def test_init_invalid_precision(self):
        precision = "invalid"
        amp_02 = True
        device = "cuda"
        scaler = None
        with pytest.raises(ValueError):
            MegatronMixedPrecision(precision, amp_02, device, scaler)
