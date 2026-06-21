import torch
import yaml

from csi_er.models import build_model


def test_deep_model_bank_factory_accepts_all_configured_families():
    cfg = yaml.safe_load(open("configs/deep_model_bank.yaml", encoding="utf-8"))
    x = torch.randn(2, 1, 32, 16)
    for family, sizes in cfg["families"].items():
        params = dict(sizes["small"])
        model = build_model(family, 7, input_shape=(1, 32, 16), **params)
        y = model(x)
        assert y.shape == (2, 7)
