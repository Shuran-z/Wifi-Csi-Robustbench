import torch
from csi_er.models import build_model

def test_model_parameters_stable_after_forward():
    for name in ['MLP','SimpleCNN','GRU','LSTM','CNNGRU','TinyViT']:
        model=build_model(name,7,input_shape=(1,250,90))
        before={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
        _=model(torch.randn(2,1,250,90))
        after={n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
        assert before==after, name
