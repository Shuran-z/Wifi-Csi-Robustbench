import numpy as np

def test_er_formula_signs():
    clean_acc=0.8; noise_accs=[0.7,0.6,0.6,0.5,0.5]
    e_clean=1-clean_acc; e_noise=1-np.mean(noise_accs); e_hat=0.35
    csi_er=e_hat-e_noise
    assert abs(e_clean-0.2)<1e-9
    assert abs(e_noise-0.42)<1e-9
    assert csi_er<0
    assert (0.50-e_noise)>0
