import numpy as np

def test_er_v2_toy_formula():
    clean_acc=np.array([0.8,0.6]); mpc_g=np.array([0.7,0.5]); mpc_t=np.array([0.75,0.45])
    e_clean=1-clean_acc; e_g=1-mpc_g; e_overall=1-np.vstack([mpc_g,mpc_t]).mean(axis=0)
    a,b=np.polyfit(e_clean,e_overall,1); pred=a*e_clean+b; er=pred-e_overall
    assert np.allclose(e_clean,[0.2,0.4])
    assert np.allclose(e_g,[0.3,0.5])
    assert np.allclose(e_overall,[0.275,0.525])
    assert abs(er[0])<1e-9 and abs(er[1])<1e-9
    assert (0.4-0.3)>0
