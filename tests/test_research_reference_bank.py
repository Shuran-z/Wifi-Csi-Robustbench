from csi_robustbench.reference_bank import build_reference_configs, load_yaml

def test_reference_bank_has_required_size():
    cfg = load_yaml("configs/reference_bank.yaml")
    rows = build_reference_configs(cfg)
    base = {r.base_configuration_id for r in rows}
    assert len(base) == 144
    assert len(rows) == 720
    first = rows[0]
    assert first.dataset == "UT_HAR"
    assert first.feature_params
    assert first.classifier_params
    assert first.protocol_hash
