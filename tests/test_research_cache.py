from csi_robustbench.cache import corruption_cache_name, stable_json_hash, validate_cache_metadata

def test_cache_key_contains_hashes():
    h = stable_json_hash({"a": 1})
    name = corruption_cache_name(dataset="UT_HAR", corruption="gaussian", severity=1, seed=42, config_hash=h, data_hash="abcdef1234567890")
    assert "cfg" + h in name
    assert "dataabcdef123456" in name

def test_cache_metadata_mismatch_raises():
    try:
        validate_cache_metadata({"seed": 1}, {"seed": 2})
    except ValueError:
        return
    raise AssertionError("expected mismatch")
