from nfp.registry import load_registry

REG = "data/registry/registry_sample.csv"


def test_registry_loads_and_finds_traits():
    reg = load_registry(REG)
    assert len(reg.trait_codes) == 8
    assert "PSY_RES" in reg.trait_codes
    assert reg.label_for("COG_TEC") == "Technical aptitude"


def test_registry_blocks():
    reg = load_registry(REG)
    assert set(reg.blocks) >= {"PSY", "SOC", "COG", "PHY", "LDR", "OUT"}
