# SPDX-License-Identifier: Apache-2.0
"""Trn1 (NeuronCore-v2) platform selection: instance family, NKI grid, gen2 detection."""

import pytest

hardware_config = pytest.importorskip("vllm_neuron.utils.hardware_config")
neuron_utils = pytest.importorskip("vllm_neuron.utils.neuron_utils")


@pytest.mark.parametrize("family,expected", [("trn1", "trn1"), ("trn1n", "trn1"), ("trn2", "trn2")])
def test_instance_family_from_product_name(monkeypatch, tmp_path, family, expected):
    monkeypatch.delenv("VLLM_NEURON_EFA_INSTANCE_FAMILY", raising=False)
    real_open = open

    def fake_open(path, *a, **k):
        if path == "/sys/devices/virtual/dmi/id/product_name":
            p = tmp_path / "product_name"
            p.write_text(f"{family}.2xlarge\n")
            return real_open(p, *a, **k)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    assert hardware_config.get_instance_family() == expected


def test_trn1_has_two_cores_per_device_and_no_efa_mapping(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_EFA_INSTANCE_FAMILY", "trn1")
    assert hardware_config.get_cores_per_device() == 2
    with pytest.raises(ValueError, match="NEURON_SKIP_EFA_AFFINITY"):
        hardware_config.get_efa_bdf_mapping()
    monkeypatch.setenv("VLLM_NEURON_EFA_INSTANCE_FAMILY", "trn2")
    assert hardware_config.get_cores_per_device() == 8


def test_nki_grid_follows_logical_nc_config(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    assert neuron_utils.nki_grid() == 1
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert neuron_utils.nki_grid() == 2


@pytest.mark.parametrize("target,gen2", [("trn1", True), ("trn1n", True), ("inf2", True),
                                         ("trn2", False), ("trn3", False)])
def test_gen2_platform_from_target_override(monkeypatch, target, gen2):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
    assert neuron_utils.is_gen2_platform() is gen2
