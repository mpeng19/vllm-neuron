# SPDX-License-Identifier: Apache-2.0
"""Chunked top-k used on Trn1 / Inf2, where torch.topk over a vocab shard does not compile."""

import pytest
import torch

topk_mod = pytest.importorskip("vllm_neuron.functional.topk")


@pytest.mark.parametrize("shape,k", [((2, 75968), 256), ((1, 4, 5000), 50), ((3, 2049), 7)])
def test_chunked_topk_matches_torch(shape, k):
    x = torch.randperm(torch.Size(shape).numel()).float().view(shape)  # distinct: no ties
    values, index = topk_mod._topk_torch_chunked(x, k, 2048)
    ref = torch.topk(x, k, dim=-1)
    assert torch.equal(values, ref.values)
    assert torch.equal(index, ref.indices) and index.dtype == torch.int64


def test_gen2_fallback_selects_chunked(monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    x = torch.randperm(20000).float().view(2, 10000)
    values, index = topk_mod._topk_torch(x, 16, -1)
    ref = torch.topk(x, 16, dim=-1)
    assert torch.equal(values, ref.values) and torch.equal(index, ref.indices)
