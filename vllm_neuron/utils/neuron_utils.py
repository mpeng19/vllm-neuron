# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import functools
import os
from typing import TYPE_CHECKING

import torch

from vllm_neuron import envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def can_run_kernel(device: torch.Tensor | str = "") -> bool:
    """Check if NKI kernels can run on the given device."""
    if envs.VLLM_NEURON_DISABLE_NKI_KERNELS:
        return False
    if envs.VLLM_NEURON_CPU_MODE:
        return os.environ.get("NKI_SIMULATOR") == "1"
    device_str = str(device.device) if isinstance(device, torch.Tensor) else device
    return device_str != "cpu"


def nki_grid() -> int:
    """SPMD grid for NKI kernel launches: one program per physical core of a logical
    NeuronCore (NEURON_LOGICAL_NC_CONFIG; 2 on Trn2/Trn3, 1 on Trn1/Inf2, which have no LNC=2)."""
    return int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))


def is_gen2_platform() -> bool:
    """Trn1 / Inf2 (NeuronCore-v2). Kernels that need gen3 (Trn2+) must take another path.

    Uses NEURON_PLATFORM_TARGET_OVERRIDE when set (CPU compile / CPU mode), else the runtime's
    instance info. CPU mode without an override is treated as Trn2 (the default target).
    """
    target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")
    if target is None:
        if envs.VLLM_NEURON_CPU_MODE:
            return False
        target = _runtime_platform_target()
    return target.startswith(("trn1", "inf2"))


@functools.lru_cache(maxsize=1)
def _runtime_platform_target() -> str:
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    return get_platform_target()


def model_forward_context(
    vllm_config: VllmConfig,
) -> contextlib.AbstractContextManager[None]:
    """Context manager for model forward: skips fail_on_recompile in CPU mode."""
    if envs.VLLM_NEURON_CPU_MODE and vllm_config.model_config.enforce_eager:
        return contextlib.nullcontext()
    return torch.compiler.set_stance("fail_on_recompile")
