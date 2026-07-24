"""Behavioral drift check: our ported GPT vs the sibling's (ADR 0016).

nanochat is not importable as a package (torch pin), so gpt.py was copied and
its three nanochat-internal imports repointed (cankar.model.gpt docstring). This
guards against SILENT architecture drift if the sibling is later updated: build
both GPTs with the same config, copy our weights into the sibling model, and
assert the forward logits are equal. Same weights + same architecture => same
output; any divergence in the forward path fails here. Runs only where the
checkout exists (like the vendored-BPB drift test).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

_CHECKOUT = Path(
    os.environ.get("NANOCHAT_CHECKOUT", str(Path.home() / "PROJECTS" / "PERSONAL" / "nanochat"))
)
SIBLING_GPT = _CHECKOUT / "nanochat" / "gpt.py"

_CFG = dict(
    sequence_len=64, vocab_size=320, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L"
)


def _load_sibling_gpt() -> types.ModuleType:
    """Exec the sibling gpt.py with stubbed nanochat.* modules. gpt.py imports
    only common (COMPUTE_DTYPE/print0/get_dist_info), optim (MuonAdamW, unused in
    forward), and flash_attention.flash_attn (fed OUR identical module)."""
    from cankar.model import compute, flash_attention

    common = types.ModuleType("nanochat.common")
    common.COMPUTE_DTYPE = compute.COMPUTE_DTYPE
    common.print0 = compute.print0
    common.get_dist_info = lambda: (0, 1, 0)  # imported but not called in forward
    optim = types.ModuleType("nanochat.optim")
    optim.MuonAdamW = object  # only setup_optimizer uses it; not the forward path
    fa = types.ModuleType("nanochat.flash_attention")
    fa.flash_attn = flash_attention.flash_attn
    pkg = types.ModuleType("nanochat")
    pkg.__path__ = []  # mark as a package
    for name, mod in [
        ("nanochat", pkg),
        ("nanochat.common", common),
        ("nanochat.optim", optim),
        ("nanochat.flash_attention", fa),
    ]:
        sys.modules[name] = mod
    spec = importlib.util.spec_from_file_location("sibling_gpt", SIBLING_GPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not SIBLING_GPT.exists(), reason="sibling nanochat checkout not present")
def test_ported_gpt_forward_matches_sibling() -> None:
    from cankar.model.gpt import GPT, GPTConfig

    sib = _load_sibling_gpt()
    torch.manual_seed(0)
    ours = GPT(GPTConfig(**_CFG))
    ours.init_weights()
    theirs = sib.GPT(sib.GPTConfig(**_CFG))
    theirs.init_weights()
    theirs.load_state_dict(ours.state_dict())  # identical weights -> isolate the forward path
    ours.eval()
    theirs.eval()

    x = torch.randint(0, 300, (2, 48))
    with torch.no_grad():
        logits_ours = ours(x)
        logits_theirs = theirs(x)
    max_abs = (logits_ours - logits_theirs).abs().max().item()
    assert torch.allclose(logits_ours, logits_theirs, atol=1e-5), f"forward drift: {max_abs}"
