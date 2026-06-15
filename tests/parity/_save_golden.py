"""Save the M0 golden reference tensor for cross-port comparison.

Run as a script:

    .venv/Scripts/python.exe tests/parity/_save_golden.py

Writes ``tests/parity/golden_m0.pt`` containing params, inputs, and the
torch_port output. Other ports load this artefact to assert numerical parity
with the PyTorch ground truth.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Allow running as ``python tests/parity/_save_golden.py`` (without pytest's
# conftest path injection) by adding the repo root to sys.path first.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from runtimes.torch_port import run_block  # noqa: E402

from tests.parity._canonical import _canonical_m0_setup  # noqa: E402


_GOLDEN_PATH = Path(__file__).resolve().parent / "golden_m0.pt"


def main() -> None:
    setup = _canonical_m0_setup()
    out = run_block(setup.graph, setup.params, setup.inputs)

    payload = {
        "params": setup.params,
        "inputs": setup.inputs,
        "output": out,
    }
    torch.save(payload, _GOLDEN_PATH)

    h = hashlib.sha256(out.detach().contiguous().numpy().tobytes()).hexdigest()
    print(f"Saved golden_m0.pt -> {_GOLDEN_PATH}")
    print(f"  shape={tuple(out.shape)} mean={out.mean().item():.4e} std={out.std().item():.4e}")
    print(f"  sha256={h}")


if __name__ == "__main__":
    main()
