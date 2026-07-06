# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build pipeline for pre-exported ONNX models.

Provides build_onnx_model() which runs the same stages as build_hf_model()
minus Load and Export. Intended for users who already have an ONNX file and
want to optimize, quantize, and compile it for WinML deployment.

Pipeline: [Optimize] -> [Analyze<->Optimize] -> [Quantize] -> [Compile] -> [Finalize]
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..onnx import copy_onnx_model
from .common import run_build_stages
from .hf import BuildResult


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig

logger = logging.getLogger(__name__)


def build_onnx_model(
    onnx_path: Path | str,
    *,
    config: WinMLBuildConfig,
    output_dir: Path | str,
    rebuild: bool = False,
    ep: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> BuildResult:
    """Build from a pre-exported ONNX model.

    Pipeline: [Optimize] -> [Analyze<->Optimize] -> [Quantize] -> [Compile] -> [Finalize]

    Same stages as build_hf_model minus Load and Export. The config should
    have ``export=None`` for ONNX builds.

    Args:
        onnx_path: Path to input ONNX model.
        config: Build configuration (export should be None for ONNX builds).
        output_dir: Directory for output artifacts. Created if missing.
        rebuild: Force rebuild even if output exists.
        ep: Target execution provider for the analyzer (e.g., ``"qnn"``).
        device: Target device for the analyzer (e.g., ``"NPU"``).
        **kwargs: Additional options:
            - ``hack_max_optim_iterations`` (int, default 3): Max analyzer
              iterations. 0 disables analyzer.
            - ``use_external_data`` (bool, default True): Whether to use ONNX
              external data format.

    Returns:
        BuildResult with paths to artifacts and build metadata.

    Raises:
        FileNotFoundError: If onnx_path doesn't exist.
        ValueError: If onnx_path is not a file or config validation fails.
        RuntimeError: If a pipeline stage fails.
    """
    hack_max_optim_iterations: int = kwargs.pop("hack_max_optim_iterations", 3)
    onnx_kwargs = {
        "use_external_data": kwargs.get("use_external_data", True),
    }

    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    # =========================================================================
    # [0] VALIDATE & SETUP
    # =========================================================================
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not onnx_path.is_file():
        raise ValueError(f"ONNX path is not a file: {onnx_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")

    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Config validation failed before build: {e}") from e

    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    stem = onnx_path.stem
    optimized_path = output_dir / f"{stem}_optimized.onnx"
    quantized_path = output_dir / f"{stem}_quantized.onnx"
    compiled_path = output_dir / f"{stem}_compiled.onnx"
    final_path = output_dir / "model.onnx"
    config_path = output_dir / "winml_build_config.json"
    manifest_path = output_dir / "build_manifest.json"

    # Check for existing artifact (skip build if present and not rebuilding)
    if final_path.exists() and not rebuild:
        logger.info("Existing artifact found: %s", final_path)
        return BuildResult(
            output_dir=output_dir,
            final_onnx_path=final_path,
            config_path=config_path,
            reused=True,
            elapsed=time.monotonic() - start_time,
        )

    # Rebuild: clean old ONNX artifacts to prevent stale files
    if rebuild:
        for old in output_dir.glob("*.onnx"):
            old.unlink()
            logger.debug("Removed old artifact: %s", old.name)

    stages_completed: list[str] = []
    stages_skipped: list[str] = []
    stage_timings: dict[str, float] = {}

    # Copy input ONNX to output dir as starting point
    current_path = output_dir / onnx_path.name
    if current_path.resolve() != onnx_path.resolve():
        copy_onnx_model(onnx_path, current_path)

    # =========================================================================
    # [1]-[4] OPTIMIZE -> QUANTIZE -> COMPILE -> FINALIZE
    # Shared with build_hf_model via ``common.run_build_stages``.
    # =========================================================================
    skip_optimize: bool = kwargs.pop("skip_optimize", False)
    stages = run_build_stages(
        current_path=current_path,
        optimized_path=optimized_path,
        quantized_path=quantized_path,
        compiled_path=compiled_path,
        final_path=final_path,
        config=config,
        config_path=config_path,
        ep=ep,
        device=device,
        hack_max_optim_iterations=hack_max_optim_iterations,
        skip_optimize=skip_optimize,
        onnx_kwargs=onnx_kwargs,
    )
    stages_completed.extend(stages.stages_completed)
    stages_skipped.extend(stages.stages_skipped)
    stage_timings.update(stages.stage_timings)
    current_path = stages.current_path
    analyze_iters = stages.analyze_iterations
    analyze_unsupported = stages.analyze_unsupported_nodes
    analyze_details = stages.analyze_details
    quant_result = stages.quant_result

    elapsed = time.monotonic() - start_time
    logger.info("Build complete in %.1fs -> %s", elapsed, final_path)

    # =========================================================================
    # [5] BUILD MANIFEST — Machine-readable build provenance
    # =========================================================================
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": "onnx",
        "input_onnx": str(onnx_path),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "stages": [],
        "final_artifact": final_path.name,
        "analyze_iterations": analyze_iters,
        "analyze_unsupported_node_count": analyze_unsupported,
        "analyze_details": analyze_details,
    }

    stage_filenames = {
        "optimize": optimized_path.name,
        "quantize": quantized_path.name,
        "compile": compiled_path.name,
    }
    for stage_name in ["optimize", "quantize", "compile"]:
        if stage_name in stages_completed:
            entry: dict[str, Any] = {
                "name": stage_name,
                "status": "completed",
                "filename": stage_filenames[stage_name],
                "elapsed_seconds": round(stage_timings.get(stage_name, 0), 3),
            }
            # Thread QuantizeResult metrics into manifest
            if stage_name == "quantize" and quant_result is not None:
                entry["nodes_quantized"] = quant_result.nodes_quantized
                entry["nodes_skipped"] = quant_result.nodes_skipped
                entry["calibration_time_seconds"] = round(quant_result.calibration_time_seconds, 3)
                entry["qdq_insertion_time_seconds"] = round(
                    quant_result.qdq_insertion_time_seconds, 3
                )
            manifest["stages"].append(entry)
        elif stage_name in stages_skipped:
            manifest["stages"].append(
                {
                    "name": stage_name,
                    "status": "skipped",
                    "filename": None,
                    "elapsed_seconds": None,
                }
            )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.debug("Build manifest persisted: %s", manifest_path)

    return BuildResult(
        output_dir=output_dir,
        final_onnx_path=final_path,
        config_path=config_path,
        stages_completed=stages_completed,
        stages_skipped=stages_skipped,
        stage_timings=stage_timings,
        elapsed=elapsed,
        manifest_path=manifest_path,
    )
