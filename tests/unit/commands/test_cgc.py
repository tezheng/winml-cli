# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the hardware-free half of ``winml cgc adapters`` and ``cgc patterns``.

Every expectation here is derived from data this module builds, never from a
transcript: MLIR sources are assembled from a known number of patterns and
``any_of`` branches and the counts are computed from those parameters, and the
PE files are synthesized around a chosen ``D3D12SDKVersion`` value which is then
read back. The DXCore/D3D12 half needs real hardware and is not covered here.
"""

from __future__ import annotations

import ctypes
import json
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import pytest

from winml.modelkit.commands.cgc import (
    CGC_MAX_IR_VERSION,
    DUMP_FILES,
    MIN_SDK_VERSION,
    PATTERNS_JSON_FORMAT,
    REDIST_ENV_VARS,
    Adapter,
    FeatureDataMLIRExchange721,
    Options,
    RedistUnusable,
    _named_redist,  # pyright: ignore[reportPrivateUsage] - CLAUDE.md allows it in tests
    adapter_slug,
    atlas_page,
    close_adapters,
    dump_dir,
    format_driver_version,
    format_pattern_listing,
    host_machine,
    hrs,
    inspect_redist,
    list_adapters,
    normalize_text_payload,
    open_atlas,
    pattern_groups,
    patterns_document,
    pe_export_u32,
    print_adapters,
    probe_mlir_support,
    redist_candidates,
    redist_failure,
    release,
    report_patterns,
    resolve_redist,
    resolve_run,
    run_patterns,
    select_adapter,
    split_patterns,
    write_metadata,
)


#: The documented exit codes: 1 when the question could not be asked, 2 for bad
#: arguments.
EXIT_FAIL, EXIT_USAGE = 1, 2


if TYPE_CHECKING:
    from collections.abc import Callable


class ClosedStdoutError(OSError):
    """What writing to stdout raises once the reader has gone."""


def fake_device(*_args: object, **_kwargs: object) -> object:
    """Stand in for a D3D12 device, so report_patterns runs without one."""
    return object()


def no_release(_pointer: object) -> None:
    """Stand in for release(): there is no real COM pointer to let go of."""


def exchange_returning(payload: bytes) -> Callable[..., tuple[bytes, int]]:
    """Stand in for mlir_exchange: the driver answers with *payload*."""

    def exchange(*_args: object, **_kwargs: object) -> tuple[bytes, int]:
        return payload, 0

    return exchange


def count_patterns(data: bytes) -> tuple[int, int]:
    """Count patterns and rules the way the command reports them."""
    records = split_patterns(data)
    return len(records), sum(r["rules"] for r in records)


# --------------------------------------------------------------- MLIR builders

#: One marker per kind, taken from the classifier's own contract: a body holding
#: this token must be filed under that kind.
KIND_MARKERS = {
    "kernel": "jitFunction",
    "fusion": "subgraph_rewrite_desc",
    "hint": "cgc_add_pattern_cluster_hint",
    "mark": "cgc_mark_pattern_cluster_op",
    "rewrite": "some_other_op",
}


def make_pattern(name: str, kind: str, benefit: int | None = None, branches: int = 1) -> str:
    """Build one ``cgc_pattern.pattern`` whose kind and rule count are known.

    Args:
        name: Pattern name.
        kind: One of :data:`KIND_MARKERS`.
        benefit: Optional benefit value.
        branches: Number of ``all_of`` alternatives inside one ``any_of`` block;
            1 emits no ``any_of`` at all, so the pattern expands to a single rule.

    Returns:
        The pattern source.
    """
    head = f"cgc_pattern.pattern @{name}"
    if benefit is not None:
        head += f" benefit({benefit})"
    body = [f"  {KIND_MARKERS[kind]} : i32"]
    if branches > 1:
        alts = "\n".join(f"      all_of {{ op_{i} }}" for i in range(branches))
        body.append("  any_of {\n" + alts + "\n  }")
    return head + " {\n" + "\n".join(body) + "\n}\n"


# ----------------------------------------------------------------- PE builders

_MACHINE_FOR_ARCH = {"x86": 0x014C, "x64": 0x8664, "arm64": 0xAA64}


def make_pe(
    sdk_version: int | None, arch: str = "x64", export_name: bytes = b"D3D12SDKVersion"
) -> bytes:
    """Synthesize a minimal PE32+ exporting one UINT32 data symbol.

    The single section maps RVA to file offset one-to-one, so the export walk has
    a real table to traverse rather than a stub.

    Args:
        sdk_version: Value the export should carry, or None to emit no export
            directory at all.
        arch: Machine to stamp into the COFF header.
        export_name: Name to publish the value under.

    Returns:
        The PE image bytes.
    """
    pe_off, sec_rva = 0x80, 0x400
    buf = bytearray(0x1400)
    buf[0:2] = b"MZ"
    struct.pack_into("<I", buf, 0x3C, pe_off)
    buf[pe_off : pe_off + 4] = b"PE\0\0"
    struct.pack_into("<HH", buf, pe_off + 4, _MACHINE_FOR_ARCH[arch], 1)  # machine, nsec
    opt_size = 240
    struct.pack_into("<H", buf, pe_off + 20, opt_size)
    opt = pe_off + 24
    struct.pack_into("<H", buf, opt, 0x20B)  # PE32+

    # Section table: vaddr == rawptr keeps RVA and file offset identical.
    sec = opt + opt_size
    struct.pack_into("<IIII", buf, sec + 8, 0x1000, sec_rva, 0x1000, sec_rva)

    if sdk_version is None:
        return bytes(buf)  # data directory left zeroed -> no export table

    struct.pack_into("<I", buf, opt + 112, sec_rva)  # export dir RVA
    funcs, names, ords = sec_rva + 0x28, sec_rva + 0x2C, sec_rva + 0x30
    name_rva, value_rva = sec_rva + 0x40, sec_rva + 0x60
    struct.pack_into("<IIII", buf, sec_rva + 24, 1, funcs, names, ords)
    struct.pack_into("<I", buf, funcs, value_rva)
    struct.pack_into("<I", buf, names, name_rva)
    struct.pack_into("<H", buf, ords, 0)
    buf[name_rva : name_rva + len(export_name)] = export_name
    struct.pack_into("<I", buf, value_rva, sdk_version)
    return bytes(buf)


def write_redist(directory: Path, sdk_version: int | None, arch: str = "x64") -> Path:
    """Write a synthesized ``D3D12Core.dll`` into *directory*.

    Args:
        directory: Destination, created if absent.
        sdk_version: Value for the ``D3D12SDKVersion`` export.
        arch: Machine to stamp.

    Returns:
        The directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "D3D12Core.dll").write_bytes(make_pe(sdk_version, arch))
    return directory


# ------------------------------------------------------------------- adapters


def make_adapter(
    index: int, description: str, *, hardware: bool, integrated: bool = False
) -> Adapter:
    """Build an adapter record of the shape ``list_adapters`` returns.

    Args:
        index: Selection index.
        description: DXCore DriverDescription.
        hardware: Whether the adapter is hardware.
        integrated: Whether it is integrated.

    Returns:
        The record.
    """
    return {
        "index": index,
        "description": description,
        "driver_version": "1.2.3.4",
        "is_hardware": hardware,
        "is_integrated": integrated,
        "generic_ml": True,
        "core_compute": True,
        "d3d12_graphics": True,
        "mlir": None,
        "ir_version": 0,
    }


class TestAdapterSlug:
    """The dump directory name derived from an adapter description."""

    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("NVIDIA GeForce RTX 5090 D", "nvidia-geforce-rtx-5090-d"),
            ("AMD Radeon(TM) 880M Graphics", "amd-radeon-880m-graphics"),
            ("Intel(R) Graphics", "intel-graphics"),
            ("Microsoft Basic Render Driver", "microsoft-basic-render-driver"),
        ],
    )
    def test_known_descriptions(self, description: str, expected: str) -> None:
        assert adapter_slug(description) == expected

    def test_leading_and_trailing_separators_are_trimmed(self) -> None:
        assert adapter_slug("  Weird   Name!!! ") == "weird-name"

    def test_slug_is_filesystem_safe(self) -> None:
        slug = adapter_slug(r'A/B\C:D*E?F"G<H>I|J')
        assert not set(slug) & set('/\\:*?"<>|')


class TestDriverVersion:
    """The DXCore u64 rendered as four 16-bit parts."""

    @pytest.mark.parametrize(
        "parts",
        [(32, 0, 16, 3004), (0, 0, 0, 0), (0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF), (10, 0, 26100, 9278)],
    )
    def test_renders_each_16_bit_word_in_order(self, parts: tuple[int, int, int, int]) -> None:
        packed = sum(part << shift for part, shift in zip(parts, (48, 32, 16, 0), strict=True))
        assert format_driver_version(packed) == ".".join(str(p) for p in parts)

    def test_high_word_first(self) -> None:
        assert format_driver_version(1 << 48) == "1.0.0.0"


class TestHresultRendering:
    """HRESULTs print as hex, with a name when one is known."""

    def test_known_code_carries_its_name(self) -> None:
        assert hrs(0x80070057) == "0x80070057 (E_INVALIDARG)"

    def test_unknown_code_prints_raw(self) -> None:
        assert hrs(0x81234567) == "0x81234567"

    def test_negative_input_is_rendered_unsigned(self) -> None:
        assert hrs(-2147024809) == hrs(0x80070057)


class TestPatternParser:
    """Counting patterns and the rules they expand to."""

    def test_pattern_count_matches_what_was_built(self) -> None:
        built = 7
        src = "".join(make_pattern(f"p.n{i}", "kernel") for i in range(built))
        patterns, rules = count_patterns(src.encode())
        assert (patterns, rules) == (built, built)

    def test_any_of_branches_multiply_into_rules(self) -> None:
        branches = [1, 2, 3, 4]
        src = "".join(make_pattern(f"p.n{i}", "kernel", branches=b) for i, b in enumerate(branches))
        patterns, rules = count_patterns(src.encode())
        assert patterns == len(branches)
        assert rules == sum(branches)

    def test_comments_are_stripped_before_parsing(self) -> None:
        src = "// cgc_pattern.pattern @commented.out {\n" + make_pattern("p.real", "kernel")
        assert count_patterns(src.encode()) == (1, 1)

    def test_is_any_of_constraint_is_not_a_rule_multiplier(self) -> None:
        plain = make_pattern("p.plain", "kernel")
        decoyed = plain.replace(
            "  jitFunction : i32",
            '  jitFunction : i32\n  apply_native_constraint "cgc_is_any_of"',
        )
        assert count_patterns(decoyed.encode()) == count_patterns(plain.encode())

    def test_attributes_block_is_not_mistaken_for_the_body(self) -> None:
        src = (
            "cgc_pattern.pattern @p.deps attributes { depends = [@other] } {\n"
            "  jitFunction : i32\n}\n"
        )
        assert split_patterns(src.encode())[0]["kind"] == "kernel"

    def test_declaration_without_a_body_is_skipped(self) -> None:
        assert count_patterns(b"cgc_pattern.pattern @p.nobody") == (0, 0)

    @pytest.mark.parametrize("kind", sorted(KIND_MARKERS))
    def test_kind_is_read_from_the_body_not_the_name(self, kind: str) -> None:
        # Name every pattern after a *different* kind to prove the name is ignored.
        misleading = "kernel" if kind != "kernel" else "rewrite"
        src = make_pattern(f"vendor.{misleading}_looking_name", kind)
        assert split_patterns(src.encode())[0]["kind"] == kind

    def test_groups_are_ordered_and_totalled(self) -> None:
        counts = {"kernel": 3, "fusion": 2, "mark": 4}
        src = "".join(
            make_pattern(f"p.{kind}{i}", kind) for kind, n in counts.items() for i in range(n)
        )
        groups = pattern_groups(split_patterns(src.encode()))
        assert [g["kind"] for g in groups] == ["kernel", "fusion", "mark"]
        assert {g["kind"]: g["patterns"] for g in groups} == counts

    def test_empty_kinds_are_omitted(self) -> None:
        groups = pattern_groups(split_patterns(make_pattern("p.only", "hint").encode()))
        assert [g["kind"] for g in groups] == ["hint"]

    def test_members_sort_by_descending_benefit(self) -> None:
        src = "".join(make_pattern(f"p.b{b}", "kernel", benefit=b) for b in (10, 200, 36))
        members = pattern_groups(split_patterns(src.encode()))[0]["members"]
        assert [m["benefit"] for m in members] == [200, 36, 10]


class TestPeExportParser:
    """Reading D3D12SDKVersion without loading the DLL."""

    @pytest.mark.parametrize("version", [614, 616, 720, 721, 1])
    def test_reads_back_the_value_it_was_given(self, tmp_path: Path, version: int) -> None:
        dll = tmp_path / "D3D12Core.dll"
        dll.write_bytes(make_pe(version))
        assert pe_export_u32(dll, b"D3D12SDKVersion") == ("x64", version)

    def test_reports_machine_for_every_arch(self, tmp_path: Path) -> None:
        for arch in _MACHINE_FOR_ARCH:
            dll = tmp_path / f"{arch}.dll"
            dll.write_bytes(make_pe(720, arch))
            assert pe_export_u32(dll, b"D3D12SDKVersion")[0] == arch

    def test_missing_export_yields_no_value(self, tmp_path: Path) -> None:
        dll = tmp_path / "D3D12Core.dll"
        dll.write_bytes(make_pe(720, export_name=b"SomethingElse"))
        assert pe_export_u32(dll, b"D3D12SDKVersion") == ("x64", None)

    def test_absent_export_directory_yields_no_value(self, tmp_path: Path) -> None:
        dll = tmp_path / "D3D12Core.dll"
        dll.write_bytes(make_pe(None))
        assert pe_export_u32(dll, b"D3D12SDKVersion") == ("x64", None)

    def test_non_pe_file_is_rejected(self, tmp_path: Path) -> None:
        dll = tmp_path / "D3D12Core.dll"
        dll.write_bytes(b"not a PE at all" + bytes(64))
        assert pe_export_u32(dll, b"D3D12SDKVersion") == (None, None)


class TestInspectRedist:
    """Deciding whether a directory is a usable Agility SDK redist."""

    def test_accepts_a_core_at_the_floor(self, tmp_path: Path) -> None:
        version, reason = inspect_redist(write_redist(tmp_path / "ok", MIN_SDK_VERSION))
        assert version == MIN_SDK_VERSION
        assert reason == f"SDK {MIN_SDK_VERSION}"

    def test_accepts_a_newer_core(self, tmp_path: Path) -> None:
        assert inspect_redist(write_redist(tmp_path / "new", MIN_SDK_VERSION + 1))[0] is not None

    def test_rejects_a_core_below_the_floor(self, tmp_path: Path) -> None:
        version, reason = inspect_redist(write_redist(tmp_path / "old", MIN_SDK_VERSION - 1))
        assert version is None
        assert str(MIN_SDK_VERSION) in reason

    def test_rejects_a_missing_directory(self, tmp_path: Path) -> None:
        assert inspect_redist(tmp_path / "absent") == (None, "missing")

    def test_rejects_an_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert inspect_redist(empty) == (None, "empty")

    def test_rejects_a_directory_without_the_core(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        (other / "readme.txt").write_text("nothing here", encoding="utf-8")
        assert inspect_redist(other) == (None, "no D3D12Core.dll")

    def test_rejects_a_foreign_architecture(self, tmp_path: Path) -> None:
        foreign = next(a for a in _MACHINE_FOR_ARCH if a != host_machine())
        version, reason = inspect_redist(write_redist(tmp_path / "arch", 720, foreign))
        assert version is None
        assert foreign in reason


class TestRedistCandidates:
    """Which directories are searched, and in what order."""

    def test_explicit_flag_is_the_only_candidate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.setenv(var, r"C:\from-env")
        candidates = list(redist_candidates(r"C:\from-flag"))
        assert candidates == [("--d3d12-dir", Path(r"C:\from-flag"))]

    def test_first_environment_variable_set_wins_and_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.setenv(var, f"C:\\{var}")
        candidates = list(redist_candidates())
        assert candidates == [(f"${REDIST_ENV_VARS[0]}", Path(f"C:\\{REDIST_ENV_VARS[0]}"))]

    def test_later_variable_is_used_when_the_first_is_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(REDIST_ENV_VARS[0], raising=False)
        monkeypatch.setenv(REDIST_ENV_VARS[1], r"C:\second")
        assert next(iter(redist_candidates()))[0] == f"${REDIST_ENV_VARS[1]}"

    def test_bin_roots_are_probed_when_nothing_is_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        labels = [label for label, _ in redist_candidates()]
        assert labels, "expected implicit bin/ roots"
        assert all("bin" in label for label in labels)

    def test_each_root_is_probed_as_d3d12_then_bare_bin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        paths = [p for _, p in redist_candidates()]
        for d3d12 in (p for p in paths if p.name == "D3D12"):
            assert d3d12.parent in paths, f"bare bin/ missing for {d3d12}"

    def test_candidates_are_not_repeated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        paths = [p for _, p in redist_candidates()]
        assert len(paths) == len(set(paths))


class TestNamedRedist:
    """Whether the caller named a redist by hand, and by which source."""

    def test_flag_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert _named_redist(r"C:\x") == ("--d3d12-dir", Path(r"C:\x"))

    def test_flag_outranks_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REDIST_ENV_VARS[0], r"C:\env")
        assert _named_redist(r"C:\x") == ("--d3d12-dir", Path(r"C:\x"))

    def test_environment_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REDIST_ENV_VARS[0], r"C:\env")
        assert _named_redist() == (f"${REDIST_ENV_VARS[0]}", Path(r"C:\env"))

    def test_nothing_named_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        assert _named_redist() is None


class TestSelectAdapter:
    """Resolving ``-a`` to one adapter."""

    @pytest.fixture
    def adapters(self) -> list[Adapter]:
        return [
            make_adapter(0, "Microsoft Basic Render Driver", hardware=False),
            make_adapter(1, "NVIDIA GeForce RTX 5090 D", hardware=True),
            make_adapter(2, "AMD Radeon(TM) 880M Graphics", hardware=True, integrated=True),
        ]

    def test_default_prefers_the_first_hardware_adapter(self, adapters: list[Adapter]) -> None:
        assert select_adapter(adapters, "")["index"] == 1

    def test_default_falls_back_to_index_zero(self) -> None:
        software = [make_adapter(0, "Software", hardware=False)]
        assert select_adapter(software, "")["index"] == 0

    def test_bare_index_selects_positionally(self, adapters: list[Adapter]) -> None:
        assert select_adapter(adapters, "2")["index"] == 2

    def test_substring_is_case_insensitive(self, adapters: list[Adapter]) -> None:
        assert select_adapter(adapters, "nvidia")["index"] == 1

    def test_a_number_that_is_no_index_is_matched_as_a_substring(
        self, adapters: list[Adapter]
    ) -> None:
        # "5090" is a model number, not an index, on a three-adapter machine.
        assert select_adapter(adapters, "5090")["index"] == 1

    def test_an_index_wins_over_a_substring(self, adapters: list[Adapter]) -> None:
        # A number that names a real index stays positional, and is never retried
        # as a substring.
        assert select_adapter(adapters, "2")["index"] == 2

    def test_out_of_range_index_is_a_usage_error(self, adapters: list[Adapter]) -> None:
        with pytest.raises(click.UsageError) as excinfo:
            select_adapter(adapters, "99")
        assert excinfo.value.exit_code == EXIT_USAGE

    def test_unmatched_substring_is_a_usage_error(self, adapters: list[Adapter]) -> None:
        with pytest.raises(click.UsageError) as excinfo:
            select_adapter(adapters, "nosuchvendor")
        assert excinfo.value.exit_code == EXIT_USAGE


class TestFlagRules:
    """How the flags constrain one another."""

    def test_dump_without_an_adapter_is_a_usage_error(self) -> None:
        # Refused before DXCore is touched, so this needs no hardware.
        with pytest.raises(click.UsageError) as excinfo:
            run_patterns(Options(dump=True))
        assert excinfo.value.exit_code == EXIT_USAGE

    def test_click_exceptions_carry_the_documented_codes(self) -> None:
        # The command raises these rather than exiting itself, so the documented
        # contract -- 1 could not ask, 2 bad arguments -- rests on Click's values.
        assert click.UsageError("x").exit_code == EXIT_USAGE
        assert click.ClickException("x").exit_code == EXIT_FAIL


class TestWriteMetadata:
    """The provenance file written beside a dump."""

    @staticmethod
    def read_keys(path: Path) -> dict[str, str]:
        text = path.read_text(encoding="utf-8")
        return dict(line.split("=", 1) for line in text.splitlines())

    def test_records_counts_for_a_successful_dump(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.txt"
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        write_metadata(target, adapter, "dumped", "text", 720, Path(r"C:\redist"), (51, 63))
        keys = self.read_keys(target)
        assert keys["status"] == "dumped"
        assert keys["patterns"] == "51"
        assert keys["rules"] == "63"
        assert keys["text_file"] == "patterns.mlir"
        assert keys["bytecode_file"] == ""
        assert keys["driver_version"] == adapter["driver_version"]

    def test_leaves_counts_empty_when_unavailable(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.txt"
        adapter = make_adapter(0, "AMD Radeon(TM) 880M Graphics", hardware=True)
        write_metadata(target, adapter, "unsupported", "none", 720, Path(r"C:\redist"), None)
        keys = self.read_keys(target)
        assert keys["status"] == "unsupported"
        assert keys["patterns"] == ""
        assert keys["rules"] == ""
        assert keys["text_file"] == ""

    def test_bytecode_sets_only_the_bytecode_file(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.txt"
        adapter = make_adapter(0, "Vendor Device", hardware=True)
        write_metadata(target, adapter, "dumped", "bytecode", 721, Path(r"C:\r"), None)
        keys = self.read_keys(target)
        assert keys["bytecode_file"] == "patterns.mlirbc"
        assert keys["text_file"] == ""

    def test_first_six_keys_keep_their_documented_order(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.txt"
        write_metadata(
            target,
            make_adapter(0, "Vendor Device", hardware=True),
            "dumped",
            "text",
            720,
            Path(r"C:\r"),
            (1, 1),
        )
        order = [line.split("=", 1)[0] for line in target.read_text(encoding="utf-8").splitlines()]
        assert order[:6] == [
            "adapter",
            "driver_version",
            "status",
            "received_encoding",
            "text_file",
            "bytecode_file",
        ]

    def test_written_with_newline_endings(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.txt"
        write_metadata(
            target,
            make_adapter(0, "Vendor Device", hardware=True),
            "dumped",
            "text",
            720,
            Path(r"C:\r"),
            (1, 1),
        )
        assert b"\r\n" not in target.read_bytes()


# ---------------------------------------------------------------------------
# Cases ported from the original standalone suite (DXML/python/test_cgc.py).
# The MLIR fixtures below are shaped like real driver output rather than the
# minimal sources built above: bodies nest a `rewrite %x { ... }` block, a
# kernel carries its jitFunction *inside* a subgraph_rewrite_desc (so it also
# matches the fusion marker), and one pattern puts a `depends` attribute
# dictionary between the name and the body.
# ---------------------------------------------------------------------------

KERNEL = (
    b"cgc_pattern.pattern @nvidia.gemm_out : benefit(35) {\n"
    b"  rewrite %gemm {\n"
    b"    %d = attribute = #cgc_subgraph_pattern.subgraph_rewrite_desc<\n"
    b'        jitFunction = "nvidia_gemm_output_major">\n'
    b"  }\n}\n"
)
FUSION = (
    b"cgc_pattern.pattern @nvidia.expr_native_fusion : benefit(30) {\n"
    b'  %c = operation "cgc_subgraph_pattern.pattern_cluster"\n'
    b"  rewrite %c {\n"
    b"    %d = attribute = #cgc_subgraph_pattern.subgraph_rewrite_desc<\n"
    b"        cluster_fusion = true>\n"
    b"  }\n}\n"
)
HINT = (
    b"cgc_pattern.pattern @nvidia.conv_prologue_hint : benefit(50) {\n"
    b"  rewrite %conv {\n"
    b'    apply_native_rewrite "cgc_add_pattern_cluster_hint" (%h, %conv)\n'
    b"  }\n}\n"
)
MARK = (
    b"cgc_pattern.pattern @nvidia.slice_prologue_mark : benefit(0) {\n"
    b"  rewrite %slice {\n"
    b'    apply_native_rewrite "cgc_mark_pattern_cluster_op" (%slice, %cat)\n'
    b"  }\n}\n"
)
REWRITE = (
    b"cgc_pattern.pattern @nvidia.fold_transpose : benefit(1) {\n"
    b"  rewrite %t {\n"
    b"    replace %t with %x\n"
    b"  }\n}\n"
)
WITH_ATTRS = (
    b"cgc_pattern.pattern @nvidia.gemm_epilogue_cluster\n"
    b"        : benefit(38) attributes {\n"
    b"            depends = [@nvidia.gemm_expression_epilogue_cluster_hint]\n"
    b"        } {\n"
    b"  rewrite %gemm {\n"
    b"    %d = attribute = #cgc_subgraph_pattern.subgraph_rewrite_desc<\n"
    b'        jitFunction = "nvidia_gemm_epilogue">\n'
    b"  }\n}\n"
)
BRANCHING = (
    b"cgc_pattern.pattern @nvidia.rank_agnostic : benefit(36) {\n"
    b"  any_of {\n"
    b"    all_of { %a = operand }\n"
    b"    all_of { %b = operand }\n"
    b"    all_of { %c = operand }\n"
    b"  }\n"
    b"  rewrite %r {\n"
    b'    apply_native_rewrite "cgc_mark_pattern_cluster_op" (%r, %cat)\n'
    b"  }\n}\n"
)

ALL_FIXTURES = KERNEL + FUSION + HINT + MARK + REWRITE + WITH_ATTRS + BRANCHING


class TestPatternKindsOnRealisticSources:
    """The five kinds, read off bodies shaped like real driver output."""

    @pytest.mark.parametrize(
        ("source", "kind"),
        [
            (KERNEL, "kernel"),
            (FUSION, "fusion"),
            (HINT, "hint"),
            (MARK, "mark"),
            (REWRITE, "rewrite"),
        ],
    )
    def test_each_kind_is_recognised(self, source: bytes, kind: str) -> None:
        assert split_patterns(source)[0]["kind"] == kind

    def test_a_kernel_is_not_mistaken_for_a_fusion(self) -> None:
        # Both carry a subgraph_rewrite_desc; only a kernel names a jitFunction,
        # so marker precedence -- not mere presence -- decides the kind.
        assert b"subgraph_rewrite_desc" in KERNEL
        assert split_patterns(KERNEL)[0]["kind"] == "kernel"
        assert split_patterns(FUSION)[0]["kind"] == "fusion"

    def test_name_and_benefit_are_parsed(self) -> None:
        record = split_patterns(KERNEL)[0]
        assert record["name"] == "nvidia.gemm_out"
        assert record["benefit"] == 35

    def test_a_depends_list_before_the_body_is_not_the_body(self) -> None:
        # Taking the first { after the name reads the attribute dictionary as the
        # body, finds no jitFunction in it, and files a kernel under rewrite.
        record = split_patterns(WITH_ATTRS)[0]
        assert record["kind"] == "kernel"
        assert record["benefit"] == 38

    def test_rules_follow_the_any_of_expansion(self) -> None:
        assert split_patterns(BRANCHING)[0]["rules"] == 3
        assert split_patterns(MARK)[0]["rules"] == 1

    def test_two_any_of_blocks_in_one_pattern_multiply(self) -> None:
        block = b"  any_of {\n    all_of { a }\n    all_of { b }\n  }\n"
        src = b"cgc_pattern.pattern @a : benefit(0) {\n" + block + block + b"}\n"
        assert count_patterns(src) == (1, 4)

    def test_empty_input(self) -> None:
        assert count_patterns(b"") == (0, 0)


class TestPatternGroupingAndListing:
    """Bucketing by kind and the -v listing built from it."""

    def test_groups_run_in_the_fixed_kind_order(self) -> None:
        groups = pattern_groups(split_patterns(ALL_FIXTURES))
        assert [g["kind"] for g in groups] == ["kernel", "fusion", "hint", "mark", "rewrite"]

    def test_group_carries_its_pattern_and_rule_totals(self) -> None:
        groups = pattern_groups(split_patterns(MARK + BRANCHING))
        assert (groups[0]["patterns"], groups[0]["rules"]) == (2, 4)

    def test_patterns_are_listed_by_descending_benefit(self) -> None:
        groups = pattern_groups(split_patterns(KERNEL + WITH_ATTRS))
        assert [r["benefit"] for r in groups[0]["members"]] == [38, 35]

    def test_listing_names_every_pattern_exactly_once(self) -> None:
        records = split_patterns(ALL_FIXTURES)
        text = "\n".join(format_pattern_listing(records))
        for record in records:
            assert text.count(record["name"].split(".")[-1]) == 1

    def test_listing_marks_rule_counts_only_where_they_differ(self) -> None:
        text = "\n".join(format_pattern_listing(split_patterns(BRANCHING + MARK)))
        assert "3 rules" in text
        assert "1 rules" not in text


class TestAdapterTable:
    """The rendered listing: one row per adapter."""

    @staticmethod
    def column(capsys: pytest.CaptureFixture[str], name: str, **overrides: object) -> str:
        """Print one adapter, and read one column of its row by the header's layout."""
        row = make_adapter(0, "Some GPU", hardware=True)
        row.update(overrides)
        print_adapters([row])
        header, _, line = capsys.readouterr().out.splitlines()
        names = header.split()
        start = header.index(name)
        after = names.index(name) + 1
        end = header.index(names[after]) if after < len(names) else None
        return line[start:end].strip()

    @pytest.mark.parametrize(("mlir", "mark"), [(True, "yes"), (False, "no"), (None, "?")])
    def test_support_marks(
        self, capsys: pytest.CaptureFixture[str], mlir: bool | None, mark: str
    ) -> None:
        assert self.column(capsys, "MLIR", mlir=mlir) == mark

    def test_header_names_every_column(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_adapters([])
        header = capsys.readouterr().out.splitlines()[0]
        assert header.split() == ["IDX", "ADAPTER", "DRIVER", "TYPE", "ATTRIBUTES", "MLIR"]

    def test_a_row_is_written_per_adapter(self, capsys: pytest.CaptureFixture[str]) -> None:
        rows = [make_adapter(i, f"GPU {i}", hardware=True) for i in range(3)]
        print_adapters(rows)
        # header, rule, then one line per adapter
        assert len(capsys.readouterr().out.splitlines()) == 2 + len(rows)

    @pytest.mark.parametrize(
        ("hardware", "integrated", "kind"),
        [
            (True, True, "integrated"),
            (True, False, "hardware"),
            (False, True, "software"),  # software wins over integrated
        ],
    )
    def test_type(
        self, capsys: pytest.CaptureFixture[str], hardware: bool, integrated: bool, kind: str
    ) -> None:
        column = self.column(capsys, "TYPE", is_hardware=hardware, is_integrated=integrated)
        assert column == kind

    def test_only_advertised_attributes_are_listed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert self.column(capsys, "ATTRIBUTES", core_compute=False) == "ML GFX"

    def test_no_attributes_renders_a_dash(self, capsys: pytest.CaptureFixture[str]) -> None:
        bare = {"generic_ml": False, "core_compute": False, "d3d12_graphics": False}
        assert self.column(capsys, "ATTRIBUTES", **bare) == "-"


class TestDumpWrites:
    """What --dump leaves on disk."""

    MODULE = "winml.modelkit.commands.cgc"

    @pytest.fixture
    def adapter(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """An adapter whose driver answers with KERNEL, in a fresh working directory."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(f"{self.MODULE}.probe_mlir_support", fake_device)
        monkeypatch.setattr(f"{self.MODULE}.mlir_exchange", exchange_returning(KERNEL))
        monkeypatch.setattr(f"{self.MODULE}.release", no_release)
        return make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)

    def test_a_dump_is_complete_even_when_stdout_fails(
        self, adapter: Adapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A reader that stops early (`| head -1`) makes every later write to stdout
        # fail. --overwrite has cleared the previous dump by then, so every file of the
        # new one must be on disk before the first line of the answer is printed.
        dest = dump_dir(adapter)
        dest.mkdir(parents=True)
        for name in DUMP_FILES:
            (dest / name).write_bytes(b"previous dump")

        def closed_stdout(*_args: object, **_kwargs: object) -> None:
            raise ClosedStdoutError(22, "Invalid argument")  # closed before the first line

        monkeypatch.setattr("click.echo", closed_stdout)
        options = Options(adapter="nvidia", dump=True, overwrite=True)

        with pytest.raises(ClosedStdoutError):
            report_patterns(options, adapter, Path("redist"), 720)

        assert sorted(p.name for p in dest.iterdir()) == [
            "metadata.txt",
            "patterns.json",
            "patterns.mlir",
        ]
        assert (dest / "patterns.mlir").read_bytes() == normalize_text_payload(KERNEL)

    def test_the_answer_is_printed_whole_and_in_order(
        self, adapter: Adapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Printing after the writes must not change what is printed, or its order.
        printed: list[str] = []

        def capture(message: object = "", **_kwargs: object) -> None:
            printed.append(str(message))

        monkeypatch.setattr("click.echo", capture)

        report_patterns(Options(adapter="nvidia", dump=True), adapter, Path("redist"), 720)

        patterns, rules = count_patterns(KERNEL)
        assert printed[0] == f"{adapter['description']}: {patterns} patterns / {rules} rules"
        assert [Path(line.split()[1]).name for line in printed[1:]] == [
            "patterns.mlir",
            "patterns.json",
            "metadata.txt",
        ]

    def test_patterns_json_round_trips_with_lf_endings(self, adapter: Adapter) -> None:
        report_patterns(Options(adapter="nvidia", dump=True), adapter, Path("redist"), 720)

        raw = (dump_dir(adapter) / "patterns.json").read_bytes()
        assert b"\r\n" not in raw
        expected = patterns_document(normalize_text_payload(KERNEL), adapter, 720, Path("redist"))
        assert json.loads(raw) == expected

    def test_overwrite_replaces_the_dump_and_keeps_other_files(self, adapter: Adapter) -> None:
        # Replaced, not merged: a stale file of any dump kind must not survive beside
        # the new one, and a file cgc did not write is left alone.
        dest = dump_dir(adapter)
        dest.mkdir(parents=True)
        for name in DUMP_FILES:
            (dest / name).write_bytes(b"stale")
        (dest / "unrelated.txt").write_text("keep me", encoding="utf-8")

        options = Options(adapter="nvidia", dump=True, overwrite=True)
        report_patterns(options, adapter, Path("redist"), 720)

        assert not (dest / "patterns.mlirbc").exists()
        assert (dest / "patterns.mlir").read_bytes() == normalize_text_payload(KERNEL)
        assert (dest / "unrelated.txt").read_text(encoding="utf-8") == "keep me"

    def test_a_dump_that_cannot_be_written_is_a_clean_error(
        self, adapter: Adapter, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A file where the adapter's directory should be: the dump cannot be written,
        # and that is reported like any other failure rather than as a traceback. The
        # driver's answer is still printed.
        blocker = dump_dir(adapter).parent
        blocker.parent.mkdir(parents=True)
        blocker.write_text("not a directory", encoding="utf-8")

        with pytest.raises(click.ClickException, match="could not write the dump"):
            report_patterns(Options(adapter="nvidia", dump=True), adapter, Path("redist"), 720)

        patterns, rules = count_patterns(KERNEL)
        assert capsys.readouterr().out.splitlines() == [
            f"{adapter['description']}: {patterns} patterns / {rules} rules"
        ]

    def test_a_bytecode_answer_is_stored_as_is(
        self, adapter: Adapter, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bytecode is kept byte for byte and never split into patterns.json.
        payload = b"ML\xefR" + bytes(range(8))  # the MLIR bytecode magic, then a body
        monkeypatch.setattr(f"{self.MODULE}.mlir_exchange", exchange_returning(payload))

        report_patterns(Options(adapter="nvidia", dump=True), adapter, Path("redist"), 720)

        dest = dump_dir(adapter)
        assert (dest / "patterns.mlirbc").read_bytes() == payload
        assert not (dest / "patterns.json").exists()


class TestDumpLayout:
    """Where a dump is filed."""

    def test_slug_and_version_directories(self) -> None:
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        adapter["driver_version"] = "32.0.16.3004"
        assert dump_dir(adapter) == Path("patterns") / "nvidia-geforce-rtx-5090-d" / "32.0.16.3004"

    def test_an_unknown_driver_version_keeps_the_layout(self) -> None:
        # An empty segment would be joined away, and the dump would lose the level
        # that keeps one driver version's capture apart from another's.
        adapter = make_adapter(0, "Some GPU", hardware=True)
        adapter["driver_version"] = ""
        assert dump_dir(adapter) == Path("patterns") / "some-gpu" / "unknown-version"

    def test_the_root_is_always_patterns(self) -> None:
        adapter = make_adapter(0, "Intel(R) Graphics", hardware=True)
        adapter["driver_version"] = "32.0.101.6127"
        assert dump_dir(adapter).parts[0] == "patterns"

    def test_two_vendors_never_collide(self) -> None:
        first = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        second = make_adapter(1, "Intel(R) Graphics", hardware=True)
        assert dump_dir(first) != dump_dir(second)


class TestRedistLine:
    """The redist is reported on every run that resolves one."""

    def test_names_the_directory_the_sdk_and_the_machine(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        redist = write_redist(tmp_path / "D3D12", MIN_SDK_VERSION, host_machine())

        resolve_run(Options(d3d12_dir=str(redist)))

        expected = f"redist: {redist} (SDK {MIN_SDK_VERSION}, {host_machine()})"
        assert capsys.readouterr().out.splitlines() == [expected]

    def test_says_none_when_there_is_no_redist(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in REDIST_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr("winml.modelkit.commands.cgc._redist_roots", list)

        resolve_run(Options())

        line = capsys.readouterr().out
        assert line.startswith("redist: none")
        assert "-v" in line


class TestRedistFailureReport:
    """Silence about a rejected candidate costs a day."""

    def test_names_every_candidate_and_refuses_system32(self) -> None:
        _, _, tried = resolve_redist(r"C:\definitely-not-here")
        text = redist_failure(tried)
        assert r"C:\definitely-not-here" in text
        assert "missing" in text
        assert "System32" in text
        assert "does not fall back" in text

    def test_states_the_minimum_sdk(self) -> None:
        _, _, tried = resolve_redist(r"C:\definitely-not-here")
        assert str(MIN_SDK_VERSION) in redist_failure(tried)

    def test_offers_a_way_out(self) -> None:
        _, _, tried = resolve_redist(r"C:\definitely-not-here")
        text = redist_failure(tried)
        assert "--d3d12-dir" in text
        assert REDIST_ENV_VARS[0] in text


class TestExchangeStructs:
    """The wire layout the driver reads by size."""

    def test_721_exchange_struct_is_48_bytes(self) -> None:
        # 4 bytes of padding sit between Type and IRVersion. A layout that drifts
        # is silently wrong rather than loudly wrong.
        assert ctypes.sizeof(FeatureDataMLIRExchange721) == 48

    def test_the_ir_version_bound_is_cgc_0_7(self) -> None:
        # cmake/version.cmake CGC_VERSION, packed as a D3D12_VERSION_NUMBER.
        assert format_driver_version(CGC_MAX_IR_VERSION) == "0.7.0.0"


class TestMetadataConsumerCompatibility:
    """Downstream readers parse this file; the shape is a contract."""

    def test_driver_version_is_findable_by_regex(self, tmp_path: Path) -> None:
        # A consumer finds the driver version with exactly this regex. If it stops
        # matching, that consumer silently reports a blank driver version.
        target = tmp_path / "metadata.txt"
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        adapter["driver_version"] = "32.0.16.3004"
        write_metadata(target, adapter, "dumped", "text", 720, Path(r"C:\r"), (51, 63))
        match = re.search(r"driver_version=(\S+)", target.read_text(encoding="utf-8"))
        assert match is not None
        assert match.group(1) == "32.0.16.3004"

    def test_every_key_is_present_even_when_empty(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.txt"
        write_metadata(
            target,
            make_adapter(0, "Vendor Device", hardware=True),
            "unsupported",
            "none",
            None,
            None,
            None,
        )
        keys = [line.split("=", 1)[0] for line in target.read_text(encoding="utf-8").splitlines()]
        assert len(keys) == len(set(keys)) == 13


# --------------------------------------------------------------------- hardware


class TestCapabilityOnRealHardware:
    """Every adapter creates a device; only some drivers answer the exchange."""

    def test_every_adapter_answers_or_declines(self) -> None:
        redist, sdk, _ = resolve_redist()
        if redist is None or sdk is None:
            pytest.skip("no Agility SDK redist on this machine (see docs/commands/cgc.md)")
        try:
            adapters = list_adapters()
        except click.ClickException as exc:  # DXCore unavailable
            pytest.skip(f"DXCore unavailable: {exc}")
        else:
            if not adapters:
                pytest.skip("no D3D12 adapters")
            try:
                for adapter in adapters:
                    try:
                        release(probe_mlir_support(adapter, redist, sdk))
                    except RedistUnusable as exc:  # an environment limit, not a code defect
                        pytest.skip(
                            f"D3D12 cannot use the redist found here: {exc.format_message()}"
                        )
                    assert adapter["mlir"] in (True, False)
            finally:
                close_adapters(adapters)


# ---------------------------------------------------------------------------
# patterns.json: splitting a dump into one record per pattern.
#
# These fixtures use the syntax real driver dumps actually contain, which differs
# from the minimal sources above in three ways that matter here: the kernel is
# written as #cgc.string<"Name">, source files are separated by "// from" markers,
# and prose comments can also begin with "// from".
# ---------------------------------------------------------------------------

REAL_KERNEL = (
    b"cgc_pattern.pattern @nvidia.gemm_cluster : benefit(38) {\n"
    b"  rewrite %gemm {\n"
    b"    %d = attribute = #cgc_subgraph_pattern.subgraph_rewrite_desc<\n"
    b'        foreign_config = {jitFunction = #cgc.string<"GemmCluster">}>\n'
    b"  }\n}\n"
)


def merged_dump(*sections: tuple[str, bytes], trailer: bytes = b"") -> bytes:
    """Assemble a merged dump the way the driver's does: a manifest, then sections.

    Args:
        *sections: ``(source file, pattern sources)`` pairs, in order.
        trailer: Bytes appended after the module, such as a NUL terminator.

    Returns:
        The dump.
    """
    out = [b"// Merged pattern set - do not edit.\n", b"module {\n"]
    for name, body in sections:
        out.append(b"    // from " + name.encode() + b"\n")
        out.append(body)
    out.append(b"}\n")
    return b"".join(out) + trailer


class TestSplitPatterns:
    """One record per pattern, with the facts a reader needs beyond the counts."""

    def test_one_record_per_pattern_in_declaration_order(self) -> None:
        records = split_patterns(ALL_FIXTURES)
        assert [r["index"] for r in records] == list(range(1, len(records) + 1))

    def test_each_slice_is_exactly_its_own_pattern(self) -> None:
        for record in split_patterns(ALL_FIXTURES):
            text = record["mlir"]
            assert text.startswith(f"cgc_pattern.pattern @{record['name']}")
            assert text.rstrip().endswith("}")
            assert text.count("cgc_pattern.pattern @") == 1
            assert text.count("{") == text.count("}")

    def test_short_name_drops_the_vendor_prefix(self) -> None:
        assert split_patterns(KERNEL)[0]["short_name"] == "gemm_out"

    def test_line_count_matches_the_slice(self) -> None:
        record = split_patterns(REAL_KERNEL)[0]
        assert record["lines"] == record["mlir"].count("\n") + 1

    def test_kernel_name_is_read_from_the_real_syntax(self) -> None:
        assert split_patterns(REAL_KERNEL)[0]["kernel"] == "GemmCluster"

    def test_kernel_name_accepts_the_plain_string_form(self) -> None:
        assert split_patterns(KERNEL)[0]["kernel"] == "nvidia_gemm_output_major"

    def test_non_kernels_declare_no_kernel(self) -> None:
        for source in (FUSION, HINT, MARK, REWRITE):
            assert split_patterns(source)[0]["kernel"] is None

    def test_a_kernel_named_only_in_a_comment_is_not_a_kernel(self) -> None:
        src = MARK.replace(
            b"  rewrite %slice {", b'  // jitFunction = #cgc.string<"Ghost">\n  rewrite %slice {'
        )
        assert split_patterns(src)[0]["kernel"] is None

    def test_patterns_are_attributed_to_their_source_file(self) -> None:
        dump = merged_dump(("a.pdll", KERNEL + MARK), ("b.mlir", HINT))
        assert [r["source"] for r in split_patterns(dump)] == ["a.pdll", "a.pdll", "b.mlir"]

    def test_a_prose_from_comment_does_not_open_a_section(self) -> None:
        prose = b"  // from outside the root's own chain. Both hints create clusters\n"
        dump = merged_dump(("a.pdll", KERNEL + prose + MARK))
        assert {r["source"] for r in split_patterns(dump)} == {"a.pdll"}

    def test_a_pattern_before_any_section_is_unattributed(self) -> None:
        assert split_patterns(KERNEL)[0]["source"] is None

    def test_the_trailing_nul_is_dropped_before_splitting(self) -> None:
        # The driver reports its payload size including the C string terminator.
        dump = normalize_text_payload(merged_dump(("a.pdll", KERNEL), trailer=b"\x00"))
        assert not dump.endswith(b"\x00")
        records = split_patterns(dump)
        assert len(records) == 1
        assert "\x00" not in records[0]["mlir"]

    def test_a_declaration_without_a_body_does_not_adopt_the_next_one(self) -> None:
        # The body search stops at the next declaration, so a body-less one is skipped
        # rather than swallowing the pattern that follows it.
        source = b"cgc_pattern.pattern @p.nobody\n" + KERNEL
        records = split_patterns(source)
        assert [r["name"] for r in records] == [r["name"] for r in split_patterns(KERNEL)]
        assert records[0]["mlir"].count("cgc_pattern.pattern @") == 1

    def test_a_comment_marker_inside_a_string_is_not_a_comment(self) -> None:
        # Blanking from "//" to end of line would take the closing brace with it, and
        # the pattern would swallow the next one.
        source = (
            b"cgc_pattern.pattern @p.one : benefit(1) {\n"
            b'  jitFunction = "http://example/kernel" }\n'
        ) + KERNEL
        records = split_patterns(source)
        assert [r["short_name"] for r in records] == [
            "one",
            split_patterns(KERNEL)[0]["short_name"],
        ]
        assert records[0]["lines"] == 2

    def test_a_brace_inside_a_string_does_not_end_the_pattern(self) -> None:
        # A quoted "}" -- a GUID, a snippet in a note -- must not close the body.
        source = (
            b"cgc_pattern.pattern @p.quoted : benefit(1) {\n"
            b'  note = "a } brace in a string"\n'
            b"  cgc_mark_pattern_cluster_op\n}\n"
        )
        record = split_patterns(source)[0]
        assert record["kind"] == "mark"
        assert record["mlir"] == source.decode().rstrip("\n")

    def test_a_declaration_inside_a_string_is_not_a_pattern(self) -> None:
        source = (
            b"cgc_pattern.pattern @p.real : benefit(1) {\n"
            b'  note = "cgc_pattern.pattern @p.ghost"\n'
            b"  any_of { all_of { a } all_of { b } }\n"
            b"  cgc_mark_pattern_cluster_op\n}\n"
        )
        assert [r["short_name"] for r in split_patterns(source)] == ["real"]

    def test_any_of_inside_a_string_does_not_multiply_the_rules(self) -> None:
        source = (
            b"cgc_pattern.pattern @p.counted : benefit(1) {\n"
            b'  note = "any_of { all_of { x } all_of { y } }"\n'
            b"  cgc_mark_pattern_cluster_op\n}\n"
        )
        assert split_patterns(source)[0]["rules"] == 1

    def test_a_brace_in_a_kernel_name_is_read_but_not_counted(self) -> None:
        # The kernel is read out of its string; the brace inside it is not structure,
        # so it neither ends the pattern nor swallows the next.
        name = "Gemm{Cluster}"
        source = REAL_KERNEL.replace(b"GemmCluster", name.encode()) + MARK
        records = split_patterns(source)
        assert [r["kernel"] for r in records] == [name, None]
        assert records[0]["mlir"].count("cgc_pattern.pattern @") == 1

    def test_comments_survive_in_the_displayed_text(self) -> None:
        src = KERNEL.replace(b"  rewrite %gemm {", b"  // keep me\n  rewrite %gemm {")
        assert "// keep me" in split_patterns(src)[0]["mlir"]

    def test_a_brace_inside_a_comment_cannot_unbalance_the_slice(self) -> None:
        src = KERNEL.replace(b"  rewrite %gemm {", b"  // stray } brace\n  rewrite %gemm {")
        record = split_patterns(src + MARK)[0]
        assert record["name"] == "nvidia.gemm_out"
        assert record["mlir"].count("cgc_pattern.pattern @") == 1
        assert record["kind"] == "kernel"


class TestParserEquivalence:
    """Blanking comments must count exactly what deleting them did."""

    @staticmethod
    def deleting_parser(data: bytes) -> tuple[int, int]:
        """The parser's previous behaviour: delete comments, then count."""
        return count_patterns(re.sub(rb"//[^\n]*", b"", data))

    @pytest.mark.parametrize(
        "source",
        [ALL_FIXTURES, KERNEL + MARK, BRANCHING, WITH_ATTRS, REAL_KERNEL],
        ids=["all", "kernel+mark", "branching", "depends", "real-kernel"],
    )
    def test_counts_are_unchanged(self, source: bytes) -> None:
        assert count_patterns(source) == self.deleting_parser(source)

    def test_counts_are_unchanged_with_comments_everywhere(self) -> None:
        noisy = b"\n".join(line + b"  // trailing note" for line in ALL_FIXTURES.split(b"\n"))
        assert count_patterns(noisy) == self.deleting_parser(noisy)


class TestPatternsDocument:
    """The patterns.json document written beside a dump."""

    @staticmethod
    def document(data: bytes) -> dict[str, Any]:
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        return patterns_document(data, adapter, 720, Path(r"C:\redist"))

    def test_carries_its_format_tag(self) -> None:
        assert self.document(ALL_FIXTURES)["format"] == PATTERNS_JSON_FORMAT

    def test_meta_records_provenance(self) -> None:
        meta = self.document(ALL_FIXTURES)["meta"]
        assert meta["adapter"] == "NVIDIA GeForce RTX 5090 D"
        assert meta["sdk_version"] == 720
        assert meta["bytes"] == len(ALL_FIXTURES)

    def test_summary_agrees_with_the_patterns_it_carries(self) -> None:
        doc = self.document(ALL_FIXTURES)
        pats, summary = doc["patterns"], doc["summary"]
        assert summary["patterns"] == len(pats)
        assert summary["rules"] == sum(p["rules"] for p in pats)
        assert summary["kernel_patterns"] == sum(1 for p in pats if p["kernel"])
        assert summary["top_benefit"] == max(p["benefit"] or 0 for p in pats)

    def test_summary_agrees_with_count_patterns(self) -> None:
        summary = self.document(ALL_FIXTURES)["summary"]
        assert (summary["patterns"], summary["rules"]) == count_patterns(ALL_FIXTURES)

    def test_every_pattern_is_in_exactly_one_source_group(self) -> None:
        doc = self.document(merged_dump(("a.pdll", KERNEL + MARK), ("b.mlir", HINT)))
        grouped = [i for s in doc["sources"] for i in s["patterns"]]
        assert sorted(grouped) == [p["index"] for p in doc["patterns"]]
        assert [s["name"] for s in doc["sources"]] == ["a.pdll", "b.mlir"]

    def test_kind_totals_sum_to_the_pattern_count(self) -> None:
        doc = self.document(ALL_FIXTURES)
        assert sum(doc["kinds"].values()) == doc["summary"]["patterns"]

    def test_distinct_kernels_counts_each_name_once(self) -> None:
        doc = self.document(REAL_KERNEL + REAL_KERNEL.replace(b"gemm_cluster", b"gemm_cluster_2"))
        assert doc["summary"]["kernel_patterns"] == 2
        assert doc["summary"]["distinct_kernels"] == 1


class TestAtlasPage:
    """--open shows a dump in the viewer that ships with the docs."""

    @staticmethod
    def seed_of(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        start = text.index('id="seed">') + len('id="seed">')
        return text[start : text.index("</script>", start)]

    @pytest.fixture
    def opened(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
        # Seeded pages land in tmp_path, not the real temp directory, so runs leave
        # nothing behind.
        seen: list[str] = []

        def browse(uri: str) -> bool:
            seen.append(uri)
            return True

        monkeypatch.setattr("webbrowser.open", browse)
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        return seen

    def test_the_page_ships_with_the_docs(self) -> None:
        assert atlas_page().name == "cgc-pattern-atlas.html"

    def test_without_a_dump_the_page_itself_is_opened(self, opened: list[str]) -> None:
        open_atlas(None)
        assert opened == [atlas_page().as_uri()]

    def test_a_dump_is_embedded_so_the_page_opens_on_it(
        self, tmp_path: Path, opened: list[str]
    ) -> None:
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        document = tmp_path / "patterns.json"
        document.write_text(
            json.dumps(patterns_document(ALL_FIXTURES, adapter, 720, None)), encoding="utf-8"
        )

        seeded = json.loads(self.seed_of(open_atlas(document)))
        assert seeded["patterns"] == json.loads(document.read_text(encoding="utf-8"))["patterns"]
        assert seeded["meta"]["label"] == str(document)

    def test_open_with_a_file_shows_it_without_touching_any_driver(
        self, tmp_path: Path, opened: list[str]
    ) -> None:
        # Needs neither a redist nor a GPU: run_patterns returns before DXCore.
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        document = tmp_path / "patterns.json"
        document.write_text(
            json.dumps(patterns_document(ALL_FIXTURES, adapter, 720, None)), encoding="utf-8"
        )

        run_patterns(Options(open_atlas=str(document)))

        # Seeded pages land in tmp_path (see `opened`); this run wrote exactly one.
        (page,) = tmp_path.glob("winml-*-cgc-pattern-atlas.html")
        seeded = json.loads(self.seed_of(page))
        assert seeded["meta"]["label"] == str(document)

    def test_open_with_a_missing_file_is_a_usage_error(self, tmp_path: Path) -> None:
        with pytest.raises(click.UsageError):
            run_patterns(Options(open_atlas=str(tmp_path / "absent.json")))

    def test_open_with_a_file_and_dump_together_is_a_usage_error(self, tmp_path: Path) -> None:
        # Two dumps to show; preferring either silently throws away what was asked for.
        with pytest.raises(click.UsageError):
            run_patterns(Options(adapter="nvidia", dump=True, open_atlas=str(tmp_path / "p.json")))

    def test_open_reports_an_unreadable_file_as_a_usage_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A path the OS refuses to stat must exit like any other bad argument.
        document = tmp_path / "locked.json"
        document.write_text("{}", encoding="utf-8")

        real_is_file = Path.is_file

        def deny_this_one(self: Path) -> bool:
            if self == document:
                raise PermissionError(13, "denied")
            return real_is_file(self)

        monkeypatch.setattr(Path, "is_file", deny_this_one)
        with pytest.raises(click.UsageError, match="could not be read"):
            run_patterns(Options(open_atlas=str(document)))

    @pytest.mark.parametrize(
        ("name", "content", "reason"),
        [
            ("notes.txt", "not json at all", "is not JSON"),
            ("other.json", '{"format": "something-else/1"}', f"is not a {PATTERNS_JSON_FORMAT}"),
            ("bare.json", "[]", f"is not a {PATTERNS_JSON_FORMAT}"),
        ],
    )
    def test_open_refuses_a_file_that_is_not_a_dump(
        self, tmp_path: Path, name: str, content: str, reason: str
    ) -> None:
        # A mistyped path that happens to exist is a bad argument, not a traceback,
        # and the message says what is wrong with it.
        wrong = tmp_path / name
        wrong.write_text(content, encoding="utf-8")
        with pytest.raises(click.UsageError, match=reason):
            run_patterns(Options(open_atlas=str(wrong)))

    def test_two_captures_of_one_driver_get_their_own_page(
        self, tmp_path: Path, opened: list[str]
    ) -> None:
        # Same adapter and driver version, captured under two roots: opening the
        # second must not rewrite the page the first one is showing.
        documents: list[Path] = []
        pages: list[Path] = []
        for capture in ("before", "after"):
            directory = tmp_path / capture / "nvidia-geforce-rtx-5090-d" / "32.0.16.3004"
            directory.mkdir(parents=True)
            document = directory / "patterns.json"
            adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
            document.write_text(
                json.dumps(patterns_document(KERNEL, adapter, 720, None)), encoding="utf-8"
            )
            documents.append(document)
            pages.append(open_atlas(document))

        assert pages[0] != pages[1]
        assert opened == [page.as_uri() for page in pages]
        first = json.loads(self.seed_of(pages[0]))
        assert first["meta"]["label"] == str(documents[0])

    def test_one_dump_gets_one_page_however_it_is_named(
        self, tmp_path: Path, opened: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = tmp_path / "nvidia-geforce-rtx-5090-d" / "32.0.16.3004"
        directory.mkdir(parents=True)
        adapter = make_adapter(0, "NVIDIA GeForce RTX 5090 D", hardware=True)
        document = json.dumps(patterns_document(KERNEL, adapter, 720, None))
        (directory / "patterns.json").write_text(document, encoding="utf-8")

        open_atlas(directory / "patterns.json")
        monkeypatch.chdir(directory)
        open_atlas(Path("patterns.json"))

        assert opened[0] == opened[1]
        assert "nvidia-geforce-rtx-5090-d" in opened[1]

    def test_a_pattern_cannot_close_the_script_element(
        self, tmp_path: Path, opened: list[str]
    ) -> None:
        # Every "<" is escaped, so a pattern whose own text holds </script> -- which
        # would end the element early and break the page -- arrives intact.
        document = tmp_path / "patterns.json"
        hostile: dict[str, Any] = {
            "format": PATTERNS_JSON_FORMAT,
            "meta": {},
            "patterns": [{"mlir": "</script><script>alert(1)</script>"}],
        }
        document.write_text(json.dumps(hostile), encoding="utf-8")

        raw = self.seed_of(open_atlas(document))
        assert "<" not in raw
        assert json.loads(raw)["patterns"] == hostile["patterns"]
