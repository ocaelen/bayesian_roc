"""Small reproducibility records stored beside completed experiment CSVs."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from threadpoolctl import threadpool_info, threadpool_limits


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading the full experimental CSV into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command, *, cwd=None) -> str | None:
    """Read optional local environment details; never run network commands."""
    try:
        result = subprocess.run(
            command, cwd=cwd, text=True, capture_output=True, check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_run_metadata(
    *, source_root, source_paths, configuration,
    metadata_schema="synthetic_run_metadata_v1",
) -> dict:
    """Record scientific configuration and the environment outside timing.

    Source hashes identify the actual working files even with uncommitted edits.
    Git details and the processor description are best-effort local diagnostics.
    """
    if metadata_schema not in ("synthetic_run_metadata_v1", "real_run_metadata_v1"):
        raise ValueError("Unknown experiment provenance schema.")
    packages = ("numpy", "scipy", "threadpoolctl")
    if metadata_schema == "real_run_metadata_v1":
        packages += ("scikit-learn",)
    root = Path(source_root).resolve()
    sources = {
        str(Path(path).resolve().relative_to(root)): file_sha256(path)
        for path in sorted(source_paths)
    }
    assert sources, "At least one source file must be identified."
    revision = _command_output(["git", "rev-parse", "HEAD"], cwd=root)
    status = _command_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root,
    ) if revision is not None else None
    processor = platform.processor()
    if platform.system() == "Darwin":
        processor = _command_output(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
        ) or processor
    # This diagnostic context is separate from every construction's timer.
    with threadpool_limits(limits=1):
        native_libraries = threadpool_info()
        assert all(info["num_threads"] == 1 for info in native_libraries)
    return {
        "metadata_schema": metadata_schema,
        "configuration": configuration,
        "source_sha256": sources,
        "git_revision": revision,
        "git_worktree_modified": None if status is None else bool(status),
        "software": {
            "python": platform.python_version(),
            **{name: version(name) for name in packages},
        },
        "timing_environment": {
            "operating_system": platform.platform(),
            "architecture": platform.machine(),
            "processor": processor or "unavailable",
            "logical_cpu_count": os.cpu_count(),
            "timing_threads": 1,
            "clock": "time.perf_counter_ns",
            "native_libraries_under_timing_limit": native_libraries,
        },
    }


def metadata_path_for_csv(csv_path, *, checksum=None) -> Path:
    """Identify a CSV's companion without modifying any previous companion."""
    csv_path = Path(csv_path)
    checksum = file_sha256(csv_path) if checksum is None else checksum
    return csv_path.with_name(f"{csv_path.stem}.{checksum}.metadata.json")


def assert_sources_unchanged(metadata, *, source_root) -> None:
    """Do not publish a long run under source files edited while it was running."""
    for relative_path, expected in metadata["source_sha256"].items():
        path = Path(source_root) / relative_path
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Experiment source changed during this run: {relative_path}")


def publish_run_metadata(temporary_csv, output_csv, metadata) -> Path:
    """Publish provenance before the caller atomically publishes its CSV.

    A checksum-specific filename keeps an old CSV's provenance intact if a new
    CSV fails to publish. A failure can leave an unused metadata file, but cannot
    replace the provenance belonging to different, previously completed results.
    """
    checksum = file_sha256(temporary_csv)
    destination = metadata_path_for_csv(output_csv, checksum=checksum)
    payload = dict(metadata, raw_csv_sha256=checksum)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise ValueError(f"Different provenance already exists for this CSV: {destination}")
        return destination
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def read_run_metadata(csv_path, *, metadata_schema="synthetic_run_metadata_v1") -> dict:
    """Read a companion and verify that it identifies these exact CSV bytes."""
    if metadata_schema not in ("synthetic_run_metadata_v1", "real_run_metadata_v1"):
        raise ValueError("Unknown experiment provenance schema.")
    checksum = file_sha256(csv_path)
    path = metadata_path_for_csv(csv_path, checksum=checksum)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metadata_schema") != metadata_schema:
        raise ValueError(f"Expected {metadata_schema} experiment provenance.")
    if payload.get("raw_csv_sha256") != checksum:
        raise ValueError("Provenance does not match the experimental CSV.")
    return payload
