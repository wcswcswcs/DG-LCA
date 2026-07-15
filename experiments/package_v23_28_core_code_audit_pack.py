#!/usr/bin/env python3
"""Create the v23.28 core-code and evidence audit zip.

The package is intentionally source-and-evidence focused: it includes DG-KAN
source, tests, the v23.28 runner, close v23.27 context, v23.28/v23.27 docs and
logs, and the lightweight v23.28 result artifacts created by this run. It
excludes large/generated binary payloads and older audit archives.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


INCLUDE_ROOTS = (
    "dgkan",
    "tests",
)

INCLUDE_FILES = (
    "experiments/run_v23_27_persistent_compositional_curvature_lie_generator_fu.py",
    "experiments/run_v23_28_persistent_compositional_transport_shaping_fu.py",
    "experiments/package_v23_28_core_code_audit_pack.py",
    "requirement.txt",
)

INCLUDE_GLOBS = (
    "docs/DG-KAN_v23.27_*.md",
    "docs/DG-KAN_v23.28_*.md",
    "results/v23_28*/v23_28_*.csv",
    "results/v23_28*/v23_28_*.json",
    "results/v23_28*/v23_28_*.md",
    "results/v23_28*/v23_28_*.txt",
    "results/diag_v23_28*/v23_28_*.csv",
    "results/diag_v23_28*/v23_28_*.json",
    "results/diag_v23_28*/v23_28_*.md",
    "results/diag_v23_28*/v23_28_*.txt",
    "results/v23_28*/partB_generated_checkpoints/*.npz",
    "results/v23_27*/v23_27_partD_exported_checkpoint_manifest.csv",
    "results/v23_27*/v23_27_exported_checkpoints/*.npz",
    "results/v22_66/chunks/*v23_28_partI_mcga_probe*.csv",
)

EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "code_audit_pack",
}
EXCLUDED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".zip",
    ".tar",
    ".tar.gz",
    ".npy",
    ".npz",
    ".sqlite",
    ".db",
)


def is_excluded(rel_path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in rel_path.parts):
        return True
    if rel_path.match("results/v23_28*/partB_generated_checkpoints/*.npz"):
        return False
    if rel_path.match("results/v23_27*/v23_27_exported_checkpoints/*.npz"):
        return False
    return any(rel_path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def iter_files(repo_root: Path) -> list[Path]:
    files: set[Path] = set()

    for rel_root in INCLUDE_ROOTS:
        root = repo_root / rel_root
        if not root.exists():
            continue
        for path in root.rglob("*"):
            rel = path.relative_to(repo_root)
            if path.is_file() and not is_excluded(rel):
                files.add(path)

    for rel_file in INCLUDE_FILES:
        path = repo_root / rel_file
        if path.is_file() and not is_excluded(path.relative_to(repo_root)):
            files.add(path)

    for pattern in INCLUDE_GLOBS:
        for path in repo_root.glob(pattern):
            rel = path.relative_to(repo_root)
            if path.is_file() and not is_excluded(rel):
                files.add(path)

    return sorted(files, key=lambda p: p.relative_to(repo_root).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output zip path.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    files = iter_files(repo_root)
    if not files:
        raise RuntimeError("No files selected for the v23.28 core-code audit package.")

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as zf:
        for path in files:
            zf.write(path, path.relative_to(repo_root).as_posix())
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Zip integrity check failed at {bad_member}")

    digest = sha256_file(output)
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.as_posix()}\n", encoding="utf-8")

    manifest_path = output.with_name(f"{output.stem}_manifest.txt")
    manifest_path.write_text(
        "\n".join(path.relative_to(repo_root).as_posix() for path in files) + "\n",
        encoding="utf-8",
    )

    file_sha_path = output.with_name(f"{output.stem}_file_sha256sums.txt")
    file_sha_path.write_text(
        "\n".join(f"{sha256_file(path)}  {path.relative_to(repo_root).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )

    print(f"zip={output}")
    print(f"sha256={digest}")
    print(f"sha256_file={sha_path}")
    print(f"manifest={manifest_path}")
    print(f"file_sha256sums={file_sha_path}")
    print(f"file_count={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
