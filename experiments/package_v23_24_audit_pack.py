#!/usr/bin/env python3
"""Create the v23.24 audit zip with a reproducible include/exclude policy."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path


INCLUDE_ROOTS = (
    "dgkan",
    "experiments",
    "tests",
    "results/v23_24",
    "results/v23_24_minreal_A_to_D",
    "results/v23_24_minreal_E_to_H",
    "results/v23_24_h20_A_to_D",
    "results/v23_24_h20_E_to_H",
    "results/v23_24_combo_K0_K2",
    "results/v23_24_combo_K3_K5",
    "third_party",
    "KANbeFair",
)

EXCLUDED_DIR_NAMES = {".git", "__pycache__", ".pytest_cache", "code_audit_pack"}
EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".zip", ".tar.gz")


def is_excluded(path: Path) -> bool:
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    return any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


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

    docs_root = repo_root / "docs"
    for path in docs_root.glob("DG-KAN_v23.24_*.md"):
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
        raise RuntimeError("No files selected for audit package.")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as zf:
        for path in files:
            zf.write(path, path.relative_to(repo_root).as_posix())
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Zip integrity check failed at {bad}")

    digest = sha256_file(output)
    sha_path = output.with_suffix(output.suffix + ".sha256")
    sha_path.write_text(f"{digest}  {output.as_posix()}\n", encoding="utf-8")

    manifest_path = output.with_name(f"{output.stem}_manifest.txt")
    manifest_path.write_text(
        "\n".join(path.relative_to(repo_root).as_posix() for path in files) + "\n",
        encoding="utf-8",
    )

    print(f"zip={output}")
    print(f"sha256={digest}")
    print(f"manifest={manifest_path}")
    print(f"file_count={len(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
