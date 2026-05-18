"""Uniform artifact writer for v9 experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(val) for val in value]
    return value


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=2, sort_keys=True)


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], *, fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(row.get(key, "")) for key in fieldnames})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hash_rows(paths: Iterable[Path], *, root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(paths):
        if path.is_file():
            rows.append({"artifact": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return rows
