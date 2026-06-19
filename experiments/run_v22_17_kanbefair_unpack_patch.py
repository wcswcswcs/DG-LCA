#!/usr/bin/env python3
"""Unpack uploaded KANbeFair zip, create worktree, and patch known blockers."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_v22_17_common import (  # noqa: E402
    OUT_ROOT,
    RAW_ROOT,
    WORKTREE_ROOT,
    ZIP_PATH,
    append_exec,
    copy_uploaded_zip_if_needed,
    ensure_out,
    file_manifest,
    sha256_file,
    sha256_text,
    tree_manifest_sha,
    write_execution_manifests,
    write_json,
    write_rows,
)
from dgkan.integration.kanbefair_adapter import firewall_rows  # noqa: E402


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true", help="recreate raw and worktree from zip")
    return p


def _chmod_readonly(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        try:
            if path.is_file():
                path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        except OSError:
            pass


def _patch_chdir(path: Path) -> tuple[int, int, str]:
    before = path.read_text(encoding="utf-8")
    detected = int("os.chdir('/home/yurunpeng/Repos/KANBeFair/src')" in before)
    after = before.replace(
        "os.chdir('/home/yurunpeng/Repos/KANBeFair/src')",
        "_THIS_DIR = os.path.dirname(os.path.abspath(__file__))\nos.chdir(_THIS_DIR)",
    )
    after = after.replace(
        "from fvcore.common.timer import Timer",
        (
            "try:\n"
            "    from fvcore.common.timer import Timer\n"
            "except Exception:\n"
            "    import time as _timer_time\n"
            "    class Timer:\n"
            "        def __init__(self):\n"
            "            self.reset()\n"
            "        def reset(self):\n"
            "            self._start = _timer_time.perf_counter()\n"
            "            self._elapsed = 0.0\n"
            "            self._paused = False\n"
            "        def pause(self):\n"
            "            if not self._paused:\n"
            "                self._elapsed += _timer_time.perf_counter() - self._start\n"
            "                self._paused = True\n"
            "        def resume(self):\n"
            "            if self._paused:\n"
            "                self._start = _timer_time.perf_counter()\n"
            "                self._paused = False\n"
            "        def is_paused(self):\n"
            "            return self._paused\n"
            "        def seconds(self):\n"
            "            if self._paused:\n"
            "                return self._elapsed\n"
            "            return self._elapsed + (_timer_time.perf_counter() - self._start)\n"
            "        def avg_seconds(self):\n"
            "            return self.seconds()"
        ),
    )
    patched = int(after != before)
    if patched:
        path.write_text(after, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"raw/{path.name}",
            tofile=f"worktree/{path.name}",
        )
    )
    return detected, patched, diff


def _patch_utils_lazy(path: Path) -> tuple[int, int, str]:
    before = path.read_text(encoding="utf-8")
    detected = int(
        "import torch, random, numpy, torchtext" in before
        or "from data.text import create_text_loader" in before
        or "from fvcore.nn import FlopCountAnalysis, parameter_count" in before
    )
    after = before
    after = after.replace("import torch, random, numpy, torchtext", "import torch, random, numpy")
    after = after.replace(
        "from fvcore.nn import FlopCountAnalysis, parameter_count",
        (
            "try:\n"
            "    from fvcore.nn import FlopCountAnalysis, parameter_count\n"
            "except Exception:\n"
            "    FlopCountAnalysis = None\n"
            "    parameter_count = None"
        ),
    )
    after = after.replace(
        "from data.text import create_text_loader, get_IMDb_dataset\nfrom data.audio import SubsetSC, get_SC_loader, get_US_dataset\n",
        (
            "from_uciml_to_dataset = None\n"
            "torchtext = None\n"
            "create_text_loader = None\n"
            "get_IMDb_dataset = None\n"
            "SubsetSC = None\n"
            "get_SC_loader = None\n"
            "get_US_dataset = None\n\n"
            "def _require_text_deps():\n"
            "    global torchtext, create_text_loader, get_IMDb_dataset\n"
            "    if torchtext is None:\n"
            "        import torchtext as _torchtext\n"
            "        torchtext = _torchtext\n"
            "    if create_text_loader is None or get_IMDb_dataset is None:\n"
            "        from data.text import create_text_loader as _create_text_loader, get_IMDb_dataset as _get_IMDb_dataset\n"
            "        create_text_loader = _create_text_loader\n"
            "        get_IMDb_dataset = _get_IMDb_dataset\n\n"
            "def _require_audio_deps():\n"
            "    global SubsetSC, get_SC_loader, get_US_dataset\n"
            "    if SubsetSC is None or get_SC_loader is None or get_US_dataset is None:\n"
            "        from data.audio import SubsetSC as _SubsetSC, get_SC_loader as _get_SC_loader, get_US_dataset as _get_US_dataset\n"
            "        SubsetSC = _SubsetSC\n"
            "        get_SC_loader = _get_SC_loader\n"
            "        get_US_dataset = _get_US_dataset\n"
        ),
    )
    after = after.replace("from data.uciml import from_uciml_to_dataset\n", "")
    helper_anchor = "def _require_text_deps():\n"
    if helper_anchor in after and "def _require_uciml_deps():" not in after:
        after = after.replace(
            helper_anchor,
            (
                "def _require_uciml_deps():\n"
                "    global from_uciml_to_dataset\n"
                "    if from_uciml_to_dataset is None:\n"
                "        from data.uciml import from_uciml_to_dataset as _from_uciml_to_dataset\n"
                "        from_uciml_to_dataset = _from_uciml_to_dataset\n\n"
                f"{helper_anchor}"
            ),
        )
    after = after.replace(
        "train_dataset, test_dataset = from_uciml_to_dataset(",
        "_require_uciml_deps()\n        train_dataset, test_dataset = from_uciml_to_dataset(",
    )
    for marker in ['elif args.dataset == "AG_NEWS":', 'elif args.dataset == "CoLA":', 'elif args.dataset == "IMDb":']:
        after = after.replace(marker, marker + "\n        _require_text_deps()")
    for marker in ['elif args.dataset == "SpeechCommand":', 'elif args.dataset == "UrbanSound8K":']:
        after = after.replace(marker, marker + "\n        _require_audio_deps()")
    patched = int(after != before)
    if patched:
        path.write_text(after, encoding="utf-8")
    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile="raw/utils.py",
            tofile="worktree/utils.py",
        )
    )
    return detected, patched, diff


def _copy_local_torchvision_cache(worktree: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_root = ROOT / "data"
    target_root = worktree / "dataset"
    target_root.mkdir(parents=True, exist_ok=True)
    for name in ["MNIST", "FashionMNIST", "KMNIST"]:
        src = source_root / name
        dst = target_root / name
        if not src.exists():
            rows.append({"dataset": name, "cache_action": "source_cache_missing", "source": str(src), "target": str(dst)})
            continue
        if dst.exists():
            rows.append({"dataset": name, "cache_action": "already_present", "source": str(src), "target": str(dst)})
            continue
        shutil.copytree(src, dst)
        rows.append({"dataset": name, "cache_action": "copied_local_cache_to_worktree", "source": str(src), "target": str(dst)})
    return rows


def main() -> None:
    args = parser().parse_args()
    ensure_out()
    zip_path, zip_note = copy_uploaded_zip_if_needed()
    if args.fresh:
        for root in [RAW_ROOT, WORKTREE_ROOT]:
            if root.exists():
                shutil.rmtree(root)
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if not (RAW_ROOT / "KANbeFair-main").exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(RAW_ROOT)
    if WORKTREE_ROOT.exists() and args.fresh:
        shutil.rmtree(WORKTREE_ROOT)
    if not WORKTREE_ROOT.exists():
        shutil.copytree(RAW_ROOT / "KANbeFair-main", WORKTREE_ROOT)

    raw_rows = file_manifest(RAW_ROOT / "KANbeFair-main")
    (OUT_ROOT / "v22_17_kanbefair_raw_filelist.txt").write_text(
        "\n".join(str(r["relative_path"]) for r in raw_rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "v22_17_kanbefair_raw_sha256_manifest.txt").write_text(
        "\n".join(f"{r['sha256']}  {r['relative_path']}" for r in raw_rows) + "\n",
        encoding="utf-8",
    )
    initial_worktree_rows = file_manifest(WORKTREE_ROOT)
    (OUT_ROOT / "v22_17_kanbefair_worktree_initial_filelist.txt").write_text(
        "\n".join(str(r["relative_path"]) for r in initial_worktree_rows) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "v22_17_kanbefair_zip_sha256.txt").write_text(f"{sha256_file(ZIP_PATH)}  {ZIP_PATH}\n", encoding="utf-8")

    chdir_diffs = []
    patch_rows: list[dict[str, object]] = []
    for rel in ["src/train.py", "src/train_continual_learning.py"]:
        path = WORKTREE_ROOT / rel
        detected, patched, diff = _patch_chdir(path)
        chdir_diffs.append(diff)
        patch_rows.append(
            {
                "patch_id": f"chdir:{rel}",
                "file": rel,
                "detected": detected,
                "patched": patched,
                "patch_sha256": sha256_text(diff),
                "reason": "replace hardcoded os.chdir with script directory and add Timer fallback if fvcore is absent",
            }
        )
    chdir_diff = "\n".join(d for d in chdir_diffs if d)
    (OUT_ROOT / "v22_17_kanbefair_chdir_patch.diff").write_text(chdir_diff, encoding="utf-8")

    utils_detected, utils_patched, utils_diff = _patch_utils_lazy(WORKTREE_ROOT / "src/utils.py")
    (OUT_ROOT / "v22_17_kanbefair_lazy_import_patch.diff").write_text(utils_diff, encoding="utf-8")
    patch_rows.append(
        {
            "patch_id": "lazy_import:src/utils.py",
            "file": "src/utils.py",
            "detected": utils_detected,
            "patched": utils_patched,
            "patch_sha256": sha256_text(utils_diff),
            "reason": "make torchtext/torchaudio imports lazy so vision/tabular smoke is not blocked",
        }
    )
    cache_rows = _copy_local_torchvision_cache(WORKTREE_ROOT)
    write_rows(OUT_ROOT / "v22_17_kanbefair_dataset_cache_actions.csv", cache_rows)

    write_rows(OUT_ROOT / "v22_17_kanbefair_worktree_patch_manifest.csv", patch_rows)
    write_rows(OUT_ROOT / "v22_17_goal_drift_firewall.csv", firewall_rows())
    _chmod_readonly(RAW_ROOT / "KANbeFair-main")
    raw_after = file_manifest(RAW_ROOT / "KANbeFair-main")
    write_json(
        OUT_ROOT / "v22_17_kanbefair_provenance.json",
        {
            "kanbefair_source_kind": "uploaded_zip",
            "kanbefair_zip_path": str(ZIP_PATH),
            "kanbefair_zip_sha256": sha256_file(ZIP_PATH),
            "zip_note": zip_note,
            "kanbefair_raw_root": str((RAW_ROOT / "KANbeFair-main").resolve()),
            "kanbefair_worktree_root": str(WORKTREE_ROOT.resolve()),
            "kanbefair_raw_file_count": len(raw_rows),
            "kanbefair_raw_tree_manifest_sha256": tree_manifest_sha(raw_rows),
            "kanbefair_raw_tree_manifest_sha256_after_patch": tree_manifest_sha(raw_after),
            "kanbefair_git_commit": "NA_uploaded_zip_no_git_metadata",
            "kanbefair_original_raw_unchanged": int(tree_manifest_sha(raw_rows) == tree_manifest_sha(raw_after)),
            "hardcoded_chdir_detected": int(any(int(r["detected"]) for r in patch_rows if str(r["patch_id"]).startswith("chdir"))),
            "hardcoded_chdir_patched": int(all(int(r["patched"]) for r in patch_rows if str(r["patch_id"]).startswith("chdir"))),
            "lazy_text_audio_patch_used": utils_patched,
        },
    )
    write_execution_manifests()
    append_exec(
        f"{sys.executable} experiments/run_v22_17_kanbefair_unpack_patch.py {'--fresh' if args.fresh else ''}".strip(),
        task_id="A-provenance-patch",
        status="pass",
        gpu="0",
        files=(
            "results/v22_17/v22_17_kanbefair_provenance.json, "
            "results/v22_17/v22_17_kanbefair_worktree_patch_manifest.csv"
        ),
        note="raw tree kept read-only; patches only applied to worktree",
        exit_code=0,
    )


if __name__ == "__main__":
    main()
