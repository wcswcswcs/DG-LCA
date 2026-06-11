"""Source-channel readback helpers for v20 experiments.

These helpers deliberately only summarize already measured rows.  They do not
generate functional-update directions.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable


HORIZONS = ("h800", "h1600", "h3200", "h4800")


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in {"", None}:
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def retention_ratio(current_source: Any, previous_source: Any, eps: float = 1.0e-12) -> float | str:
    previous = finite_float(previous_source)
    current = finite_float(current_source)
    if not math.isfinite(previous) or previous <= 0.0 or not math.isfinite(current):
        return ""
    return max(0.0, current) / max(float(eps), previous)


def debt_recovery(peak: Any, final: Any, eps: float = 1.0e-12) -> float | str:
    peak_f = finite_float(peak)
    final_f = finite_float(final)
    if not math.isfinite(peak_f) or not math.isfinite(final_f):
        return ""
    if peak_f <= 0.0:
        return ""
    return max(0.0, min(1.0, (peak_f - final_f) / (peak_f + float(eps))))


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman(xs: Iterable[Any], ys: Iterable[Any]) -> float | str:
    pairs = [(finite_float(x), finite_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return ""
    rx = _rank([p[0] for p in pairs])
    ry = _rank([p[1] for p in pairs])
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    vx = sum((x - mx) ** 2 for x in rx)
    vy = sum((y - my) ** 2 for y in ry)
    if vx <= 0.0 or vy <= 0.0:
        return ""
    cov = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    return cov / math.sqrt(vx * vy)


def grouped_source_retention(
    rows: Iterable[dict[str, Any]],
    *,
    group_keys: tuple[str, ...] = ("carrier", "basis_repair_variant", "continuation_id"),
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k, "")) for k in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        item = {k: v for k, v in zip(group_keys, key)}
        item["rows"] = len(group)
        if group:
            item["mechanism"] = group[0].get("mechanism", "")
        for horizon in HORIZONS:
            vals = [finite_float(r.get(f"source_{horizon}")) for r in group]
            vals = [v for v in vals if math.isfinite(v)]
            item[f"source_{horizon}_mean"] = sum(vals) / len(vals) if vals else ""
            item[f"source_{horizon}_pass_count"] = sum(1 for v in vals if v >= 0.005)
        item["retention_h1600_over_h800"] = retention_ratio(item.get("source_h1600_mean"), item.get("source_h800_mean"))
        item["retention_h3200_over_h1600"] = retention_ratio(item.get("source_h3200_mean"), item.get("source_h1600_mean"))
        item["retention_h4800_over_h3200"] = retention_ratio(item.get("source_h4800_mean"), item.get("source_h3200_mean"))
        item["retained_h3200_candidate"] = int(
            finite_float(item.get("source_h800_mean"), -999.0) >= 0.005
            and finite_float(item.get("source_h1600_mean"), -999.0) >= 0.005
            and finite_float(item.get("source_h3200_mean"), -999.0) >= 0.005
            and finite_float(item.get("retention_h1600_over_h800"), 0.0) >= 0.40
            and finite_float(item.get("retention_h3200_over_h1600"), 0.0) >= 0.50
        )
        item["retained_h4800_candidate"] = int(
            bool(item["retained_h3200_candidate"])
            and finite_float(item.get("source_h4800_mean"), -999.0) >= 0.005
            and finite_float(item.get("retention_h4800_over_h3200"), 0.0) >= 0.50
        )
        item["late_positive_candidate"] = int(
            finite_float(item.get("source_h800_mean"), 0.0) <= 0.0
            and (
                finite_float(item.get("source_h1600_mean"), -999.0) >= 0.005
                or finite_float(item.get("source_h3200_mean"), -999.0) >= 0.005
                or finite_float(item.get("source_h4800_mean"), -999.0) >= 0.005
            )
        )
        out.append(item)
    return out


def best_control_attribution(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source = finite_float(row.get("source_h3200_mean"), finite_float(row.get("source_h1600_mean")))
        best_control = finite_float(row.get("best_control_source_h3200"), float("nan"))
        out.append(
            {
                "carrier": row.get("carrier", ""),
                "mechanism": row.get("mechanism", row.get("continuation_id", "")),
                "source_h3200_mean": source if math.isfinite(source) else "",
                "best_control_source_h3200": best_control if math.isfinite(best_control) else "",
                "control_equivalent": int(math.isfinite(source) and math.isfinite(best_control) and best_control >= source - 0.005),
            }
        )
    return out
