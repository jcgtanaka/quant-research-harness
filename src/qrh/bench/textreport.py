"""Plain-text bench report, built only from the receipt."""

from __future__ import annotations

from typing import Any

from qrh.bench.compare import within_receipt
from qrh.bench.receipt import BenchReceipt

TITLES = {
    "planted-bugs": "Planted bugs (S1 and S2 detectors)",
    "noise-fpr": "Noise false positives (S3-S5)",
    "known-edge-power": "Known-edge power (S3-S5)",
    "history": "History (stored trials re-evaluated, read-only)",
    "onboarding": "Dataset onboarding (verification of correct and planted-wrong instrument proposals)",
    "param-stability": "Parameter stability (S5: planted spikes must fail the neighbour check, planted plateaus must pass)",
    "llm-gate": "LLM contamination gate (S2: cutoff gate, declaration lint, replay of truncated runs)",
}


def pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def cell(m: dict[str, Any]) -> str:
    """``k/n rate [lo, hi]``"""
    if m["n"] == 0:
        return "0/0"
    lo, hi = m["ci"]
    return f"{m['k']}/{m['n']} {pct(m['rate'])} [{pct(lo)}, {pct(hi)}]"


def arms(r: BenchReceipt, suite: str) -> list[tuple[str, dict[str, Any]]]:
    if r.baseline_results is not None and suite in r.baseline_results:
        return [("default", r.baseline_results[suite]), ("candidate", r.results[suite])]
    return [("results", r.results[suite])]


def format_comparison(cmp: dict[str, Any], gated_only: bool = True) -> str:
    lines = [f"{cmp['a']} (a) vs {cmp['b']} (b); alpha {cmp['alpha']}; delta = b - a"]
    for mid, (x, y) in cmp["method_differences"].items():
        lines.append(f"  method {mid}: {x} -> {y}")
    lines += [f"  note: {n}" for n in cmp["notes"]]
    rows = [r for r in cmp["rows"] if r["gate"] or not gated_only]
    for r in rows:
        a, b = r["a"], r["b"]
        p = "n/a" if r["p_value"] is None else f"{r['p_value']:.3g}"
        delta = "n/a" if r["delta"] is None else f"{100 * r['delta']:+.1f} pts"
        flag = "distinguishable" if r["distinguishable"] else "not distinguishable"
        better = {"a": "a better", "b": "b better", "tie": "tie"}.get(r["better"], "n/a")
        lines.append(
            f"  {r['name']:<46} a {a['k']}/{a['n']}  b {b['k']}/{b['n']}  delta {delta}  p={p}  {flag}  ({better})"
        )
    return "\n".join(lines)


def _planted(r: BenchReceipt, suite: str = "planted-bugs") -> list[str]:
    lines = []
    for label, res in arms(r, suite):
        lines.append(f"[{label}] replications per variant: {res['reps']}")
        for name, m in res["metrics"].items():
            if name.count("/") == 1:
                kind = ("false alarm" if name.startswith("false_alarm") else
                        "expected undetectable (flagged anyway)" if name.startswith("expected_undetectable") else "detected")
                lines.append(f"  {name.split('/')[1]:<28} {kind:<12} {cell(m)}")
        lines.append("  by detector (flags over all variants and replications):")
        for bug, det in res["detectors"].items():
            lines.append("    " + f"{bug:<28}" + "  ".join(f"{k}={v}" for k, v in det.items()))
    return lines


def _noise(r: BenchReceipt) -> list[str]:
    lines = []
    for label, res in arms(r, "noise-fpr"):
        lines.append(f"[{label}] nominal level (1 - min_dsr): {res['nominal']:.3f}; pass = S4 and S5 gates")
        for key, info in res["cells"].items():
            m = res["metrics"][f"fpr/{key}"]
            over = "  EXCEEDS NOMINAL" if info["exceeds_nominal"] else ""
            lines.append(f"  {key:<14} false positives {cell(m)}  errors {info['errors']}{over}")
        for name in ("fpr", "pass_s4", "pass_pbo", "pass_dsr", "pass_s5"):
            lines.append(f"  pooled {name:<8} {cell(res['metrics'][f'{name}/pooled'])}")
    return lines


def _power(r: BenchReceipt) -> list[str]:
    lines = []
    for label, res in arms(r, "known-edge-power"):
        lines.append(f"[{label}] power by planted trend strength (latent drift standard deviation per bar)")
        for key, info in res["cells"].items():
            lines.append(f"  {key:<8} power {cell(res['metrics'][f'power/{key}'])}  errors {info['errors']}")
    return lines


def _history(r: BenchReceipt) -> list[str]:
    h = r.results["history"]
    s = h["summary"]
    lines = [
        f"source {h['source']} (input hash {h['input_hash'][:12]}): evaluated {s['evaluated']}, skipped {s['skipped']}, "
        f"default recompute reproduced the stored verdict for {s['reproduced']}",
        f"verdicts through S5 that would change: {s['changed']} ({s['stricter']} stricter, {s['looser']} looser)",
    ]
    for t in h["trials"]:
        if t["status"] == "skipped":
            lines.append(f"  {t['trial_id']}  skipped: {t['reason']}")
        elif t["changed"]:
            lines.append(f"  {t['trial_id']}  {t['direction']}: default pass={t['default']['pass_all']}, candidate pass={t['candidate']['pass_all']}")
    return lines


def format_receipt(r: BenchReceipt) -> str:
    lines = [
        f"Bench {r.bench_id()}   profile {r.profile}   seed {r.seed}",
        f"Receipt hash: {r.receipt_hash()}",
        f"Harness code hash: {r.code_hash}",
        "Methods used:",
    ]
    for mid, version in sorted(r.methods.items()):
        flag = "  EXPERIMENTAL: not the default version" if mid in r.non_default_methods else ""
        lines.append(f"  {mid}={version}{flag}")
    if r.baseline_methods is not None:
        lines.append("Compared with the defaults: " + ", ".join(f"{k}={v}" for k, v in sorted(r.baseline_methods.items()) if r.methods.get(k) != v))
    for suite in r.suites:
        lines += ["", f"== {TITLES[suite]} =="]
        lines += {"planted-bugs": _planted, "noise-fpr": _noise, "known-edge-power": _power, "history": _history,
                  "llm-gate": lambda rec: _planted(rec, "llm-gate"),
                  "onboarding": lambda rec: _planted(rec, "onboarding"),
                  "param-stability": lambda rec: _planted(rec, "param-stability")}[suite](r)
    if r.baseline_results is not None:
        lines += ["", "== Default vs candidate (gated metrics) ==", format_comparison(within_receipt(r))]
    lines += ["", "Synthetic data and research use only; not investment advice."]
    return "\n".join(lines) + "\n"
