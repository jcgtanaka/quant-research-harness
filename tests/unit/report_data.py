"""Small synthetic stage data for the report tests (shapes match what the stages persist)."""

import numpy as np
import pandas as pd
from qrh.pipeline.receipt import Receipt, Stage, Verdict

TRIAL = "trial-synthetic"


def _receipt(stage, verdict=Verdict.PASS, reasons=(), metrics=None, params=None):
    return Receipt(
        trial_id=TRIAL, stage=Stage(stage), hypothesis_hash="h" * 64, code_hash="c" * 64,
        verdict=verdict, reasons=list(reasons), metrics=metrics or {}, params=params or {},
    )


def _idx(n=120, freq="D"):
    return pd.date_range("2021-01-01", periods=n, freq=freq, tz="UTC", name="timestamp")


def _combos(grid):
    rng = np.random.default_rng(3)
    names = sorted(grid)
    out = []
    import itertools

    for i, values in enumerate(itertools.product(*(grid[k] for k in names))):
        sr = float(rng.normal(0.5, 0.8))
        out.append({
            "combo": f"c{i:03d}", "params": dict(zip(names, values)), "sharpe_annualized": sr,
            "sharpe_per_period": sr / 16, "cagr": sr / 10, "max_drawdown": -abs(sr) / 8, "n_obs": 120.0,
        })
    return {"param_names": names, "periods_per_year": 252.0, "combos": out}


def build(stage, verdict=Verdict.PASS, variant=None):
    """(receipt, artifacts) for one stage. ``variant`` selects alternative shapes."""
    rng = np.random.default_rng(1)
    stage = Stage(stage)
    if stage is Stage.S0:
        art = {"hypothesis": {
            "trial_id": TRIAL, "text": "A synthetic rule carries information.", "strategy_family": "trend",
            "asset_class": "synthetic", "frequency": "1D", "family_id": "trend|synthetic|1D",
            "parameter_space_size": 9, "difference_statement": "Baseline.", "hypothesis_hash": "h" * 64,
            "thresholds": {"max_pbo": 0.2, "min_dsr": 0.95, "min_oos_sharpe": 0, "bootstrap_reps": 2000},
            "adapter": "adapter:A", "datasets": [{"name": "prices", "slow": False, "publication_column": None}],
            "windows": {"development": ["2021-01-01", "2021-12-31"]}, "bounds": {"max_gross_leverage": 1.0},
            "walk_forward": {"train": 60, "test": 20}, "periods_per_year": None, "s2_params": None,
        }}
        return _receipt("S0", verdict, params={"thresholds": art["hypothesis"]["thresholds"]}), art
    if stage is Stage.S1:
        rules = [
            {"rule": "gaps#1", "type": "gaps", "passed": variant != "fail", "offenders": ["2021-02-03T00:00:00+00:00"] if variant == "fail" else [], "reasons": []},
            {"rule": "range#2", "type": "range", "passed": True, "offenders": [], "reasons": []},
        ]
        ds = [
            {"name": "prices", "status": "PASS", "reasons": [], "start": "2021-01-01T00:00:00+00:00",
             "end": "2021-04-30T00:00:00+00:00", "rows": 118, "step_seconds": 86400.0, "gap_count": 1,
             "gaps": [{"after": "2021-02-02T00:00:00+00:00", "before": "2021-02-05T00:00:00+00:00", "missing": 2}], "rules": rules},
            {"name": "macro", "status": "PASS", "reasons": [], "start": "2021-01-05T00:00:00+00:00",
             "end": "2021-04-20T00:00:00+00:00", "rows": 106, "step_seconds": 86400.0, "gap_count": 0, "gaps": [], "rules": []},
        ]
        return _receipt("S1", verdict, params={}), {"coverage": {"datasets": ds}}
    if stage is Stage.S2:
        idx = _idx(60)
        pos = np.sign(np.sin(np.arange(60) / 6.0))
        cutoffs = []
        for frac, k in ((0.25, 15), (0.5, 30), (0.75, 45)):
            full = pos[:k].tolist()
            trunc = list(full)
            div = None
            status = "PASS"
            if variant == "diverge" and frac == 0.5:
                trunc[20:] = [-v for v in trunc[20:]]
                div = {"timestamp": idx[20].isoformat(), "what": "positions"}
                status = "FAIL"
            cutoffs.append({
                "fraction": frac, "cutoff": idx[k].isoformat(), "status": status, "first_divergence": div,
                "column": "w", "columns_total": 1, "timestamps": [t.isoformat() for t in idx[:k]],
                "full": full, "truncated": trunc, "stride": 1,
            })
        art = {
            "lint": {"verdict": "PASS", "findings": [
                {"rule": "LA01", "at": "adapter.py:12", "line": 12, "severity": "WARN", "message": "centered window", "snippet": "x.rolling(5, center=True)", "suppressed": False, "reason": None},
                {"rule": "LA02", "at": "adapter.py:30", "line": 30, "severity": "BLOCK", "message": "negative shift", "snippet": "x.shift(-1)", "suppressed": True, "reason": "label only"},
            ]},
            "truncation": {"verdict": "PASS", "reasons": [], "notes": [], "tolerance": 1e-12, "cutoffs": cutoffs},
            "sanity": {"verdict": "PASS", "reasons": [], "rows": [
                {"name": "max_abs_period_return", "value": 0.03, "bound": 0.5, "ok": True},
                {"name": "max_gross_leverage", "value": 1.0, "bound": 1.0, "ok": True},
                {"name": "median_abs_period_return", "value": 0.004, "bound": 0.5, "ok": True},
            ]},
        }
        if variant == "empty_lint":
            art["lint"]["findings"] = []
        return _receipt("S2", verdict, params={}), art
    if stage is Stage.S3:
        grid = {"fast": [5, 10, 20], "slow": [40, 80, 160]}
        if variant == "wide":
            grid = {"a": list(range(8)), "b": [1, 2, 3, 4, 5], "c": [0.1, 0.2]}
        return _receipt("S3", verdict, metrics={"grid_size": 9.0}, params={"periods_per_year": 252.0}), {"combos": _combos(grid)}
    if stage is Stage.S4:
        idx = _idx(80)
        ret = pd.Series(rng.normal(0.001, 0.01, 80), index=idx)
        folds = [
            {"fold": i, "train": [idx[0].isoformat() if i == 0 else idx[i * 20].isoformat(), idx[i * 20 + 39].isoformat()],
             "test": [idx[i * 20 + 40].isoformat() if i * 20 + 40 < 80 else idx[-1].isoformat(), idx[min(i * 20 + 59, 79)].isoformat()],
             "combo_hash": f"c{i}", "params": {"fast": 5 * (i + 1), "slow": 40}, "train_sharpe": 1.5 - i * 0.2, "test_sharpe": 0.8 - i * 0.1}
            for i in range(2)
        ]
        art = {
            "folds": {"folds": folds, "window": [idx[0].isoformat(), idx[-1].isoformat()], "periods_per_year": 252.0, "walk_forward": {"train": 40, "test": 20}},
            "oos_returns": ret.to_frame("returns"),
            "oos_equity": (1 + ret).cumprod().to_frame("equity"),
        }
        return _receipt("S4", verdict, metrics={"oos_sharpe": 1.2, "n_folds": 2.0}, params={}), art
    if stage is Stage.S5:
        logits = rng.normal(1.0 if variant != "overfit" else -0.2, 1.0, 500)
        pbo = float((logits <= 0).mean())
        d = {"sharpe": 0.08, "psr": 0.99, "sr0": 0.03, "dsr": 0.97, "n_trials": 9, "var_sharpe": 0.001, "skew": -0.2,
             "kurtosis": 3.4, "n_obs": 80, "pbo": pbo, "pbo_combinations": 500, "cscv_blocks": 16, "max_pbo": 0.2, "min_dsr": 0.95}
        art = {"pbo_logits": pd.DataFrame({"logit": logits}), "dsr_inputs": d}
        if variant == "stability":  # a receipt that carries the variance floor and the parameter stability check
            d.update(var_sharpe_grid=0.0004, var_sharpe_estimator=0.001, var_sharpe_source="estimator")
            art["param_stability"] = {
                "status": "FAIL", "reasons": ["parameter stability: neighbour check failed"], "skipped": ["mode"],
                "unchecked": [{"dimension": "lag", "value": None, "reason": "no distinct neighbour value"}],
                "selected": {"label": "c1", "picks": {"c1": 2}, "n_folds": 2, "params": {"fast": 10, "slow": 40}},
                "neighbour": {"applicable": True, "meaningful": True, "source": "grid", "selected_sharpe": 1.4,
                              "mean_neighbour_sharpe": 0.2, "pooled_mean_ratio": 0.143, "ratio": 0.143,
                              "worst_parameter": "fast", "threshold": 0.5, "passed": False,
                              "n_neighbours": 2, "undefined_as_zero": 0, "note": "",
                              "per_parameter": [{"dimension": "fast", "kind": "numeric", "n_neighbours": 2,
                                                 "mean_sharpe": 0.2, "ratio": 0.143, "sources": ["generated"]}],
                              "neighbours": [{"dimension": "fast", "value": 9, "sharpe": 0.1, "source": "generated", "distance": 0.1},
                                             {"dimension": "fast", "value": 11, "sharpe": 0.3, "source": "generated", "distance": 0.1}]},
                "plateau": {"applicable": True, "meaningful": True, "reference_sharpe": 1.4, "closeness": 0.7, "share": 0.0,
                            "pass_mark": 0.5, "passed": False, "n_close": 0, "n_neighbours": 2, "note": "",
                            "global_info": {"applicable": True, "best_sharpe": 1.4, "closeness": 0.7, "share": 0.11,
                                            "n_close": 1, "n_combinations": 9, "note": "information only"}},
                "diagnostics": {"parameters": [{"dimension": "fast", "pairs": 6, "mean_abs_sharpe_gap": 0.9,
                                                "mean_correlation": 0.4}], "most_fragile": "fast", "note": ""},
                "thresholds": {"closeness": 0.7, "plateau_share": 0.5, "neighbour_ratio": 0.5, "steps": [0.1, 0.2]},
                "thresholds_source": "frozen in the hypothesis", "trial_count_note": "not counted as trials",
            }
        return _receipt("S5", verdict, metrics={"pbo": pbo, "dsr": 0.97}, params={}), art
    if stage is Stage.S6:
        idx = _idx(80)
        reps = 300
        samples = pd.DataFrame({
            "sharpe": rng.normal(1.2, 0.5, reps), "cagr": rng.normal(0.15, 0.08, reps),
            "max_drawdown": -np.abs(rng.normal(0.08, 0.03, reps)),
        })
        if variant == "nan":
            samples.loc[:5, "sharpe"] = np.nan
        def pct(col):
            v = samples[col].dropna().to_numpy()
            return [{"q": q, "value": float(np.percentile(v, q))} for q in (1, 5, 25, 50, 75, 95, 99)]
        summ = {"level": 0.95, "reps": reps, "seed": 1, "mean_block_length": 5.0, "metrics": {
            k: {"point": float(samples[k].mean()), "ci": [float(np.nanpercentile(samples[k], 2.5)), float(np.nanpercentile(samples[k], 97.5))],
                "percentiles": pct(k), "undefined_share": float(samples[k].isna().mean())}
            for k in ("sharpe", "cagr", "max_drawdown")}}
        base = (1 + pd.Series(rng.normal(0.001, 0.01, 80), index=idx)).cumprod()
        eq = pd.DataFrame({"base": base, "stressed": base * np.linspace(1, 0.97, 80)})
        return _receipt("S6", verdict, metrics={"cost_stress_multiplier": 2.0, "sharpe_ci_low": 0.3}, params={}), {
            "bootstrap_samples": samples, "bootstrap_summary": summ, "equity_compare": eq}
    raise ValueError(stage)
