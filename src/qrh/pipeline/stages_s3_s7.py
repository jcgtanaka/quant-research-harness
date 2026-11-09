"""S3 (in-sample search), S4 (walk-forward), S5 (overfitting statistics),
S6 (uncertainty and costs) and the S7 report.

Fail closed: anything that cannot be computed or verified is BLOCKED. Verdicts
come from deterministic code and the thresholds frozen in the hypothesis.
"""

from __future__ import annotations

import inspect
import itertools
import math
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrh.adapters.base import DataView
from qrh.adapters.loader import AdapterLoadError, load_adapter
from qrh.hashing import hash_json, sha256_file
from qrh.llm.cache import LLMCacheCorrupt, LLMCacheMiss
from qrh.llm.variants import VARIANT_KEY, resolve_variants
from qrh.memory.catalog import Catalog, load_indexed_frame
from qrh.pipeline.artifacts import Artifacts
from qrh.pipeline.receipt import Receipt, Stage, Verdict, collect_environment, load_receipt
from qrh.pipeline.llm_support import display_params, receipt_bits as llm_receipt_bits
from qrh.pipeline.stages import MISS_TEXT, StageContext, make_receipt
from qrh.pipeline.stages_s1_s2 import _dataset_hashes
from qrh.methods.registry import Method
from qrh.validation.min_track_record import min_track_record_length
from qrh.validation.metrics import annualized_sharpe, cagr, max_drawdown, sharpe
from qrh.validation.overfit import overfitting_statistics
from qrh.validation.param_stability import StabilityBlocked, StabilityThresholds
from qrh.validation.wfo import WalkForwardResult

DEFAULT_BOOTSTRAP_SEED = 20240601
IS_RETURNS_FILE = "is_returns.parquet"
OOS_RETURNS_FILE = "oos_returns.parquet"
COMBO_METRICS = ("sharpe_annualized", "sharpe_per_period", "cagr", "max_drawdown", "n_obs")
BOOTSTRAP_PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


class StageBlocked(Exception):
    """A stage cannot run or cannot trust its inputs; carries the reasons."""

    def __init__(self, *reasons: str):
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


def _plain(obj: Any) -> Any:
    """JSON-safe copy (numpy scalars become Python numbers)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj.item() if hasattr(obj, "item") else obj


def _metrics(**values: float | None) -> dict[str, float]:
    """Finite floats only; undefined (None) values are left out."""
    out: dict[str, float] = {}
    for k, v in values.items():
        if v is None:
            continue
        if not math.isfinite(v):
            raise StageBlocked(f"metric {k} is not finite")
        out[k] = float(v)
    return out


def _threshold(trial: dict[str, Any], name: str, integer: bool = False) -> float:
    v = trial["thresholds"].get(name)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise StageBlocked(f"threshold {name!r} is missing or not a number in the registered hypothesis")
    if integer and int(v) != v:
        raise StageBlocked(f"threshold {name!r} must be an integer, got {v!r}")
    return int(v) if integer else float(v)


def grid_combos(space: dict[str, list[Any]], variants: list[dict[str, Any]] | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Full cartesian grid over sorted parameter names: (combo_hash, params) in a fixed order.

    A strategy that uses a language model multiplies the grid by its variants (prompt, model and params choices):
    each (parameters, variant) is one combination, with the variant under the ``llm_variant`` key.
    """
    keys = sorted(space)
    out = []
    for variant in variants or [None]:
        for values in itertools.product(*(space[k] for k in keys)):
            params = dict(zip(keys, values))
            if variant is not None:
                params[VARIANT_KEY] = variant
            out.append((hash_json(params), params))
    return out


@dataclass
class Env:
    trial: dict[str, Any]
    adapter: Any
    frame: pd.DataFrame
    view: DataView
    start: pd.Timestamp
    end: pd.Timestamp
    bars: pd.DatetimeIndex
    data_hashes: dict[str, str]
    ppy: float
    ppy_source: str
    grid: list[tuple[str, dict[str, Any]]]
    adapter_code_hash: str = ""

    @property
    def windows(self) -> dict[str, tuple[str, str]]:
        return {k: (a, b) for k, (a, b) in self.trial["windows"].items()}

    @property
    def params_by_label(self) -> dict[str, dict[str, Any]]:
        return dict(self.grid)


def _prepare(ctx: StageContext) -> Env:
    trial = ctx.registry.get_trial(ctx.trial_id)
    if trial is None:
        raise StageBlocked("no hypothesis registered for this trial")
    missing = [
        k for k, v in (
            ("adapter", trial["adapter"]), ("datasets", trial["datasets"]),
            ("windows.development", trial["windows"].get("development")),
        ) if not v
    ]
    if missing:
        raise StageBlocked(f"hypothesis does not declare: {', '.join(missing)}")
    base = Path(trial["base_dir"]).resolve() if trial["base_dir"] else None
    llm = trial.get("llm")
    try:
        loaded = load_adapter(trial["adapter"], base, dependencies=(llm or {}).get("prompt_files") or ())
        variants = resolve_variants(llm, base) if llm else []
        if llm:
            ctx.method("llm.replay_cache")  # recorded in the receipt of every stage that can replay calls
        adapter = loaded.adapter
        needed = {r.name for r in adapter.required_data()}
        space = adapter.parameter_space()
    except AdapterLoadError as exc:
        raise StageBlocked(str(exc)) from exc
    except Exception as exc:
        raise StageBlocked(f"adapter metadata unavailable: {type(exc).__name__}: {exc}") from exc
    declared = {d["name"] for d in trial["datasets"]}
    if needed - declared:
        raise StageBlocked(f"adapter requires datasets not declared in the hypothesis: {sorted(needed - declared)}")
    with closing(Catalog(ctx.registry.path)) as cat:
        hashes, reasons = _dataset_hashes(cat, trial["datasets"])
        primary = trial["datasets"][0]
        rec = cat.get_dataset(primary["name"])
    if reasons or rec is None:
        raise StageBlocked(*(reasons or ["primary dataset unknown"]))
    try:
        frame = load_indexed_frame(rec["path"])
        tz = frame.index.tz
        start, end = (
            pd.Timestamp(t).tz_localize(tz) if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t)
            for t in trial["windows"]["development"]
        )
        view = DataView(frame, primary.get("publication_column"))
    except Exception as exc:
        raise StageBlocked(f"dataset cannot be loaded: {type(exc).__name__}: {exc}") from exc
    bars = frame.index[(frame.index >= start) & (frame.index <= end)]
    if len(bars) < 2:
        raise StageBlocked(f"development window has {len(bars)} bar(s)")
    if trial["periods_per_year"] is not None:
        ppy, source = float(trial["periods_per_year"]), "hypothesis"
    else:
        inferred = ctx.method("metrics.periods_per_year").entry(bars)
        if inferred is None:
            raise StageBlocked(
                "periods_per_year cannot be inferred from the timestamps; set periods_per_year in the hypothesis"
            )
        ppy, source = inferred, "observed timestamp density"
    return Env(trial, adapter, frame, view, start, end, bars, hashes, ppy, source, grid_combos(space, variants), loaded.code_hash)


def _save_frame(ctx: StageContext, name: str, df: pd.DataFrame) -> tuple[str, str]:
    rel = Path("trials") / ctx.trial_id / name
    path = ctx.home / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return rel.as_posix(), sha256_file(path)


def _load_verified(ctx: StageContext, stage: Stage, rel_key: str, hash_key: str) -> tuple[pd.DataFrame, Receipt]:
    """Load a stage's stored parquet file after checking the receipt chain and the file hash."""
    entry = ctx.registry.get_receipt_entry(ctx.trial_id, stage.value)
    if entry is None:
        raise StageBlocked(f"no receipt for {stage.value}")
    try:
        receipt = load_receipt(entry["path"])
    except Exception as exc:
        raise StageBlocked(f"receipt for {stage.value} cannot be verified: {exc}") from exc
    if receipt.receipt_hash() != entry["receipt_hash"]:
        raise StageBlocked(f"receipt for {stage.value} does not match the registry index")
    path = ctx.home / receipt.params[rel_key]
    try:
        actual = sha256_file(path)
        df = pd.read_parquet(path)
    except Exception as exc:
        raise StageBlocked(f"{stage.value} data file cannot be read: {type(exc).__name__}: {exc}") from exc
    if actual != receipt.params[hash_key]:
        raise StageBlocked(f"{stage.value} data file does not match the hash in its receipt")
    return df, receipt


def _load_is_matrix(ctx: StageContext, env: Env) -> pd.DataFrame:
    df, _ = _load_verified(ctx, Stage.S3, "is_returns_path", "is_returns_hash")
    if list(df.columns) != [h for h, _ in env.grid] or not df.index.equals(env.bars):
        raise StageBlocked("stored in-sample returns do not match the adapter's current grid or window")
    return df


def _load_oos(ctx: StageContext) -> tuple[pd.Series, Receipt]:
    df, receipt = _load_verified(ctx, Stage.S4, "oos_returns_path", "oos_returns_hash")
    return df["returns"], receipt


def _wf_spec(env: Env, method: Method) -> Any:
    """The hypothesis' walk-forward spec, checked by the selected s4.walkforward method against the window."""
    raw = env.trial["walk_forward"]
    if raw is None:
        raise StageBlocked("hypothesis declares no walk_forward spec")
    try:
        assert method.validate is not None
        method.validate(raw, len(env.bars))
        return raw
    except ValueError as exc:
        raise StageBlocked(str(exc)) from exc


def _guarded(ctx: StageContext, stage: Stage, body) -> Receipt:
    try:
        receipt = body()
        trial = ctx.registry.get_trial(ctx.trial_id)
        params, diagnostics = llm_receipt_bits(trial) if trial else ({}, {})
        if params:  # a strategy that uses a language model: the cache entries this stage used, and the call counts
            receipt = receipt.model_copy(update={
                "params": {**receipt.params, **params}, "diagnostics": {**receipt.diagnostics, **diagnostics}})
        return receipt
    except LLMCacheMiss:
        return make_receipt(ctx, stage, Verdict.BLOCKED, [MISS_TEXT.format(trial=ctx.trial_id)])
    except LLMCacheCorrupt as exc:
        return make_receipt(ctx, stage, Verdict.BLOCKED, [str(exc)])
    except StageBlocked as exc:
        return make_receipt(ctx, stage, Verdict.BLOCKED, exc.reasons)
    except ValueError as exc:  # input the statistics refuse (never turned into a verdict)
        return make_receipt(ctx, stage, Verdict.BLOCKED, [f"{type(exc).__name__}: {exc}"])


class InSampleSearch:
    """S3: evaluate every combination of the declared grid on the development window."""

    id = Stage.S3

    def run(self, ctx: StageContext) -> Receipt:
        return _guarded(ctx, self.id, lambda: self._run(ctx))

    def _run(self, ctx: StageContext) -> Receipt:
        env = _prepare(ctx)
        declared = env.trial["parameter_space_size"]
        if len({h for h, _ in env.grid}) != len(env.grid):
            raise StageBlocked("parameter grid contains duplicate combinations")
        if len(env.grid) != declared:
            raise StageBlocked(
                f"adapter parameter grid has {len(env.grid)} combinations but the hypothesis declared {declared}"
            )
        columns: dict[str, np.ndarray] = {}
        rows: list[tuple[str, dict[str, Any], dict[str, float]]] = []
        for label, params in env.grid:
            try:
                res = env.adapter.run(params, env.view, env.start, env.end)
                ret = res.returns
                if not isinstance(ret, pd.Series) or not ret.index.equals(env.bars):
                    raise ValueError("returns must be a Series covering exactly the bars of the development window")
                values = ret.to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError("returns contain NaN or inf")
            except (LLMCacheMiss, LLMCacheCorrupt):
                raise  # the stage is BLOCKED with the record hint, not with a generic "combination failed"
            except Exception as exc:
                raise StageBlocked(f"combination {_plain(display_params(params))} failed: {type(exc).__name__}: {exc}") from exc
            columns[label] = values
            rows.append((label, params, _metrics(
                sharpe_annualized=annualized_sharpe(values, env.ppy), sharpe_per_period=sharpe(values),
                cagr=cagr(values, env.ppy), max_drawdown=max_drawdown(values), n_obs=float(values.size),
            )))
        ctx.registry.clear_trial_combos(ctx.trial_id, is_oos=False)
        for _, params, metrics in rows:
            ctx.registry.record_trial(ctx.trial_id, _plain(params), metrics)
        matrix = pd.DataFrame(columns, index=env.bars)
        rel, digest = _save_frame(ctx, IS_RETURNS_FILE, matrix)
        arts = Artifacts(ctx, self.id)
        arts.add_existing("is_returns", rel, digest)
        arts.json("combos", {
            "param_names": sorted({k for _, params, _ in rows for k in display_params(params)}),
            "periods_per_year": env.ppy,
            "combos": [
                {"combo": label, "params": _plain(display_params(params)), **{k: m.get(k) for k in COMBO_METRICS}}
                for label, params, m in rows
            ],
        })
        sharpes = [m["sharpe_annualized"] for _, _, m in rows if "sharpe_annualized" in m]
        return make_receipt(
            ctx, self.id, Verdict.PASS, [],
            data_hashes=env.data_hashes,
            windows=env.windows,
            trial_family_count=ctx.registry.family_trial_count(env.trial["family_id"]),
            metrics=_metrics(
                grid_size=len(env.grid), evaluated_count=len(rows), n_bars=len(env.bars),
                best_is_sharpe=max(sharpes) if sharpes else None,
            ),
            params={
                "is_returns_path": rel, "is_returns_hash": digest, "periods_per_year": env.ppy,
                "periods_per_year_source": env.ppy_source, "adapter": env.trial["adapter"],
                "adapter_code_hash": env.adapter_code_hash,
                "undefined_sharpe_count": len(rows) - len(sharpes),
                "artifacts": arts.as_params(),
            },
        )


def _fold_report(res: WalkForwardResult, bars: pd.DatetimeIndex, ppy: float) -> list[dict[str, Any]]:
    out = []
    for fr in res.folds:
        f = fr.fold
        out.append({
            "fold": f.index,
            "train": [bars[f.train_start].isoformat(), bars[f.train_end - 1].isoformat()],
            "test": [bars[f.test_start].isoformat(), bars[f.test_end - 1].isoformat()],
            "combo_hash": fr.combo_hash, "params": _plain(display_params(fr.params)),
            "train_sharpe": fr.train_sharpe, "test_sharpe": annualized_sharpe(fr.test_returns.to_numpy(), ppy),
        })
    return out


class WalkForward:
    """S4: select per fold on train data only, stitch the out-of-sample returns."""

    id = Stage.S4

    def run(self, ctx: StageContext) -> Receipt:
        return _guarded(ctx, self.id, lambda: self._run(ctx))

    def _run(self, ctx: StageContext) -> Receipt:
        env = _prepare(ctx)
        wf = ctx.method("s4.walkforward")
        spec = _wf_spec(env, wf)
        min_sharpe = _threshold(env.trial, "min_oos_sharpe")
        matrix = _load_is_matrix(ctx, env)
        try:
            res = wf.entry(env.adapter, env.view, env.bars, matrix, env.params_by_label, spec, env.ppy)
        except (LLMCacheMiss, LLMCacheCorrupt):
            raise
        except Exception as exc:
            raise StageBlocked(f"walk-forward failed: {type(exc).__name__}: {exc}") from exc
        oos = res.oos_returns.to_numpy(dtype=float)
        if not np.isfinite(oos).all():
            raise StageBlocked("out-of-sample returns contain NaN or inf")
        oos_sharpe = annualized_sharpe(oos, env.ppy)
        if oos_sharpe is None:
            raise StageBlocked("out-of-sample Sharpe ratio is undefined (no variation in the stitched returns)")
        rel, digest = _save_frame(ctx, OOS_RETURNS_FILE, res.oos_returns.to_frame("returns"))
        folds = _plain(_fold_report(res, env.bars, env.ppy))
        arts = Artifacts(ctx, self.id)
        arts.add_existing("oos_returns", rel, digest)
        arts.json("folds", {
            "folds": folds, "window": [env.bars[0].isoformat(), env.bars[-1].isoformat()],
            "periods_per_year": env.ppy, "walk_forward": _plain(env.trial["walk_forward"]),
        })
        arts.frame("oos_equity", (1.0 + res.oos_returns).cumprod().to_frame("equity"))
        ok = oos_sharpe >= min_sharpe
        return make_receipt(
            ctx, self.id, Verdict.PASS if ok else Verdict.FAIL,
            [] if ok else [f"out-of-sample Sharpe {oos_sharpe:.3f} is below min_oos_sharpe {min_sharpe:g}"],
            data_hashes=env.data_hashes,
            windows={**env.windows, "oos": (res.oos_returns.index[0].isoformat(), res.oos_returns.index[-1].isoformat())},
            metrics=_metrics(
                oos_sharpe=oos_sharpe, oos_cagr=cagr(oos, env.ppy), oos_max_drawdown=max_drawdown(oos),
                n_folds=len(res.folds), oos_bars=len(oos), min_oos_sharpe=min_sharpe,
            ),
            params={
                "walk_forward": _plain(env.trial["walk_forward"]), "periods_per_year": env.ppy,
                "folds": folds,
                "oos_returns_path": rel, "oos_returns_hash": digest, "artifacts": arts.as_params(),
            },
        )


# Informational defaults for the minimum track record length reported by S5. They are not thresholds:
# no verdict depends on them or on the resulting number.
MIN_TRL_BENCHMARK = 0.0
MIN_TRL_CONFIDENCE = 0.95


def min_track_record_info(sharpe: float, skew: float, kurt: float) -> tuple[int | None, str | None]:
    """Periods needed for PSR >= MIN_TRL_CONFIDENCE, or (None, reason) when the inputs do not allow one."""
    try:
        return min_track_record_length(sharpe, MIN_TRL_BENCHMARK, skew, kurt, MIN_TRL_CONFIDENCE), None
    except ValueError as exc:
        return None, str(exc)


STABILITY_KEYS = ("stability_closeness", "stability_plateau_share", "stability_neighbour_ratio", "stability_steps")


def _stability_inputs(trial: dict[str, Any]) -> tuple[StabilityThresholds, str]:
    """The frozen stability thresholds and where they stand. A hypothesis registered before they existed has none: the
    defaults apply and the receipt says so (the readiness report asks the trader to confirm them)."""
    frozen = trial["thresholds"]
    missing = [k for k in STABILITY_KEYS if k not in frozen]
    source = ("frozen in the hypothesis" if not missing else "defaults (not frozen in the hypothesis)"
              if len(missing) == len(STABILITY_KEYS) else f"defaults for {', '.join(missing)} (not frozen in the hypothesis)")
    try:
        return StabilityThresholds.from_mapping(frozen), source
    except StabilityBlocked as exc:
        raise StageBlocked(*exc.reasons) from exc


class OverfittingStats:
    """S5: probability of backtest overfitting (CSCV), the deflated Sharpe ratio and parameter stability."""

    id = Stage.S5

    def run(self, ctx: StageContext) -> Receipt:
        return _guarded(ctx, self.id, lambda: self._run(ctx))

    def _run(self, ctx: StageContext) -> Receipt:
        env = _prepare(ctx)
        max_pbo = _threshold(env.trial, "max_pbo")
        min_dsr = _threshold(env.trial, "min_dsr")
        blocks = _threshold(env.trial, "cscv_blocks", integer=True)
        matrix = _load_is_matrix(ctx, env)
        oos, s4 = _load_oos(ctx)
        family_count = ctx.registry.family_trial_count(env.trial["family_id"])
        stab_thresholds, stab_source = _stability_inputs(env.trial)
        stats = overfitting_statistics(
            matrix, oos.to_numpy(), max(family_count, len(env.grid)), blocks,
            ctx.method("s5.pbo").entry, ctx.method("s5.dsr").entry,
        )
        pbo, dsr, var_sharpe, n_trials = stats.pbo, stats.dsr, stats.var_sharpe, stats.n_trials
        stab = self._stability(ctx, env, matrix, oos, s4, stab_thresholds, stab_source)
        arts = Artifacts(ctx, self.id)
        if pbo is not None:
            arts.frame("pbo_logits", pd.DataFrame({"logit": pbo.logits}))
        arts.json("dsr_inputs", {
            "sharpe": dsr.sharpe, "psr": dsr.psr, "sr0": dsr.sr0, "dsr": dsr.dsr, "n_trials": dsr.n_trials,
            "var_sharpe": dsr.var_sharpe, "skew": dsr.skew, "kurtosis": dsr.kurtosis, "n_obs": dsr.n_obs,
            "pbo": None if pbo is None else pbo.pbo, "pbo_combinations": None if pbo is None else pbo.n_combinations,
            "cscv_blocks": None if pbo is None else pbo.n_blocks,
            "max_pbo": max_pbo, "min_dsr": min_dsr,
            "pbo_note": stats.pbo_note, "var_sharpe_source": stats.var_sharpe_source,
            "var_sharpe_grid": stats.var_sharpe_grid, "var_sharpe_estimator": stats.var_sharpe_estimator,
        })
        arts.json("param_stability", stab)
        reasons = []
        notes = [
            f"Sharpe variance V used in the deflated Sharpe: {var_sharpe:.3g} (source: {stats.var_sharpe_source}; grid "
            + ("n/a, one combination" if stats.var_sharpe_grid is None else f"{stats.var_sharpe_grid:.3g}")
            + f", sampling estimator {stats.var_sharpe_estimator:.3g}; V is the larger of the two)"
        ]
        if pbo is None:  # one combination: nothing was selected, so there is nothing for PBO to measure
            notes.append(f"PBO {stats.pbo_note}; the deflated Sharpe ratio decides, with {n_trials} trial(s) charged")
        elif not pbo.pbo <= max_pbo:
            reasons.append(f"PBO {pbo.pbo:.3f} exceeds max_pbo {max_pbo:g}")
        if not dsr.dsr >= min_dsr:
            reasons.append(f"DSR {dsr.dsr:.3f} is below min_dsr {min_dsr:g}")
        if stab["status"] == "FAIL":
            reasons += stab["reasons"]
        elif stab["status"] == "NOT_APPLICABLE":
            notes += stab["reasons"]
        trl_periods, trl_reason = min_track_record_info(dsr.sharpe, dsr.skew, dsr.kurtosis)  # information only
        pbo_metrics = {} if pbo is None else dict(
            pbo=pbo.pbo, pbo_mean_oos_sharpe_of_winner=pbo.mean_oos_sharpe, pbo_prob_oos_loss=pbo.prob_oos_loss,
            pbo_combinations=pbo.n_combinations, pbo_rows_dropped=pbo.rows_dropped,
        )
        pbo_params = (
            {"pbo": {"applicable": False, "note": stats.pbo_note}, "var_sharpe_source": stats.var_sharpe_source}
            if pbo is None else {"cscv_blocks": pbo.n_blocks, "pbo_rows_used": pbo.rows_used}
        )
        nb, pl = stab["neighbour"], stab["plateau"]
        stab_metrics = _metrics(
            param_selected_oos_sharpe=nb["selected_sharpe"], param_mean_neighbour_sharpe=nb["mean_neighbour_sharpe"],
            param_neighbour_ratio=nb["ratio"], param_n_neighbours=float(nb["n_neighbours"]),
            param_plateau_share=pl["share"], param_pooled_neighbour_ratio=nb["pooled_mean_ratio"],
            param_global_plateau_share=pl["global_info"]["share"],
        )
        return make_receipt(
            ctx, self.id, Verdict.FAIL if reasons else Verdict.PASS, reasons + notes,
            data_hashes=env.data_hashes,
            windows=env.windows,
            trial_family_count=n_trials,
            metrics=_metrics(
                dsr=dsr.dsr, psr_zero=dsr.psr, sr0=dsr.sr0, oos_sharpe_per_period=dsr.sharpe,
                var_sharpe=dsr.var_sharpe, var_sharpe_grid=stats.var_sharpe_grid,
                var_sharpe_estimator=stats.var_sharpe_estimator, oos_skew=dsr.skew, oos_kurtosis=dsr.kurtosis,
                max_pbo=max_pbo, min_dsr=min_dsr, **pbo_metrics, **stab_metrics,
            ),
            params={
                **pbo_params, "n_strategies": matrix.shape[1],
                "family_trial_count": family_count, "grid_size": len(env.grid), "n_trials_used": n_trials,
                "oos_observations": dsr.n_obs, "artifacts": arts.as_params(),
                "var_sharpe": {"grid": stats.var_sharpe_grid, "estimator": stats.var_sharpe_estimator,
                               "used": stats.var_sharpe_source},
                "param_stability": stab,
                "min_track_record_periods": trl_periods,
                **({"min_track_record_reason": trl_reason} if trl_reason else {}),
            },
        )


    @staticmethod
    def _stability(ctx: StageContext, env: Env, matrix: pd.DataFrame, oos: pd.Series, s4: Receipt,
                   thresholds: StabilityThresholds, source: str) -> dict[str, Any]:
        """The parameter stability check on the stored returns; one combination's neighbours are run, never recorded."""
        method = ctx.method("s5.param_stability")
        picks = [str(f["combo_hash"]) for f in s4.params.get("folds") or []]
        try:
            result = method.entry(
                grid=env.grid, matrix=matrix, oos_index=oos.index, picks=picks, ppy=env.ppy, thresholds=thresholds,
                adapter=env.adapter, view=env.view, start=env.start, end=env.end, declared=env.trial.get("stability"),
            )
        except StabilityBlocked as exc:
            raise StageBlocked(*[f"parameter stability: {r}" for r in exc.reasons]) from exc
        doc = result.to_dict()
        doc["thresholds_source"] = source
        doc["selected"]["params"] = _plain(display_params(env.params_by_label[result.selected["label"]]))
        return _plain(doc)


class UncertaintyAndCosts:
    """S6: bootstrap intervals on the OOS returns and a cost-stressed walk-forward."""

    id = Stage.S6

    def run(self, ctx: StageContext) -> Receipt:
        return _guarded(ctx, self.id, lambda: self._run(ctx))

    def _run(self, ctx: StageContext) -> Receipt:
        env = _prepare(ctx)
        wf = ctx.method("s4.walkforward")
        spec = _wf_spec(env, wf)
        min_lower = _threshold(env.trial, "min_sharpe_lower_ci")
        min_stressed = _threshold(env.trial, "min_cost_stressed_sharpe")
        reps = _threshold(env.trial, "bootstrap_reps", integer=True)
        level = _threshold(env.trial, "ci_level")
        mult = _threshold(env.trial, "cost_stress_multiplier")
        if mult < 1:
            raise StageBlocked(f"cost_stress_multiplier must be >= 1, got {mult:g}")
        try:
            accepts = "cost_multiplier" in inspect.signature(env.adapter.run).parameters
        except (TypeError, ValueError):
            accepts = False
        if not accepts:
            raise StageBlocked("adapter does not support cost stress (run() has no cost_multiplier parameter)")
        oos, s4 = _load_oos(ctx)
        matrix = _load_is_matrix(ctx, env)
        seed = int(ctx.seeds.get("bootstrap", DEFAULT_BOOTSTRAP_SEED))
        ci = ctx.method("s6.bootstrap").entry(oos.to_numpy(), env.ppy, reps=reps, level=level, seed=seed)
        sharpe_ci = ci.intervals["sharpe"]
        if sharpe_ci is None or ci.point["sharpe"] is None:
            raise StageBlocked("bootstrap Sharpe interval is undefined (no variation in the returns)")
        try:
            stressed = wf.entry(
                env.adapter, env.view, env.bars, matrix, env.params_by_label, spec, env.ppy,
                run_kwargs={"cost_multiplier": mult},
            )
        except (LLMCacheMiss, LLMCacheCorrupt):
            raise
        except Exception as exc:
            raise StageBlocked(f"cost-stressed walk-forward failed: {type(exc).__name__}: {exc}") from exc
        if [f.combo_hash for f in stressed.folds] != [f["combo_hash"] for f in s4.params["folds"]]:
            raise StageBlocked("cost-stressed run selected different parameters than S4")
        values = stressed.oos_returns.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise StageBlocked("cost-stressed returns contain NaN or inf")
        stressed_sharpe = annualized_sharpe(values, env.ppy)
        if stressed_sharpe is None:
            raise StageBlocked("cost-stressed Sharpe ratio is undefined (no variation in the returns)")
        reasons = []
        if not sharpe_ci[0] >= min_lower:
            reasons.append(f"Sharpe lower confidence bound {sharpe_ci[0]:.3f} is below min_sharpe_lower_ci {min_lower:g}")
        if not stressed_sharpe >= min_stressed:
            reasons.append(
                f"cost-stressed (x{mult:g}) Sharpe {stressed_sharpe:.3f} is below min_cost_stressed_sharpe {min_stressed:g}"
            )
        arts = Artifacts(ctx, self.id)
        arts.frame("bootstrap_samples", pd.DataFrame({k: ci.samples[k] for k in ("sharpe", "cagr", "max_drawdown")}))
        arts.json("bootstrap_summary", {
            "level": level, "reps": reps, "seed": seed, "mean_block_length": ci.mean_block_length,
            "metrics": {
                name: {
                    "point": ci.point[name],
                    "ci": list(ci.intervals[name]) if ci.intervals[name] else None,
                    "percentiles": [
                        {"q": q, "value": float(np.nanpercentile(sample, q)) if np.isfinite(sample).any() else None}
                        for q in BOOTSTRAP_PERCENTILES
                    ],
                    "undefined_share": float(np.isnan(sample).mean()),
                }
                for name, sample in ((n, ci.samples[n]) for n in ("sharpe", "cagr", "max_drawdown"))
            },
        })
        arts.frame("equity_compare", pd.DataFrame({
            "base": (1.0 + oos).cumprod(),
            "stressed": (1.0 + stressed.oos_returns.reindex(oos.index)).cumprod(),
        }))
        intervals = {f"{k}_ci_{side}": v[i] for k, v in ci.intervals.items() if v for i, side in enumerate(("low", "high"))}
        return make_receipt(
            ctx, self.id, Verdict.FAIL if reasons else Verdict.PASS, reasons,
            data_hashes=env.data_hashes,
            windows=env.windows,
            environment=collect_environment({**ctx.seeds, "bootstrap": seed}),
            metrics=_metrics(
                oos_sharpe=ci.point["sharpe"], oos_cagr=ci.point["cagr"], oos_max_drawdown=ci.point["max_drawdown"],
                cost_stressed_sharpe=stressed_sharpe, cost_stress_multiplier=mult, bootstrap_reps=reps,
                ci_level=level, min_sharpe_lower_ci=min_lower, min_cost_stressed_sharpe=min_stressed, **intervals,
            ),
            params={
                "bootstrap_seed": seed, "mean_block_length": ci.mean_block_length, "periods_per_year": env.ppy,
                "cost_stressed_oos_bars": int(values.size), "artifacts": arts.as_params(),
            },
        )
