"""The methods that ship with the harness: version 1 of every current implementation, plus experiments.

Adding a method means adding a new version here (never editing an old one), with its references. A new
version starts ``experimental``; only a measured benchmark win and a human approval make it the default
(see ``docs/methods.md``). Identifiers (DOI, arXiv id, URL) are recorded only when checked against the
publisher record; otherwise the field stays empty and ``verified`` is false.
"""

from __future__ import annotations

from qrh.methods.registry import Method, MethodRegistry, Reference

ADDED_IN = "0.0.1"

DSR_REF = Reference(
    "Bailey, D. H. and Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, "
    "Backtest Overfitting and Non-Normality. The Journal of Portfolio Management 40(5), 94-107.",
    doi="10.3905/jpm.2014.40.5.094", verified=True,
)
PBO_REF = Reference(
    "Bailey, D. H., Borwein, J. M., Lopez de Prado, M. and Zhu, Q. J. (2017). The probability of backtest "
    "overfitting. The Journal of Computational Finance 20(4).",
    doi="10.21314/JCF.2016.322", verified=True,
)
BOOTSTRAP_REF = Reference(
    "Politis, D. N. and Romano, J. P. (1994). The Stationary Bootstrap. Journal of the American Statistical "
    "Association 89(428), 1303-1313.",
    doi="10.1080/01621459.1994.10476870", verified=True,
)
AFML_REF = Reference(
    "Lopez de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Chapter 7: Cross-Validation "
    "in Finance (purging and embargo).",
)

CARVER_REF = Reference(
    "Carver, R. (2015). Systematic Trading: A unique new method for designing trading and investing systems. "
    "Harriman House. (Source of the plateau idea only; the neighbour check and its numbers are this harness's own.)",
)

_RULE_NOTES = {
    "schema": "Required columns exist with an accepted dtype kind.",
    "range": "Values within min/max, finite, optionally strictly positive.",
    "relationship": "A column stays within a relative tolerance of a reference column.",
    "null_rate_drift": "The recent null share has not risen above the history by more than max_increase.",
    "gaps": "No duplicate, unsorted or missing timestamps at the (declared or modal) frequency.",
    "timezone": "The index is timezone-aware in the declared zone.",
}


def builtin_registry() -> MethodRegistry:
    from qrh.checks.data_quality import RULE_TYPES
    from qrh.checks.lookahead import lint_files
    from qrh.checks.pnl_linear_config import validate_reconstruction
    from qrh.checks.pnl_linear_run import ALPHA, CAPITAL_BASE, check_pnl_linear
    from qrh.checks import perturbation as pert
    from qrh.checks.pnl_reconstruction import DEFAULT_COST_BOUND, check_pnl_reconstruction, validate_execution
    from qrh.checks.same_bar import INFERRED_LAG, MAX_CANDIDATES, check_same_bar_execution
    from qrh.checks.sanity import check_sanity
    from qrh.checks.truncation import DEFAULT_ATOL, DEFAULT_CUTOFFS, run_truncation
    from qrh.llm.gate import evaluate_gate
    from qrh.llm.runtime import complete
    from qrh.validation.bootstrap import bootstrap_ci
    from qrh.validation.deflated_sharpe import deflated_sharpe
    from qrh.validation.metrics import infer_periods_per_year
    from qrh.validation.param_stability import run_param_stability
    from qrh.validation.pbo_cscv import DEFAULT_BLOCKS, pbo_cscv
    from qrh.validation.wfo import run_rolling_anchored, validate_rolling_anchored
    from qrh.validation.wfo_purged import DEFAULT_EMBARGO, DEFAULT_PURGE, run_purged_embargo, validate_purged_embargo

    reg = MethodRegistry()
    for rtype, fn in RULE_TYPES.items():
        reg.register(Method(
            id=f"s1.quality.{rtype}", version=f"{rtype}@1", stage="S1", kind="rule", entry=fn,
            parameters={"rule": "mapping of the rule fields as written in the rules file"},
            status="default", references=(), added_in=ADDED_IN, notes=_RULE_NOTES.get(rtype, ""),
        ))
    reg.register(Method(
        id="s2.lookahead_lint", version="ast_rules@1", stage="S2", kind="check", entry=lint_files,
        parameters={"rules": ["LA001 negative shift", "LA002 centered rolling", "LA003 backward fill",
                              "LA004 iloc[-1] in a loop", "LA005 full-sample statistic (warning)",
                              "LA006 np.roll with a negative shift (a named constant is resolved)",
                              "LA007 shift or np.roll amount that is not a literal number (warning)"]},
        status="default", references=(), added_in=ADDED_IN,
        notes="Static AST patterns; an early warning, not proof (the truncation and future-perturbation tests are the proof).",
    ))
    reg.register(Method(
        id="s2.truncation", version="prefix_truncation@1", stage="S2", kind="check", entry=run_truncation,
        parameters={"cutoffs": list(DEFAULT_CUTOFFS), "atol": DEFAULT_ATOL},
        status="default", references=(), added_in=ADDED_IN,
        notes="Runs the adapter on all data and on data cut before each cutoff; earlier decisions must match.",
    ))
    reg.register(Method(
        id="s2.future_perturbation", version="future_perturbation@1", stage="S2", kind="check", entry=pert.run_perturbation,
        parameters={"cutoffs": pert.DEFAULT_CUTOFFS, "threshold": "perturbation_cutoffs (default 20, 1 to 200), perturbation_change_cap (default 40, 1 to 500)",
                    "atol": DEFAULT_ATOL, "seed": pert.SEED, "row_factor_sigma": pert.SIGMA,
                    "level_shift_sigma": pert.LEVEL_SIGMA},
        status="default", references=(), added_in=ADDED_IN,
        notes="Reruns a freshly loaded adapter at every position-change bar (up to a cap), the bar before it and evenly "
              "spread cutoffs, with all data after the cutoff (and not yet published at it) replaced by shuffled, rescaled rows; positions, trades and returns up to the cutoff must be "
              "identical. One adapter run per cutoff. BLOCKED when the perturbed data never reaches the strategy. New check, "
              "no predecessor: it adds to s2.truncation and does not replace it. Strategies that declare an llm run in "
              "replay mode with a stub answer for prompts after the cutoff.",
    ))
    reg.register(Method(
        id="s2.same_bar_execution", version="same_bar@1", stage="S2", kind="check", entry=check_same_bar_execution,
        parameters={"execution": "the hypothesis `execution` block when declared", "inferred_lag_bars": INFERRED_LAG,
                    "max_candidate_columns": MAX_CANDIDATES},
        status="default", references=(), added_in=ADDED_IN,
        notes="Rebuilds returns from the reported positions and a price column with the fill one bar after the "
              "decision, and FAILs when the reported returns fit an earlier fill. BLOCKED without reported positions, "
              "INCONCLUSIVE (never PASS) when no column explains the returns. With no `execution` it tries each "
              "strictly positive numeric column. New check, no predecessor; skipped where s2.pnl_reconstruction is pinned.",
    ))
    reg.register(Method(
        id="s2.sanity", version="bounds@1", stage="S2", kind="check", entry=check_sanity,
        parameters={"bounds": "max_abs_period_return, max_gross_leverage, max_position_weight (from the hypothesis)"},
        status="default", references=(), added_in=ADDED_IN,
        notes="Return, leverage and position bounds, plus a median absolute return test for percent units.",
    ))
    reg.register(Method(
        id="s2.pnl_reconstruction", version="pnl_reconstruction@1", stage="S2", kind="check",
        entry=check_pnl_reconstruction, validate=validate_execution,
        parameters={"execution": "lag_bars (>= 1, required), price_column or price_columns, max_cost_per_turnover",
                    "max_cost_per_turnover": DEFAULT_COST_BOUND},
        status="experimental", references=(), added_in=ADDED_IN,
        notes="Recomputes returns from the reported positions and the declared price columns with an enforced "
              "execution lag; FAIL on same-bar execution, a shorter lag than declared, or a power-of-ten units "
              "mismatch; INCONCLUSIVE (never PASS) on an unexplained mismatch. Runs only where a hypothesis "
              "selects it and declares `execution`; there is no default.",
    ))
    reg.register(Method(
        id="s2.pnl_linear", version="pnl_linear@1", stage="S2", kind="check",
        entry=check_pnl_linear, validate=validate_reconstruction,
        parameters={"reconstruction": "position_units, cost_model or cost_series_column, asset_class, "
                                      "min_cost_per_turnover, publication_column, max_staleness, adjustment "
                                      "(all required); optional income, halt_flag_column and the two caps",
                    "capital_base": CAPITAL_BASE, "alpha": ALPHA},
        status="experimental", references=(), added_in=ADDED_IN,
        notes="Explains the reported PnL from lagged positions, prices and a declared cost floor, then bounds the "
              "unexplained remainder (absolute cap, share of explained PnL, positive drift) and checks mark "
              "freshness. BLOCKED when the block or a cost floor is missing, FAIL on a breached bound or a stale "
              "or future-dated mark, INCONCLUSIVE on a pending cap, a halt or too few active bars. Runs only "
              "where a hypothesis declares `reconstruction` (and `execution`); it does not replace "
              "s2.pnl_reconstruction.",
    ))
    reg.register(Method(
        id="s2.llm_gate", version="llm_gate@1", stage="S2", kind="check", entry=evaluate_gate,
        parameters={"threshold": "llm_cutoff_margin_days (default 180)",
                    "rules": ["every window starts strictly after the latest declared training cutoff plus the margin",
                              "an unknown cutoff or cutoff source BLOCKS", "live_retrieval needs evidence text",
                              "LLM001: an adapter that uses an LLM client but declares none (lint)",
                              "LLM002 (warning): provider client used outside qrh.llm.complete in a declared adapter",
                              "a replay-cache miss during a truncated run means the prompt used later data (FAIL)"]},
        status="default", references=(), added_in=ADDED_IN,
        notes="Inert for strategies that declare no llm (the declaration lint still looks for an undeclared one). "
              "A cutoff is a claim by the model's provider: the gate cannot measure what a model knows.",
    ))
    reg.register(Method(
        id="llm.replay_cache", version="replay_cache@1", stage="S2", kind="procedure", entry=complete,
        parameters={"key": "sha256 of canonical JSON {model_id, params, prompt, dataset_hash, code_hash}; as_of is audit only",
                    "modes": ["replay (default; a miss is BLOCKED)", "record"],
                    "storage": "<QRH_HOME>/llm_cache/<trial_id>/ (append-only JSONL with entry hashes, plus an index)"},
        status="default", references=(), added_in=ADDED_IN,
        notes="Every model call goes through qrh.llm.complete. Recorded answers are replayed so a verdict is "
              "reproducible; the cache hash is recorded in the receipts of S3 to S6.",
    ))
    reg.register(Method(
        id="metrics.periods_per_year", version="observed_density@1", stage="S3", kind="estimator",
        entry=infer_periods_per_year, parameters={"year_seconds": 365.25 * 86400.0},
        status="default", references=(), added_in=ADDED_IN,
        notes="(n - 1) / (span / 365.25 days); used when the hypothesis does not set periods_per_year.",
    ))
    reg.register(Method(
        id="s4.walkforward", version="rolling_anchored@1", stage="S4", kind="procedure",
        entry=run_rolling_anchored, validate=validate_rolling_anchored,
        parameters={"keys": ["train", "test", "step", "anchored"]},
        status="default", references=(), added_in=ADDED_IN,
        notes="Per-fold best in-sample Sharpe on the train span only; stitched out-of-sample test returns.",
    ))
    reg.register(Method(
        id="s4.walkforward", version="purged_embargo@1", stage="S4", kind="procedure",
        entry=run_purged_embargo, validate=validate_purged_embargo,
        parameters={"keys": ["train", "test", "step", "anchored", "purge", "embargo"],
                    "purge": DEFAULT_PURGE, "embargo": DEFAULT_EMBARGO},
        status="experimental", references=(AFML_REF,), added_in=ADDED_IN,
        notes="Rolling/anchored walk-forward with the last `purge` train bars unused for selection and the "
              "first `embargo` test bars unscored: a forward-only adaptation of purging and embargo.",
    ))
    reg.register(Method(
        id="s5.pbo", version="cscv@1", stage="S5", kind="statistic", entry=pbo_cscv,
        parameters={"n_blocks": DEFAULT_BLOCKS}, status="default", references=(PBO_REF,), added_in=ADDED_IN,
        notes="CSCV over S contiguous blocks; rank ties are averaged; PBO = share of splits with logit <= 0.",
    ))
    reg.register(Method(
        id="s5.dsr", version="bailey_lopezdeprado2014@1", stage="S5", kind="statistic", entry=deflated_sharpe,
        parameters={"n_trials": "family trial count",
                    "var_sharpe": "max(variance of in-sample per-period Sharpe ratios across the grid, sampling variance "
                                  "of the selected variant's out-of-sample Sharpe estimator); both are recorded"},
        status="default", references=(DSR_REF,), added_in=ADDED_IN,
        notes="PSR against the expected maximum Sharpe of N null trials (Euler-Mascheroni approximation). The variance "
              "floor keeps near-duplicate variants from shrinking the luck bar.",
    ))
    reg.register(Method(
        id="s5.param_stability", version="param_stability@1", stage="S5", kind="statistic", entry=run_param_stability,
        parameters={"threshold": "stability_neighbour_ratio (0.50), stability_plateau_share (0.50), "
                                 "stability_closeness (0.70), stability_steps ([0.10, 0.20]); all asked of the trader",
                    "neighbours": "one parameter at a time at each step up and down (grid points where they are at about "
                                  "that distance, else non-trial runs), never closer than the smallest step; integers round "
                                  "away; non-numeric parameters with several grid values are varied",
                    "decision": "worst parameter's neighbour ratio; local plateau share against the selected setting"},
        status="default", references=(CARVER_REF,), added_in=ADDED_IN,
        notes="Out-of-sample per-parameter neighbour check (decides a spike), local plateau share (decides a plateau) and "
              "per-parameter diagnostics; the global share against the grid's best is information only. Neighbour runs are "
              "not trials. NOT APPLICABLE only without parameters or with a non-positive selected Sharpe; BLOCKED when "
              "nothing can be nudged. New check, no predecessor.",
    ))
    reg.register(Method(
        id="s6.bootstrap", version="stationary_politis_romano1994@1", stage="S6", kind="statistic",
        entry=bootstrap_ci,
        parameters={"reps": 2000, "level": 0.95, "mean_block_length": "ceil(T ** (1/3))"},
        status="default", references=(BOOTSTRAP_REF,), added_in=ADDED_IN,
        notes="Circular stationary bootstrap with geometric blocks; percentile intervals.",
    ))
    return reg


BUILTIN = builtin_registry()
