# Contributing

Thank you for your interest in quant-research-harness (`qrh`). The project is licensed under Apache-2.0; contributions
are accepted under the same licence.

## Set up

```bash
uv sync
uv run pytest
uv run qrh version
```

Use a private state directory while you work: `export QRH_HOME=$(mktemp -d)` (it defaults to `~/.qrh`).

## Tests

`uv run pytest` runs the whole suite. Expect it to take about 4 minutes. The quick benchmark profile
(`uv run qrh bench run`) takes about 10 seconds.

## Strict TDD

Every behaviour change starts with a failing test. Write the test, watch it fail for the right reason, then change the
code until it passes. Do not weaken or delete an existing test to make a change fit; `uv run qrh audit tests --base main`
reports weakened tests in a diff.

## What must never enter the repository

- real strategy parameters, tickers, account data or private datasets;
- third-party licensed content (data, text, prompts, code);
- keys, passphrases or approval material;
- LLM replay caches (they live only in `QRH_HOME`).

Examples, fixtures and tests use synthetic data only. Nothing in the code is keyed on a symbol, a strategy name or a fixed
bar size; keep new logic general in the same way.

## Privacy and publication checks

Publication is guarded by `.pre-commit-config.yaml`:

- **gitleaks** scans for secrets, using the repository's `.gitleaks.toml`;
- **private denylist**: `scripts/denylist_check.py` checks files against a private list of terms. The list is not in the
  repository; it is supplied through the `QRH_DENYLIST` environment variable.

Do not edit these guards, the config or the scripts as part of an unrelated change. See
[docs/hooks.md](docs/hooks.md) (publication guard) for details.

## Proposing a method or evidence

Checks and statistics are versioned, cited methods in the registry (see [docs/methods.md](docs/methods.md)).

1. Add the method as a new version. It starts as experimental and is never the default on its own.
2. Cite its references.
3. Measure it on the benchmarks with known truth (`uv run qrh bench run`, and `--full` with `--compare-to default` for a
   real comparison), and include the benchmark results in your proposal.
4. Only a measured win plus a human approval promotes it (`qrh methods promote ... --bench <bench_id>`; it waits for
   `qrh approve` by the maintainer).

Claims need evidence: say how the change was measured, not only that it works.

## Pull requests

Keep changes small and focused, with the failing test first in the history where practical. Describe the behaviour change
and how you checked it. Security issues go through [SECURITY.md](SECURITY.md), not a public issue.
