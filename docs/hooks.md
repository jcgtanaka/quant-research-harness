# Hooks

Claude Code hooks are commands that receive an event as JSON on stdin and answer with an exit code.
qrh registers thirteen of them. Each is `qrh hook <name>`: a thin entry point that parses the payload and
calls library code in `src/qrh/hooks/` (one module per hook, plus `common.py` for payloads, config,
redaction and logging, and `runner.py` for dispatch, the fail-closed policy and human approvals).

Exit code 0 allows. Exit code 2 blocks: for PreToolUse the tool call does not happen; for PostToolUse
the edit already happened and stderr is fed back to the agent; for Stop the agent is told to continue;
for UserPromptSubmit the prompt is blocked. For SessionStart and UserPromptSubmit, stdout is added to
the agent's context. Any other exit code is a non-blocking error.

## The hooks

| Hook | Event | Matcher | Blocks (exit 2) | Warns / informs | Fail mode |
|---|---|---|---|---|---|
| `session-context` | SessionStart | startup, resume, clear, compact | never | up to 25 lines: security setup (the approval key fingerprint, a missing or recently changed key, a missing or changed config baseline, qrh imported from code this user can write), Stop answers the loop guard let through, open trials (final verdict not PASS, newest 10), BLOCKED datasets, overrides in the last 7 days, report render errors, and one line each when the publish guard or the runaway guard is disabled | open (always exit 0) |
| `prior-attempts` | UserPromptSubmit | none | never | when the prompt contains a trigger term, the top 3 earlier hypotheses by BM25 keyword match (trial id, verdict, family, one line of text) | open |
| `lookahead-lint` | PostToolUse | Edit, Write, MultiEdit | BLOCK findings of the look-ahead lint in a strategy file (`file:line rule message` on stderr) | WARN findings go to the log only | closed |
| `approval-guard` | PreToolUse | Bash, Read, Edit, Write, MultiEdit, NotebookEdit, Grep, Glob, WebFetch, `mcp__.*` | `qrh approve`/`config sign`/`readiness ack` by the agent; reads or writes of the keys, approvals, QRH_HOME and its ancestors; writes to config, state (including the LLM replay cache under `<QRH_HOME>/llm_cache`; reads stay allowed), settings and transcripts; interpreters naming the state in any spelling; `file:` URLs outside the working tree; approvable: Python start-up code, risky installs, scheduling (see below) | none | closed; path rules never overridable, the content stopgap and the code-path rules are approvable |
| `publish-guard` | PreToolUse | Bash, Write, Edit, MultiEdit | in a protected repository: `git push`, remote changes, `--no-verify`, hook-path changes, `gh`/package publication commands, and writes to the publication guard files (see Publish guard); approvable | the guard is disabled when `publish_guard` is absent from config | closed |
| `runaway-guard` | PreToolUse | all tools | the same call repeated, consecutive failures, or session spend at the cap (see Runaway guard) | the guard is disabled when `runaway` is absent from config | closed, except an unreadable transcript (allows, logged) |
| `runaway-failure` | PostToolUseFailure | all tools | never | records which calls failed, for the failure streak | open |
| `data-gate` | PreToolUse | Bash | a recognized backtest command whose datasets are not PASS in the catalog, are unknown, or whose file hash changed; every failing dataset is listed | none | closed |
| `data-guard` | PreToolUse | Read, Bash, Grep | an agent reading a registered dataset with `privacy: private` (the default): `Read` of the file, `Grep` returning content from it, Bash readers (`cat`, `head`, `tail`, `less`, `sed`, `awk`, `jq`, `cut`, `sort`, `grep`/`rg`, and the like, also behind `sudo`, `sh -c`, pipes, `$(...)`, variables and `cd`), and inline Python, pandas, R, node or perl code that names the file. Approvable: "this sends licensed data to the cloud model". Datasets registered with `--privacy public` stay readable. **A script that prints rows is NOT caught** (see Data guard) | none | closed |
| `impact-check` | PreToolUse | Edit, Write, MultiEdit | never | log only: affected test files from `codegraph affected` for a file under `shared_module_globs` | open |
| `pre-restart-guard` | PreToolUse | Bash | a recognized engine start with any failed check: build hash, test receipt, datasets, readiness acknowledgement when `guard.require_readiness` is set (all failures listed) | the parity-disabled reason is logged | closed |
| `claim-check` | Stop | none | "tests pass", "build is clean", "lint is clean", "all checks green" with no matching successful run after the last edit; a stated test count ("N passed") that no passing test run after the last edit reports | none | closed, with the two Stop exceptions (below) |
| `receipt-required` | Stop | none | performance numbers in the last assistant message that match no receipt | none | closed, with two stated exceptions (below) |

"Closed" means that if the check itself cannot run (corrupt registry, unreadable or invalid
`config.yaml`, unparsable payload, an exception in the hook) the hook blocks with a clear reason.
"Open" hooks never break a session: on any failure they exit 0 and log the error (an invalid config
makes them skip, and session-context prints one line saying so).

`receipt-required` and `claim-check` have two deliberate exceptions to fail-closed, because Stop is not the place to trap
the user: a repeated Stop (`stop_hook_active` true) is checked exactly like the first one; the runner counts consecutive blocked Stops per session (stored beside the session record) and from the 3rd the block states that the claim cannot be verified and the user must decide. A block is never turned into an allow by the repeat. If the transcript cannot be read it
exits 0 and logs an error.

## Installing

```bash
uv run qrh hooks print-settings                        # the hooks section, to copy
uv run qrh hooks print-settings --settings PATH        # merged into an existing file, printed only
uv run qrh hooks print-settings --settings PATH --write  # writes; copies the old file to PATH.bak-<timestamp>
```

`hooks/claude_code/settings.example.json` is the same registration. Merging never removes existing
hooks or other keys, refreshes qrh's own entries in place and is idempotent. `--write` always needs
an explicit `--settings PATH`; qrh never defaults to your real settings file. `--scope user|project`
only labels which file the output is meant for (`~/.claude/settings.json` or `.claude/settings.json`).

## Configuration: `<QRH_HOME>/config.yaml`

A missing file means the defaults below. An invalid file (bad YAML, wrong type, unknown key, bad regex)
blocks the closed hooks and makes the open hooks skip.

```yaml
strategy_globs: ["strategies/*.py"]   # files lookahead-lint always lints (relative globs match at any depth, ** allowed)
detect_strategies: true               # also lint any .py that defines a class with a run() method and imports qrh.adapters
trigger_terms: [strategy, backtest, hypothesis, signal, alpha, regime, walk-forward, overfit]   # default
shared_module_globs: ["core/**/*.py"] # impact-check watches these
backtest_commands:                    # `qrh run <id>` is always recognized
  - pattern: 'python\s+run_backtest\.py'
    datasets: [prices]                # datasets that command uses
  - pattern: 'make backtest (?P<trial>\w+)'
    trial_id: from_args               # named group `trial`, else the first hex id after the match
engine_start_commands:
  - 'engine\s+start'
guard:                                # used by pre-restart-guard and `qrh test-receipt`
  source_globs: ["src/**/*.py"]       # relative to the session's working directory
  build_hash_file: build/SOURCE_HASH  # must contain the current source hash (`qrh hooks source-hash`)
  required_datasets: [prices]
  max_dataset_age: {prices: 3D}       # pandas offset string; compared with the date the data ends
  test_receipt_max_age: 1D
  parity: required                    # warns only if the report is missing or old; or {disabled: "reason of at least 10 characters"}
  require_readiness: <trial id>       # optional: block an engine start until `qrh readiness ack <trial id>` covers the
                                      # trial's current warning set (absent = check not applied)
runaway:                              # absent = the runaway guard is disabled; no numeric defaults exist
  repeat_limit: 3                     # the same call (tool + normalized input) this many times in a row
  failure_streak: 5                   # this many consecutive failed tool calls
  spend_cap_usd: 10                   # session spend from the transcript's latest cost-state
publish_guard:                        # absent = the publish guard is disabled; no path fallbacks exist
  protected_repos: [/home/me/work/my-repo]   # absolute paths ("~" is expanded); a repo is protected when its real path
                                             # equals or is under a listed path
claim_check:
  enabled: true
  test_commands: ['^\./ci/run-all']  # extra command regexes (searched in the whole command); built-ins are always on
  build_commands: ['make dist']
  lint_commands: ['checkstyle']
receipt_required:
  enabled: true
  metric_patterns: [calmar]           # extra metric names to treat as claims
  metric_map: {calmar: [calmar_ratio]}  # claim word -> metric key words (default: the word itself)
  tolerance: {abs: 0.01, rel: 0.005}  # a claim matches if within either
  recent_days: 30                     # receipts considered when the message names no trial
```

### Pre-restart guard

(a) the build hash file equals the current hash of `source_globs`; (b) a passing test receipt exists for
that hash and is younger than `test_receipt_max_age`; (c) each required dataset is PASS, unchanged and
its data ends within its max age; (d) parity is advisory: the guard never blocks a restart because of
the parity report. If the report in `<QRH_HOME>/parity/` is missing, unreadable, from another trial or
older than 7 days, it shows a plain warning. Parity can be turned off only with a stated reason
(`guard.parity: disabled`), and the reason is written to the log. Record a test receipt with
`qrh test-receipt -- pytest -q`: it runs the command, and stores exit code, source hash and duration;
only exit code 0 counts as a pass. Parity reports are written by the parity module
(`src/qrh/parity/report.py`) into `<QRH_HOME>/parity/`.
(e) With `guard.require_readiness: <trial id>`, that trial's deployment readiness report must be acknowledged by a
human for its current warning set (`qrh readiness status <id>`; see [readiness.md](readiness.md)). An unacknowledged,
stale or unknown trial is listed with the other failures; the block carries the banner and is approvable only through the
human `qrh approve` flow. With the key absent the check is not applied.

### Receipt-required

Claims are numbers next to a metric name (Sharpe, Sortino, CAGR, return, drawdown, PBO, DSR, win rate,
volatility, profit factor) in either order, such as "Sharpe of 1.4" or "a 12% max drawdown". "Return"
counts only with a percent sign. A claim is accepted only if an observed metric of the same family
matches within tolerance: "Sharpe" is compared with keys that contain the word `sharpe`, "DSR" with `dsr`,
"drawdown" with `drawdown` or `max_dd`, "volatility" with `vol`, and so on (one table, `FAMILY_KEYS` in
`receipt_required.py`, extended or overridden by `receipt_required.metric_map`). Only `receipt.metrics`
are read, and thresholds and limits (`min_*`, `max_*`, `limit*`), counts and sizes (`n_*`, `*_count`,
`grid_size`, `folds`, `observations`) and the null-model SR0 never count. Percent and fraction forms are
both tried for percent-type metrics and for numbers written with a percent sign, and a drawdown sign is
ignored ("20% drawdown" matches -0.20). The lookup uses receipts of the trials the message names (an id
prefix of at least 8 hex characters), else receipts created in the last `recent_days` days. Receipts that
fail their recorded hash back nothing.

## Approvals (the only way past a block)

There is no environment variable, flag or command-line switch an agent can use to pass a block. A block that needs a
human decision creates a pending approval in `<QRH_HOME>/approvals/` and its stderr starts with this banner, exactly:

```
##################
WARNING: DECISION TO BE MADE BY YOU!
##################
Content of the warning:
```

followed by the hook, the redacted and truncated action, why it was blocked, the risk, and the exact command the human
must run in their own terminal, and a last line telling the agent to relay the whole banner to the user verbatim and
STOP. The same banner is used by runaway-guard trips, config-integrity blocks, `qrh methods promote|demote|deprecate`
and the `qrh pack` gates while they wait for an approval. Final blocks that no human decision can lift (most
approval-guard rules) say "This block cannot be overridden." instead.

```bash
qrh approve --init                                       # once: creates the key pair, passphrase typed twice
qrh approve <id> --reason "<why this is acceptable>"     # reason of at least 10 characters
qrh approve --init --rotate                              # new key pair; needs the old passphrase; drops open approvals
```

`qrh approve` refuses to run without a controlling terminal (exit 2). It first shows the request (hook, the action
derived from the stored request, the block reason, the session id, the expiry), then reads the passphrase from the
terminal with echo off, decrypts the Ed25519 private key (`<QRH_HOME>/keys/approval.key.enc`, PKCS8, encrypted with a
password derived from the passphrase by scrypt, mode 0600) and signs a canonical payload: id, hook, fingerprint, hash of
the summary shown, session id, creation time, ttl, expiry, approval time, both reasons and the key id. A wrong
passphrase waits 1, 2, 4, ... seconds; five wrong passphrases within 10 minutes lock signing out for the rest of the
window.

When the agent retries the IDENTICAL action, the hook verifies the signature with the PUBLIC key only
(`<QRH_HOME>/keys/approval.pub`), then checks the key id, that the stored action still hashes to the signed fingerprint,
the session id (an approval is bound to the session that was blocked), the validity window, and consumes the token with
an exclusive `.used` marker (two racing hooks cannot both win). The override is recorded in the registry `overrides`
table, in the hook log (`override: true`, `approval_id`) and shown by session-context for 7 days. A second identical
action is blocked again with a new id. Expired, used, different-action, different-session, tampered, legacy (HMAC) and
other-key approvals are rejected, and the reason is put in the block message.

The pending request is checked by recomputation, because nothing secret exists when a hook writes it: the fingerprint
must equal the hash of the stored action, the expiry must equal creation time plus a ttl of at most 30 minutes, and the
summary the human reads is derived from the stored action rather than read from the file. Changing the request before
approval either fails these checks or changes what the human is shown; changing it after approval breaks the signature.
The request keeps the raw action (so the human signs exactly what was blocked) in a 0700 directory with 0600 files that
approval-guard keeps the agent out of.

The fingerprint is the SHA-256 of the tool name and the input; a Bash command is hashed raw with only leading and
trailing whitespace stripped (whitespace inside quotes is significant), file paths are made absolute, and the working
directory is part of a Bash fingerprint. Stop hooks have no tool call, so `receipt-required` and `claim-check` submit a
fixed list of strings (the unsupported claims), which is also what the human sees. CLI gates (`qrh pack`, `qrh methods`)
use the same flow with a fixed session id, so their approvals are not tied to an agent session.

### approval-guard

`approval-guard` is a PreToolUse hook (matcher `Bash|Read|Edit|Write|MultiEdit|NotebookEdit|Grep|Glob|WebFetch|mcp__.*`). Its path
rules are final, with no override:

- commands that run `qrh approve` (also `--init`), `qrh config sign`, `qrh readiness ack`, `qrh interview answer --human`, `qrh data confirm`, `qrh data propose --as-user` or `qrh hook ...`
  by hand, import the approval, signing or interview-writing code, or set `QRH_OVERRIDE`;
- any path at or below `<QRH_HOME>/keys` or `<QRH_HOME>/approvals`, `<QRH_HOME>` itself, or any ancestor of it (your home
  directory and `/` included), for reads and writes alike. Bash tokens are checked after `$QRH_HOME`, `$HOME`, `$USER`,
  `~`, brace and glob expansion, `cd` tracking and symlink resolution; a recursive tool (`rg`, `grep -r`, `find`, `tar`,
  `cp -r`, `rsync`, `zip`, `du`, `ls -R`, ...) whose working directory is such an ancestor is blocked too. Read, Grep and
  Glob paths are checked, with Grep `glob` and Glob `pattern` resolved against the tool's `path` (else the cwd);
- writes to `<QRH_HOME>/{config.yaml,registry.sqlite3*,parity,runaway,packs.json,sessions,approvals,keys,readiness}`, to Claude
  Code settings (`~/.claude/settings*.json`, `<cwd>/.claude/settings*.json`, `<cwd>/.claude/hooks`) and to transcripts
  (`~/.claude/projects`): redirections, `tee`, `sed -i`, `mv`, `cp` and `rsync` destinations, `rm`, `ln`, `chmod`,
  `truncate`, `dd`, and any interpreter command that names such a path; sqlite3 or interpreter access to the registry.
  Reading `config.yaml` and the logs stays allowed;
- MCP tools whose input names any of those paths (read or write cannot be told apart, so both are blocked);
- state names in any spelling: an interpreter or shell command (`python`, `node`, `perl`, `bash -c`, `uv run`, ...) that
  names `registry.sqlite3`, or names `config.yaml`, `parity`, `sessions`, `keys`, `approvals` or `readiness` next to a `QRH_HOME`,
  `.qrh` or home-path reference, after quotes, `+` joins and spaces are squeezed out (so string concatenation, an
  f-string and `os.environ['QRH_HOME']` do not hide it). Ordinary commands that merely use such a word (`pytest -k
  sessions`, `{}.keys()`, `--config config.yaml`) are not affected;
- `file:` URLs in WebFetch, any other tool taking a URL, MCP input and shell commands (`curl file://...`): blocked unless
  they point (after decoding and symlink resolution) into the current working tree and not at a protected path. A host
  other than `localhost` blocks. WebFetch is therefore routed through this hook.

Two rules are best-effort and approvable (with the banner and an approval id):

- Write, Edit, MultiEdit, NotebookEdit and MCP content that imports the approval, signing or interview-writing code,
  names the key files, opens the registry, or builds those names dynamically (`importlib`, `__import__`, string
  concatenation);
- code-path and persistence actions: writes (file tools, redirection, `tee`, `cp`, `mv`, `install`, `ln`, `dd of=`,
  `sed -i`, `rm`, `touch`, `chmod`) to `sitecustomize.py` or `usercustomize.py` (anywhere), a `*.pth` file or anything else under `site-packages`,
  `dist-packages`, the user site directory or a venv `lib/pythonX` directory (a `.pth` elsewhere, such as a PyTorch
  checkpoint, is inert and not gated), or a systemd user unit; `pip install -e`, `uv pip install -e`, `pip install`/`uv add`/`uv tool
  install` of a path, archive, URL or alternate index, `uv tool install --from`, `uvx --from`, and `PYTHONPATH=` on a
  command that runs `qrh`; `crontab` (except `crontab -l`), `at`/`batch`, `systemctl --user enable|link|edit|start|
  daemon-reload|...` and `systemd-run --user`. Plain `uv add <name>`, `uv sync` and `pip install <name>` stay allowed,
  and so does editing `src/` and `pyproject.toml` (normal development; session-context warns instead, see Deployment
  hardening in [security.md](security.md)).

A determined program can hide what it imports; what actually prevents self-approval is the signature scheme. Anything the
guard cannot check blocks.

### Threat model

Designed for: an agent running as the same OS user as you, with Bash, file tools and an interpreter, but without your
passphrase and without a terminal you type into. Under that model:

- the agent cannot produce a valid approval or config baseline: both need the private key, which only your passphrase
  unlocks, and every verification uses the public key only;
- an approval covers one action, one use, one session and at most 30 minutes, and what you sign is what was blocked;
- changing `config.yaml` or the `hooks` section of a signed settings file makes every enforcement hook block until you
  sign again (see "Config integrity").

Not covered: a malicious human or root; a compromised terminal or keylogger; a weak passphrase (its strength is yours:
the encrypted key file is protected by scrypt, but an agent that copies it can try passphrases offline); an agent that
replaces the whole key pair with one it made itself (it needs a terminal; it can allocate a pseudo-terminal, so this is
detected, not prevented: session-context reports any key created or changed in the last 7 days, using the file's change
time, which an unprivileged process cannot set back); an agent that edits the installed qrh code or deletes the config
baseline together with the keys (see below). Closing those needs a separate OS user, root-owned files or a sandbox.
See [security.md](security.md).

### Known gaps

What remains:

- **Same-user code execution.** approval-guard is a text and path filter. A program that builds paths or module names at
  run time can still read or delete files the agent's user can access. That cannot forge an approval or a baseline,
  but it can delete them (an approval then simply fails).
- **Baseline.** Once a key exists, a deleted baseline blocks enforcement hooks (`qrh config sign`). Deleting the keys and
  the baseline together returns qrh to its first-use state, where a missing baseline only warns.
- **Key replacement is detected, not prevented** (see the threat model). The agent can delete the keys and run
  `qrh approve --init` under a pseudo-terminal to install a key of its own. Session-context prints the key fingerprint at
  every session start and `qrh approve --init` prints it once: note it where the agent cannot write, and compare.
- **Hooks run code the agent can edit** when qrh is an editable install or is installed in a directory the agent's user
  can write. Session-context says so; see Deployment hardening in [security.md](security.md).
- **Dynamic path construction** (for example a path decoded from base64 inside the command) cannot be caught by a text
  filter. It only exposes the encrypted key, whose protection is the passphrase.
- **CLI gates are not session-bound** (`qrh pack`, `qrh methods`), and a repeated Stop is still checked (see the
  Stop hooks above); only the block message changes after 3 consecutive blocks.

## Config integrity

```bash
qrh config sign --settings ~/.claude/settings.json [--settings .claude/settings.json ...]
```

In your own terminal, with the approval passphrase: shows the SHA-256 of `<QRH_HOME>/config.yaml` and of the `hooks`
section of each settings file, then signs that baseline (`<QRH_HOME>/keys/config-baseline.json`). Enforcement hooks that
read config verify the signature with the public key and compare the current hashes; on any difference they block with
the banner and "config or hook registration changed since it was signed; run qrh config sign", and session-context says
the same. Non-hook settings (theme, permissions) are not covered. Before any approval key exists, nothing blocks and
session-context asks you to create a key and a baseline. Once a key exists, a MISSING baseline blocks too (banner,
`qrh config sign`): deleting the baseline does not switch the check off. The hooks therefore block only while a baseline
exists and is intact, or a key makes its absence an alarm; if the whole key directory is deleted too, the check falls
back to the first-use warning (see Known gaps). A key rotation invalidates the baseline: sign again after rotating.

## Sessions

SessionStart and UserPromptSubmit payloads record the session's `transcript_path` in `<QRH_HOME>/sessions/` (first path
wins). `qrh interview answer --transcript T --session S` accepts T only if S was recorded, T equals the recorded path
and T lives under `~/.claude/projects`.

## Runaway guard

Three trip wires per session, all configured under `runaway` in `config.yaml` (a missing section disables the
guard, and session-context says so; there are no numeric fallbacks):

- `repeat_limit`: the same tool call (tool + normalized input) this many times in a row. Waiting and delegating
  tools (Task, Agent, TaskOutput, sleep, AskUserQuestion, ...) never count and end a run.
- `failure_streak`: this many consecutive tool calls that failed, from the PostToolUseFailure events
  (`runaway-failure`). Interrupts, denials and hook blocks are not failures.
- `spend_cap_usd`: session spend taken from the latest `cost-state` record in the transcript (exact, the harness
  figure). No cost-state yet means no spend trip.

`<cwd>/.qrh/config.yaml` may carry its own `runaway` section: it can lower a limit set at user level but never raise
one, and cannot switch on a wire the user level left off. When a wire trips the call is blocked (exit 2) with the
reason and the approval flow above; an approval lets one call through and restarts the counts. State is kept
per session in `<QRH_HOME>/runaway/` (hashes and counters only). Documented fail-open exception: if the transcript
cannot be read the spend check is skipped, the call is allowed and the error is logged.

## Claim check

`claim-check` (Stop) reads the last assistant message and looks for claims that tests pass, the build is clean,
lint is clean or all checks are green (case-insensitive regex families, one claim per kind). Negated, hedged,
conditional, planned or past-tense sentences are not claims ("not all tests pass", "make sure the tests pass",
"the build was clean before"), and neither is text in a code block. A claim is accepted only if the transcript shows
a matching command that ran AFTER the last Edit/Write/MultiEdit/NotebookEdit and succeeded: no error flag, no failure
summary ("2 failed", `FAILED`, `Exit code 1`). Built-in test commands: pytest, `uv run pytest`, `python -m pytest`,
tox, npm/pnpm/yarn/bun test, jest, vitest, go test, cargo test, make test, `qrh test-receipt`; build and lint
families (cargo build, tsc, ruff, mypy, eslint, ...) are built in too, and `claim_check.*_commands` adds more.
"All checks green" needs a passing test run and no failed build or lint run. Otherwise exit 2 lists the unsupported
claims.

A test count stated with the claim ("1,234 passed", "all 12 unit tests passed"; thousands separators `,` and `_` are
accepted) must also equal the "<n> passed" figure printed by a successful test run after the last edit. A count that
matches no such run is listed as unsupported, with the counts the runs did show.

A repeated Stop (`stop_hook_active`) is checked like the first one, with no skipped checks. After 3 consecutive blocked
Stops in a session the block message adds that the claim cannot be verified and the user must decide; the counter is kept
per session in the `sessions` folder of `<QRH_HOME>`, is reset by a Stop that is not blocked, and never turns a block into
an allow. An unreadable transcript allows and logs an error, like `receipt-required`.

`qrh audit tests [--base REF]` is the matching deterministic diff check for weakened tests (working tree against
`HEAD` or a ref, untracked files included): deleted test files, deleted test functions, more removed than added
assertions, and added skip / xfail / `if False` markers. A test moved to another file is not a loss. Exit 2 on any
finding. The other `qrh audit` subcommands remain BLOCKED stubs.

## Logging and privacy

Every invocation appends one JSON line to `<QRH_HOME>/logs/hooks.jsonl`: `ts`, `session_id`, `hook`,
`event`, `decision` (allow, block, warn, info, error), `reason`, `trial_id`, `duration_ms`, and for Bash
events a `command`. Prompts and file contents are never logged. Commands are redacted first (provider keys
such as `sk-`, `ghp_`, `gho_`, `ghs_`, `github_pat_`, `xox?-`, `AKIA`, `ASIA`; Bearer and Basic tokens;
`token`, `password`, `secret`, `api_key` and `authorization` pairs; `--password`, `--token`, `--secret`
flags) and then truncated to 160 characters. Reasons pass through the same redaction.

`data-gate` also appends to `<QRH_HOME>/logs/session_trials.jsonl` (`session_id`, `trial_id`, `stage`,
`kind`, `ts`) whenever a Bash command contains `qrh new|run|report <id>` or a mapped backtest command, even
if the command is then blocked. A later phase uses it to attribute AI usage to trials. (`qrh new` is
recorded with no trial id, because the id does not exist yet.)

## Publication guard

Publication is guarded outside Claude Code, by `.pre-commit-config.yaml`: gitleaks for secrets and
`scripts/denylist_check.py` for a private denylist (`QRH_DENYLIST`). A unit test checks that the denylist
hook stays wired in.

## Publish guard

`publish-guard` (PreToolUse; Bash, Write, Edit, MultiEdit) protects that outside guard from the agent session. It reads
`publish_guard.protected_repos` from `config.yaml` (absolute paths; no fallbacks: without the section the hook is
disabled and session-context prints one line saying so). A repository is protected when its real path equals or is under
a listed path (symlinks resolved, whole path components compared). In a protected repository these need a human
decision (banner and approval id):

- publishing: `git push` (also `--no-verify`), `git remote add|set-url`, `gh repo create|edit|fork|delete|transfer|rename`
  (so also `--visibility`), `gh release create|upload`, `gh pr create|merge`, `gh api` with `-X POST|PUT|PATCH|DELETE` (or
  `-f`/`-F` fields) against `repos/...` and GraphQL mutations, `npm|pnpm|yarn publish`, `twine upload`,
  `uv|cargo|poetry|hatch|flit publish`;
- weakening the pre-publication checks: `git commit --no-verify|-n`, `git config core.hooksPath` (also `git -c`; at
  `--global`/`--system` scope it is gated wherever it runs), `pre-commit uninstall`;
- any Write, Edit, MultiEdit or shell write, move or delete (redirection, `tee`, `cp`, `mv`, `rm`, `sed -i`, `git rm|mv`)
  of `.git/hooks/*`, `.git/config`, `.pre-commit-config.yaml`, `.gitleaks.toml`, `scripts/denylist_check.py` and
  `.github/workflows/*`. Shell writes include `chmod`, `chattr`, `touch`, `truncate`, `ln`, `install`, `cp`, `mv`, `rm`,
  `dd of=` and `git update-index --chmod`: git silently skips a non-executable hook, so `chmod -x .git/hooks/pre-commit`
  disables the leak scan and is gated like a deletion.

The command is read as written: quotes keep a commit message that mentions `git push` from counting, and `cd`, `git -C`,
`sh -c`, `eval` and heredocs piped to a shell are followed. Variables, aliases and what an interpreter's own code does
(`python -c "...open('.gitleaks.toml','w')..."`) are not seen, so this stops ordinary and careless actions, not a
determined program; the same-user limits of the threat model apply. Accepted limitations of a text filter, each a strict
xfail in `tests/unit/test_publish_matrix.py`:

- redirection of a verb through a shell variable (`G=git; $G push`) or a command substitution (`git $(echo push)`);
- an alias (`alias gp='git push'; gp`);
- a push or a guard-file write made by an interpreter's own code (`python -c "subprocess.run(['git','push'])"`,
  `open('.gitleaks.toml','w')`);
- git settings supplied through the environment (`GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_0=core.hooksPath`);
- a write to the global git config file (`~/.gitconfig`) with a file tool.

The stronger control is the git-level pre-push check described in "Git pre-push approval" below, because git runs it
even when the push comes from a script or an alias. Normal work (`git add|commit|status|diff|fetch|pull`,
`pre-commit run|install`, tests, builds, reading the guard files) is untouched.

## Git pre-push approval

`qrh publish pre-push` is git's own pre-push hook protocol. Git runs it for every push, whoever starts it (an agent, a
script, an alias), and it refuses unless the user has signed an approval for exactly this push. It complements
publish-guard, which only reads the text of agent commands, and does not replace it.

What an approval covers (all of it is signed, through the same Ed25519 scheme and single-use marker as every other
approval): the remote URL, and every update read from stdin as `(local_ref, local_sha, remote_ref, remote_sha)`,
deletions included (their local sha is all zeros). The binding is therefore the remote URL plus the object ids. The
real path of the repository is signed as well, but only as an extra check, not a boundary (`GIT_DIR` and `GIT_WORK_TREE`
come from the environment). The URL is bound without its password, query string and fragment (also behind a
`<helper>::` prefix); the user name stays, so `alice@host` and `bob@host` are different remotes. The order of lines does
not matter; any other ref, commit, remote sha (a force push over a remote that moved), URL or repository needs a new
approval. The remote name is not covered. The approval is valid once, within 30 minutes; the allowed push consumes it. It
is not tied to an agent session. Push options (`git push -o ...`) are not passed to pre-push hooks, so they are neither
seen nor bound.

What the user reads: `qrh approve` shows the update count and one line per update (`<local_ref> <local_sha> ->
<remote_ref> (was <remote_sha>)`), never cut and with ref names unmasked. The text shown is derived from the signed
action. If a push could not be shown in full, no request is created.

The hook does not trust the environment of whoever runs git. It requires `--home <absolute path>` and `--key-id <id>`,
ignores `QRH_HOME` and `HOME`, and refuses when the key in that home has another id, so an agent cannot point it at a home
with a key of its own.

Refused (exit 2, one line saying why): no approval key, a key id that does not match, a relative or missing home,
unreadable key or state, empty or malformed stdin, an empty URL, more than 500 updates or 256 KiB of input, a changed
sha, a different remote, a reused, expired or tampered token. When a push is only waiting for approval, the command
creates (or reuses) a pending request and prints its id and the command for the user:

```bash
qrh approve <id> --reason "<why this push is acceptable>"   # in your own terminal, never through the agent
```

then retry the push. Nothing secret is printed.

Install it yourself; nothing here installs anything. After `qrh approve --init`, in your own terminal, in the repository:

```bash
qrh publish print-hook > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

`qrh publish print-hook` is read-only: it prints the file for the current `QRH_HOME` and key and writes nothing. The
printed hook looks like this (paths and key id are yours):

```sh
#!/bin/sh
exec env -i PATH=/usr/bin:/bin /abs/venv/bin/python -I -m qrh publish pre-push --home /abs/.qrh --key-id <key id> "$@"
```

`env -i` drops the caller's environment (`QRH_HOME`, `HOME`, `PATH`, `PYTHONPATH`), `-I` makes Python ignore `PYTHON*`
variables, the user site and the current directory, and the absolute interpreter and home are fixed in the file. Print
the hook again after a key rotation (the key id changes). If git is not in `/usr/bin` or `/bin`, `print-hook` adds its
directory to the pinned `PATH`.

There is no `pre-commit` entry on purpose. The `pre-push` stage of pre-commit does not forward git's stdin; it gives only
`PRE_COMMIT_FROM_REF`/`PRE_COMMIT_TO_REF` and one local and remote branch, which cannot describe a push of several refs,
tags or deletions, so an approval could not be bound as strictly. A hook file is the only supported installation.

Limits: `git push --no-verify` skips client-side hooks (this is why publish-guard blocks `--no-verify` for agents), and a
process running as the same user can delete or edit the hook file, change `core.hooksPath`, delete an approval's `.used`
marker or push without git hooks by other means. The check reduces accidents and raises the cost of a careless or
scripted push; it is not a sandbox.

## Data guard

`data-guard` (PreToolUse; `Read`, `Bash`, `Grep`) protects licensed data. A dataset registered with `qrh data register` is
`privacy: private` unless it was registered with `--privacy public` (synthetic data only). When an agent reads a private
file, the rows go to the cloud model as tool output, so the hook blocks it with an approvable banner: "this sends licensed
data to the cloud model". A human decides in their own terminal (`qrh approve`, valid once for that exact action);
public datasets and unregistered files are never blocked.

Best-effort detection, from the command text only:

- `Read` of the file (relative paths, `..` and symlinks are resolved). `qrh data register` stores the absolute path, so the
  hook finds the file from any working directory. A dataset registered by an older version with a relative path cannot be
  located, so the hook blocks any read of a file with the same name and asks for a re-registration;
- `Grep` on the file in any output mode, or on a directory above it in `content` mode (a `glob` or `type` that cannot
  match the file is honoured);
- Bash readers that print what they read (`cat`, `head`, `tail`, `less`, `sed`, `awk`, `jq`, `cut`, `sort`, `nl`, `od`, `xxd`,
  `grep`, `rg`, ...), including behind `sudo`/`env`/`timeout`, `sh -c`, pipes, `$(...)`, `cd`, `~`, `$HOME`, brace and glob
  expansion, variables assigned in the same command and `< file` (the redirect is judged against the command it belongs to:
  `[ -f F ] && cat < F` blocks, `wc -l < F` does not); after a `cd` to a target the hook cannot resolve (`cd "$(...)"`,
  `cd -`, a bare `cd`) relative words are matched against any directory; recursive `grep`/`rg`, `find` and `xargs` that reach the file
  (`-l`, `-c` and `--files` listings of a directory are allowed). A word built from a variable the hook cannot resolve blocks
  when the rest of the word could name a private dataset;
- inline code that names the file or its basename: `python -c`, a heredoc that feeds an interpreter or a pipe into one
  (pandas, polars, numpy, `open`, `Path`), also `-e` for R, node, perl and ruby, `-r` for php and `-c` for duckdb. Whether a
  command is a reader is decided from its program word (after `sudo`, `env`, `timeout`, `uv run`, `xargs`, `find -exec`),
  never from an argument: `python run.py --data F --order sort` is a script run, not a `sort`;
- a check that would run longer than a few seconds is blocked instead of left to the hook timeout.

**A script that prints rows is NOT caught.** `python run.py --data FILE` is the normal way to compute on a private dataset,
and the hook sees the command line, not what the script then does with the path. Also not caught: a copy of the file made
first and then read, a path assembled at run time in a way a text filter cannot follow, and code in an unregistered
location that reads a registered path. The hook stops the plain act of reading a private file into the conversation; it is
not a sandbox. Commands that do not show content (`wc`, `ls`, `stat`, checksums, `cp`, `qrh data ...`) stay allowed. Like
every enforcement hook it fails closed: an unreadable registry blocks.

## Limits

- Text heuristics. The lint is AST-based but partial; recognizing backtest, engine and claim text is
  by regex. A command inside a quoted string can look like a real one, and an unusual phrasing can hide a claim.
- Claims match by metric family and value. A different key of the same family (for example an in-sample
  Sharpe quoted as the out-of-sample one) still passes, and a stage metric whose key lacks the family word
  is invisible until you add it to `metric_map`.
- One action at a time: an approval covers one blocked action, and hooks see one tool call, not the plan.
  `data-gate` and `pre-restart-guard` read the command text; they cannot see what a script does inside. The same holds for `approval-guard`: see the threat model.
- Enforcement hooks that read config also verify the signed config baseline when one exists (see Config integrity).
- PostToolUse and Stop cannot undo anything; they only feed the reason back.
- `impact-check` depends on the `codegraph` CLI being installed and indexed; otherwise it logs that the
  check was unavailable. `codegraph affected` lists affected test files, so the count is of those.
- Hooks do no network access and add well under a second on small registries.
