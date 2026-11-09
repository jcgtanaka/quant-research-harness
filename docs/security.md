# Security

qrh lets a human, and only a human, lift a block an AI agent hit. This page says what that rests on.

## Threat model

Designed for: an agent that runs as the **same OS user** as you, with a shell, file tools and an interpreter, but that
does **not** know your passphrase and has **no terminal you type into**.

Not designed for: a malicious human, root, a compromised terminal or keyboard, or malware outside the agent.

## What it guarantees

- **No self-approval for the current key.** Approvals are Ed25519 signatures. The private key is stored encrypted
  (`<QRH_HOME>/keys/approval.key.enc`, PKCS8, password derived from your passphrase with scrypt, mode 0600). Hooks and
  every check use only the public key (`approval.pub`). Without the passphrase, no valid approval can be made *for that
  key*. This holds only while the key pair and the qrh code are the ones you set up: an agent that replaces the key pair
  with its own (key replacement) or edits the code the hooks run (code editing) is outside it. See "What it does not
  guarantee" and "Deployment hardening".
- **You sign what was blocked.** `qrh approve` shows the hook, the action (derived from the stored request, not from
  text the agent could edit), the reason, the session and the expiry before asking for the passphrase. A request
  changed before approval fails its consistency checks; one changed after approval fails the signature.
- **Narrow approvals.** One action (exact fingerprint), one use, one session, at most 30 minutes.
- **Config integrity.** A signed baseline of `config.yaml` and of the `hooks` section of your Claude Code settings
  makes enforcement hooks block when either changes, until you sign again. Hooks block only while a baseline exists and
  is intact; a baseline that is deleted after a key exists also blocks (until you sign again), but one deleted together
  with the keys returns qrh to its first-use state, where a missing baseline only warns.
- **Readiness acknowledgement.** `qrh readiness ack <trial>` is a human step on the same terms as `qrh approve` (a
  controlling terminal, the passphrase, no other input path). It shows the verdict, the counts and every HIGH warning, then
  signs the report hash and the sorted warning ids under its own domain (`readiness-ack`, so an approval signature is not
  an acknowledgement and the reverse). The check uses the public key only and fails closed; any change in the warnings,
  the receipts, the verdict or the key makes the acknowledgement invalid. `approval-guard` blocks an agent from running it
  and from writing `<QRH_HOME>/readiness/`. See [readiness.md](readiness.md).
- **LLM replay cache.** `<QRH_HOME>/llm_cache/<trial_id>/` holds the prompts and answers recorded for a trial (see
  [llm.md](llm.md)). Prompts may include licensed data, so the cache lives only in `QRH_HOME`, never in the repository, and
  must never be committed or published. `approval-guard` blocks an agent from **writing** under `<QRH_HOME>/llm_cache` (Edit,
  Write, redirects, `cp`, `mv`, `rm`, interpreters that name it next to QRH_HOME) so that it cannot fabricate answers that a
  replay would serve as a model's; reading stays allowed. A declared adapter can still call a provider directly, outside
  the cache; the LLM002 lint warns about it on a best-effort basis (it does not prevent it, see [llm.md](llm.md)). Entry hashes detect corruption, not a deliberate rewrite that
  recomputes them.
- **Brute force through qrh is slow.** A wrong passphrase waits 1, 2, 4, ... seconds; 5 failures in 10 minutes lock out.
- **Banners.** Every block that needs your decision starts with `WARNING: DECISION TO BE MADE BY YOU!` and the exact
  command to run in your own terminal.

## Dataset proposals and the profile

An onboarding proposal (`qrh data propose --file`) is written by the agent, so it is **untrusted input**: JSON only, at most
64 KB, a strict schema with unknown keys refused (a claim cannot carry its own status or source), duplicate keys and `NaN`
refused, every string limited in length and free of control and invisible characters and of `$`, and never interpreted as
markup (the PDF pages draw claims and evidence as plain text). A proposal only ever creates `unverified` claims: code
verifies, and what code cannot prove is confirmed by the human through the interview. `qrh data propose --as-user` and
`qrh data confirm` record the user's own word, so approval-guard refuses them to the agent, like `qrh interview answer
--human`. The profile (`qrh data profile`) is the only view of a dataset meant for the agent: aggregates, shapes of text
columns and, for a column whose whole value set is a known flag vocabulary ({C,P}, {Y,N}, {B,S}, {BUY,SELL}, ...) and
that is not named like an identifier, its values; never a row. Column names are shown. Residual risk: a column whose
values are exactly {C,P}, {B,S}, {T,F} or {L,S} may be two tickers and would be shown (`--no-values` hides it). The data-guard hook blocks direct reads of a private dataset (see hooks.md, Data guard); a script that prints rows is not caught.

## What it does not guarantee

- Passphrase strength is yours. An agent that obtains a copy of the encrypted key can try passphrases offline; scrypt
  slows that, a long passphrase stops it. Use at least 12 characters (enforced), better a long phrase.
- A same-user program can delete or replace files. It cannot forge a signature with your key, but it can delete the
  keys and the config baseline together (first-use state, warning only) or replace the whole key pair with its own: a
  pseudo-terminal satisfies the terminal check, so `qrh approve --init` then installs an agent-owned key. Session-context
  prints the key fingerprint at every session start (compare it with the one `qrh approve --init` printed and that you
  noted elsewhere) and also mentions a key created or changed in the last 7 days, using the file change time. That
  notice is informational: it is a file time, not proof, and it vanishes after 7 days; only your out-of-band record of
  the fingerprint is lasting.
- It can also edit the qrh code the hooks run when that code is writable by the agent's user (an editable install from
  a source checkout, or an install in a directory you own). Session-context warns when it is. See "Deployment
  hardening".
- approval-guard is a text and path filter (defense in depth), not a sandbox; obfuscated programs can get past it. A
  path built at run time (for example decoded from base64 inside the command) cannot be caught by any text filter; it
  only exposes the *encrypted* key, whose protection is the passphrase.
- `qrh pack` and `qrh methods` approvals are not bound to an agent session.
- Git pre-push approval (`qrh publish pre-push`, see [hooks.md](hooks.md)) is a client-side hook. It guarantees that a
  push that runs the hook has a signed, single-use approval of the user bound to the remote URL (without password, query
  or fragment) and the exact object ids: each `(local_ref, local_sha, remote_ref, remote_sha)` update, deletions
  included. The repository path is signed too, but it is an extra check, not a boundary: `GIT_DIR` and `GIT_WORK_TREE`
  come from the environment. The hook ignores `QRH_HOME` and `HOME`; the hook file pins an absolute home, the key id,
  an empty environment and an isolated Python. It does not stop `git push --no-verify`, deleting or editing
  `.git/hooks/pre-push`, a changed `core.hooksPath`, or a program that talks to the remote without git. Push options
  (`git push -o ...`) are not passed to pre-push hooks, so they are neither seen nor bound, and a client-side hook cannot
  close that. The `.used` marker that makes an approval single-use is a same-user file: deleting it (like deleting the
  keys) is a same-user limit. Publish-guard blocks these for agent commands; the same-user limits above apply to the
  rest. Its approvals are not bound to an agent session.
- Closing these needs a separate OS user, root-owned files or a sandbox.

## Deployment hardening (recommended)

The signature scheme protects approvals; the hooks themselves are code. Make that code and the key material hard for the
agent's user to change:

- Install qrh **non-editable** into a separate virtual environment, or as a `uv tool`, in a location whose files are
  read-only or owned by another user (for example a root-owned directory, or an account that is not the one the agent
  runs as). An editable install executes whatever is in your source checkout, which the agent can edit; session-context
  prints a warning while qrh is imported from a directory the current user can write.
- Keep `QRH_HOME` (keys, config, baseline) and your Claude Code settings outside every directory the agent works in.
- Record the approval key fingerprint **out of band**: `qrh approve --init` prints it, and session-context prints the one
  in use at every session start. Note it where the agent cannot write (paper, a password manager, another machine) and
  compare. A changed value means the key was replaced; treat approvals since then as suspect.
- Optionally run the agent under a **separate OS user**, so that neither the keys nor the installed code are writable
  from its account. This is the only option here that turns the same-user limits into real boundaries.

State plainly what stays open: a path the agent builds at run time, such as one decoded from base64 inside the command,
cannot be caught by a text filter, and the base64-built path payload in the regression matrix
(`tests/unit/test_guard_matrix.py`) is an accepted, documented limitation (a strict xfail). What it reaches is the
*encrypted* key file, whose protection is the passphrase. Accepted limitations of the publish guard, which reads command text only (each is a strict xfail in
`tests/unit/test_publish_matrix.py`):

- redirection of a verb through a shell variable (`G=git; $G push`) or a command substitution (`git $(echo push)`);
- an alias (`alias gp='git push'; gp`);
- a push or a guard-file write made by an interpreter's own code (`python -c "subprocess.run(['git','push'])"`,
  `open('.gitleaks.toml','w')`);
- git settings supplied through the environment (`GIT_CONFIG_COUNT`/`GIT_CONFIG_KEY_0=core.hooksPath`);
- a write to the global git config file (`~/.gitconfig`) with a file tool.

The stronger control is the git-level pre-push check that verifies a signed approval (see "Git pre-push approval" in
[hooks.md](hooks.md)), because git runs it even when the push comes from a script or an alias.

A `.pth` file is gated only inside site directories (site-packages, dist-packages, the user site, a venv `lib/pythonX`),
the only places Python processes it; a PyTorch checkpoint elsewhere is not. `sitecustomize.py` and `usercustomize.py`
are gated everywhere. Editing `src/` and `pyproject.toml` stays allowed so that normal
development works; hardening the install location is what closes it.

## First-time setup

In your own terminal, never through the agent:

```bash
qrh approve --init                                         # passphrase typed twice, at least 12 characters; note the printed fingerprint
qrh config sign --settings ~/.claude/settings.json         # add --settings for each settings file with qrh hooks
```

Run `qrh config sign` again whenever you change `config.yaml` or the hook registrations on purpose, and after a rotation.

## Rotation and a lost passphrase

- Rotate (you know the passphrase): `qrh approve --init --rotate`. It asks for the current passphrase, then the new one
  twice, and drops every pending and approved request. Sign the config baseline again afterwards.
- Lost passphrase: there is deliberately no command or flag that skips the old passphrase (an agent with a
  pseudo-terminal could use it). Reset by hand in your own terminal, which destroys all pending approvals and the
  baseline: `rm -r ~/.qrh/keys ~/.qrh/approvals`, then `qrh approve --init` and `qrh config sign ...`. If you did not
  do this yourself and session-context reports a new key, treat every approval since then as suspect.

## How tests inject keys

Tests never use environment variables or flags for secrets. They generate a key pair at runtime in a temporary
`QRH_HOME` and sign through the library seam (`qrh.signing.unlock`, `qrh.approvals.sign_approval`), or replace the
terminal functions `qrh.tty.has_terminal`, `qrh.tty.read_secret` and `qrh.tty.write` with pytest `monkeypatch`. The
scrypt cost is lowered through the module attribute `qrh.signing.KDF_N` in `tests/conftest.py`. No key material is
committed.
