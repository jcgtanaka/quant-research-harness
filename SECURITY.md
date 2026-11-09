# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately, not in a public issue or pull request.

**Contact: jcgtanaka@outlook.com** (use the subject line "qrh security report"). You can expect a first reply within
7 days.

Include the version, what you did, what you expected and what happened, and a minimal reproduction using synthetic data.
Do not include real keys, passphrases or account data.

## In scope

- Bypasses of the human approval mechanism (self-approval, forged or reused approvals, signature or consistency-check
  flaws).
- Bypasses of the enforcement hooks or the config baseline within the threat model below.
- Leaks of private data through the dataset profile, reports, logs or the LLM replay cache.
- Unsafe handling of untrusted input such as dataset proposals.
- Secrets or private material committed to the repository.

## Threat model

Full details: [docs/security.md](docs/security.md).

Designed for: an agent that runs as the same OS user as you, with a shell, file tools and an interpreter, but that does
not know your passphrase and has no terminal you type into.

Not designed for: a malicious human, root, a compromised terminal or keyboard, or malware outside the agent.

Documented limits (these are known and accepted, so reports about them alone are out of scope):

- A same-user program can delete or replace files. It cannot forge a signature with your key, but it can delete the keys
  and the config baseline together, or replace the whole key pair with its own.
- It can edit the qrh code the hooks run when that code is writable by the agent's user (for example an editable install).
- `approval-guard` is a text and path filter (defense in depth), not a sandbox; obfuscated programs can get past it.
- `qrh pack` and `qrh methods` approvals are not bound to an agent session.
- The publish guard reads command text only; its accepted limitations are listed in docs/security.md.
- Closing these needs a separate OS user, root-owned files or a sandbox.

Deployment hardening recommendations are in docs/security.md.

## Keys and passphrases

Approval keys and passphrases must never be shared, pasted into an agent session, or committed. Do not attach them to a
report. If you think a key or passphrase is exposed, rotate it as described in docs/security.md and treat approvals made
since then as suspect.
