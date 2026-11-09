"""Human approval tokens for blocked actions, signed with the human's Ed25519 key.

When an enforcement hook (or a gated CLI command) blocks, it writes a *pending* request under
``<QRH_HOME>/approvals/``: the hook, the action itself (tool call or fixed parts), its fingerprint, the reason,
the session id and the expiry. A human runs ``qrh approve <id> --reason "..."`` in their own terminal: it shows
the request, asks for the passphrase on the terminal, unlocks the private key (``qrh.signing``) and signs a
canonical payload. The hook, on the identical retried action in the same session, verifies the signature with the
PUBLIC key only, consumes the token once and allows that one action.

Layout under ``<QRH_HOME>``::

    approvals/<id>.json   the request (pending, then approved with a ``signature``)
    approvals/<id>.used   exclusive marker created when the token is consumed (single use)

Integrity of the pending request: there is no secret when it is written, so it is checked by recomputation. The
fingerprint must equal the hash of the stored action, the expiry must equal creation time plus a ttl no longer than
``MAX_TTL_S``, and the summary the human reads is derived from the stored action, never taken from the file. An
agent that edits the request therefore either breaks those checks or changes what the human is shown.

Design adapted from agent-guardrail by Jose Carlos Gonzales Tanaka (MIT, https://github.com/jcgtanaka/agent-guardrail):
out-of-band pending/approved state and one-time consumption. See NOTICE and docs/security.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qrh import signing

MIN_REASON = 10
DEFAULT_TTL_S = 30 * 60
MAX_TTL_S = DEFAULT_TTL_S
PRUNE_AFTER = timedelta(days=1)
SUMMARY_LIMIT = 200
ID_RE = re.compile(r"^[0-9a-f]{12}$")
DOMAIN = "approval"
_PATH_KEYS = ("file_path", "notebook_path", "path")


class ApprovalError(RuntimeError):
    """An approval cannot be created, signed or verified (the message says why)."""


@dataclass(frozen=True)
class Consumed:
    ok: bool
    record: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)  # why candidate tokens were ignored (expired, bad signature, ...)


def approvals_dir(home: str | Path) -> Path:
    return Path(home) / "approvals"


# --- actions and fingerprints ---------------------------------------------------------------------

def tool_action(tool_name: str, tool_input: dict[str, Any], cwd: str = "") -> dict[str, Any]:
    """A tool call as stored in a request: enough to recompute its fingerprint and show it to a human."""
    return {"kind": "tool", "tool_name": tool_name, "tool_input": dict(tool_input), "cwd": cwd}


def parts_action(*parts: str) -> dict[str, Any]:
    """An action without a tool call (Stop claims, CLI gates): a fixed list of strings."""
    return {"kind": "parts", "parts": [str(p) for p in parts]}


def action_fingerprint(tool_name: str, tool_input: dict[str, Any], cwd: str = "") -> str:
    """SHA-256 of the action. A Bash command is hashed raw, with only leading/trailing whitespace stripped."""
    if tool_name == "Bash":
        command = tool_input.get("command")
        body = json.dumps([cwd, command.strip() if isinstance(command, str) else ""])
    else:
        norm = dict(tool_input)
        for k in _PATH_KEYS:
            v = norm.get(k)
            if isinstance(v, str) and v:
                norm[k] = os.path.normpath(v if os.path.isabs(v) else os.path.join(cwd or "", v))
        body = json.dumps(norm, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256((tool_name + "\0" + body).encode("utf-8", "surrogatepass")).hexdigest()


def stable_fingerprint(*parts: str) -> str:
    """Fingerprint for events without a tool call (Stop): hash of the given strings."""
    return hashlib.sha256("\0".join(parts).encode("utf-8", "surrogatepass")).hexdigest()


def fingerprint_of(action: Any) -> str:
    if not isinstance(action, dict):
        raise ApprovalError("the request has no action")
    if action.get("kind") == "tool" and isinstance(action.get("tool_input"), dict):
        return action_fingerprint(str(action.get("tool_name", "")), action["tool_input"], str(action.get("cwd", "")))
    if action.get("kind") == "parts" and isinstance(action.get("parts"), list):
        return stable_fingerprint(*[str(p) for p in action["parts"]])
    raise ApprovalError("the request has an unknown action")


def _short(text: str, limit: int = SUMMARY_LIMIT) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def summarize(action: Any) -> str:
    """What the human is shown: derived from the action, redacted, at most SUMMARY_LIMIT characters."""
    from qrh.hooks.common import redact, redact_command

    if not isinstance(action, dict):
        return "(no action)"
    if action.get("kind") == "parts":
        from qrh import prepush

        if prepush.is_action(action):  # a git push is shown in full: no cut, no redaction of ref names
            return prepush.full_text(action)
        return _short(redact(" | ".join(str(p) for p in action.get("parts", []))))
    tool = str(action.get("tool_name", ""))
    ti = action.get("tool_input") if isinstance(action.get("tool_input"), dict) else {}
    if tool == "Bash":
        return redact_command(str(ti.get("command", "")), SUMMARY_LIMIT)
    path = next((str(ti[k]) for k in _PATH_KEYS if isinstance(ti.get(k), str) and ti.get(k)), "")
    body = next((ti[k] for k in ("content", "new_string", "new_source") if isinstance(ti.get(k), str)), None)
    if body is None and isinstance(ti.get("edits"), list):
        body = "\n".join(str(e.get("new_string", "")) for e in ti["edits"] if isinstance(e, dict))
    text = f"{tool} {path}".strip()
    if body is not None:
        digest = hashlib.sha256(body.encode("utf-8", "surrogatepass")).hexdigest()[:12]
        text += f" [{len(body)} chars, sha256 {digest}] {redact(body)}"
    elif not path:
        text += " " + redact(json.dumps(ti, sort_keys=True, default=str))
    return _short(text)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


# --- files ----------------------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _write_private(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, sort_keys=True))
    os.replace(tmp, path)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _time(value: Any) -> datetime | None:
    try:
        when = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _records(home: Path) -> list[dict[str, Any]]:
    d = approvals_dir(home)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        rec = _read(p)
        if rec is not None and rec.get("id") == p.stem and ID_RE.match(p.stem):
            out.append(rec)
    return out


def _used(home: Path, rid: str) -> bool:
    return (approvals_dir(home) / f"{rid}.used").exists()


def _prune(home: Path, now: datetime) -> None:
    d = approvals_dir(home)
    for p in d.glob("*.json"):
        rec = _read(p)
        exp = _time(rec.get("expires_at")) if rec else None
        if exp is not None and exp < now - PRUNE_AFTER:
            p.unlink(missing_ok=True)
            (d / f"{p.stem}.used").unlink(missing_ok=True)


def invalidate_all(home: str | Path) -> int:
    """Remove every request and marker (after a key rotation). Returns how many requests were dropped."""
    d = approvals_dir(home)
    n = 0
    for p in list(d.glob("*.json")) if d.is_dir() else []:
        p.unlink(missing_ok=True)
        n += 1
    for p in list(d.glob("*.used")) if d.is_dir() else []:
        p.unlink(missing_ok=True)
    return n


# --- request --------------------------------------------------------------------------------------

def request(
    home: str | Path, *, hook: str, action: dict[str, Any], reason: str, session_id: str, now: datetime,
    ttl: int = DEFAULT_TTL_S,
) -> str:
    """Create a pending approval (or return the id of an identical open one) and return its id."""
    home = Path(home)
    if not 0 < ttl <= MAX_TTL_S:
        raise ApprovalError(f"ttl must be between 1 and {MAX_TTL_S} seconds")
    fingerprint = fingerprint_of(action)
    _ensure_dir(approvals_dir(home))
    _prune(home, now)
    for rec in _records(home):
        exp = _time(rec.get("expires_at"))
        if (rec.get("status") == "pending" and rec.get("hook") == hook and rec.get("fingerprint") == fingerprint
                and rec.get("session_id") == session_id and exp is not None and exp > now
                and not _used(home, rec["id"])):
            return str(rec["id"])
    rid = secrets.token_hex(6)
    rec = {
        "id": rid, "hook": hook, "fingerprint": fingerprint, "action": action, "summary": summarize(action),
        "reason": reason[:300], "session_id": session_id, "created_at": now.isoformat(), "ttl_seconds": ttl,
        "expires_at": (now + timedelta(seconds=ttl)).isoformat(), "status": "pending",
    }
    _write_private(approvals_dir(home) / f"{rid}.json", rec)
    return rid


# --- the human step -------------------------------------------------------------------------------

def _check_id(approval_id: str) -> None:
    if not ID_RE.match(approval_id or ""):
        raise ApprovalError(f"{approval_id!r} is not a valid approval id (12 hex characters)")


def check_reason(reason: str) -> str:
    if len((reason or "").strip()) < MIN_REASON:
        raise ApprovalError(f"the approval reason must be at least {MIN_REASON} characters")
    return reason.strip()


def load_pending(home: str | Path, approval_id: str, now: datetime) -> dict[str, Any]:
    """The pending request, verified by recomputation. Its ``summary`` is re-derived from the action."""
    home = Path(home)
    _check_id(approval_id)
    rec = _read(approvals_dir(home) / f"{approval_id}.json")
    if rec is None or rec.get("id") != approval_id:
        raise ApprovalError(f"no pending approval with id {approval_id}")
    if _used(home, approval_id):
        raise ApprovalError(f"approval {approval_id} was already used")
    if rec.get("status") == "approved":
        raise ApprovalError(f"approval {approval_id} is already approved; retry the blocked action")
    if rec.get("status") != "pending":
        raise ApprovalError(f"approval {approval_id} is not pending")
    ttl = rec.get("ttl_seconds")
    created, exp = _time(rec.get("created_at")), _time(rec.get("expires_at"))
    if isinstance(ttl, bool) or not isinstance(ttl, int) or not 0 < ttl <= MAX_TTL_S:
        raise ApprovalError(f"approval {approval_id} has an invalid ttl (max {MAX_TTL_S} s); the request was altered")
    if created is None or exp is None or exp != created + timedelta(seconds=ttl):
        raise ApprovalError(f"approval {approval_id}: expiry does not match creation time plus ttl; "
                            "the request was altered")
    if exp <= now:
        raise ApprovalError(f"approval {approval_id} expired; repeat the action to get a new id")
    if fingerprint_of(rec.get("action")) != rec.get("fingerprint"):
        raise ApprovalError(f"approval {approval_id}: the action does not match its fingerprint; "
                            "the request was altered after the hook created it")
    rec["summary"] = summarize(rec["action"])
    return rec


def describe(rec: dict[str, Any]) -> str:
    """The text a human reads before typing the passphrase."""
    session = rec.get("session_id") or "(none: a CLI gate, not tied to a session)"
    ttl_min = int(rec.get("ttl_seconds") or DEFAULT_TTL_S) // 60
    shown = summarize(rec.get("action"))
    if "\n" in shown:
        shown = "\n" + "\n".join("    " + x for x in shown.splitlines())
    return (
        f"Approval request {rec.get('id')}\n"
        f"  hook:          {rec.get('hook')}\n"
        f"  action:        {shown}\n"
        f"  blocked for:   {rec.get('reason', '')}\n"
        f"  session:       {session}\n"
        f"  request expires at {rec.get('expires_at')}; once approved it is valid once, for {ttl_min} minutes,\n"
        f"  for this exact action in this session only.\n"
        "Judge the action line, not the reason: the reason is written by the hook that blocked.\n"
    )


def _payload(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": 2, "id": rec.get("id"), "hook": rec.get("hook"), "fingerprint": rec.get("fingerprint"),
        "summary_hash": rec.get("summary_hash"), "session_id": rec.get("session_id"),
        "created_at": rec.get("created_at"), "ttl_seconds": rec.get("ttl_seconds"),
        "expires_at": rec.get("expires_at"), "approved_at": rec.get("approved_at"),
        "reason": rec.get("approval_reason"), "block_reason": rec.get("reason"), "key_id": rec.get("key_id"),
    }


def sign_approval(home: str | Path, approval_id: str, reason: str, private_key: Any,
                  now: datetime | None = None) -> dict[str, Any]:
    """Sign a pending request with an unlocked private key (``qrh.human`` gets it from the passphrase)."""
    home = Path(home)
    now = now or datetime.now(timezone.utc)
    _check_id(approval_id)
    why = check_reason(reason)
    rec = load_pending(home, approval_id, now)
    pub = private_key.public_key()
    rec.update(
        status="approved", approved_at=now.isoformat(), approval_reason=why,
        expires_at=(now + timedelta(seconds=int(rec["ttl_seconds"]))).isoformat(),
        key_id=signing.key_id_of(pub), summary_hash=_sha(rec["summary"]),
    )
    rec["signature"] = signing.sign(private_key, DOMAIN, _payload(rec))
    _write_private(approvals_dir(home) / f"{approval_id}.json", rec)
    return rec


# --- consume --------------------------------------------------------------------------------------

def _problem(rec: dict[str, Any], pub: signing.PublicKey, fingerprint: str, session_id: str, now: datetime) -> str | None:
    if rec.get("key_id") != pub.key_id:
        return "signed by a different key (rotated or forged)"
    try:
        derived = fingerprint_of(rec.get("action"))
    except ApprovalError:
        derived = None
    summary_hash = _sha(summarize(rec.get("action")))
    if derived != fingerprint or rec.get("summary_hash") != summary_hash or _sha(str(rec.get("summary"))) != summary_hash:
        return "the action does not match what was signed"
    if not signing.verify(pub, DOMAIN, _payload(rec), rec.get("signature")):
        return "signature does not verify"
    ttl, approved, exp = rec.get("ttl_seconds"), _time(rec.get("approved_at")), _time(rec.get("expires_at"))
    if (isinstance(ttl, bool) or not isinstance(ttl, int) or not 0 < ttl <= MAX_TTL_S or approved is None
            or exp is None or exp != approved + timedelta(seconds=ttl)):
        return "invalid validity window"
    if rec.get("session_id") != session_id:
        return "approved for a different session"
    if exp <= now:
        return "expired"
    return None


def consume(home: str | Path, *, hook: str, fingerprint: str, session_id: str, now: datetime) -> Consumed:
    """Use one valid approval for this exact action in this session. Raises ApprovalError if the key is unusable."""
    home = Path(home)
    notes: list[str] = []
    candidates = [r for r in _records(home) if r.get("status") == "approved" and r.get("hook") == hook
                  and r.get("fingerprint") == fingerprint and not _used(home, str(r["id"]))]
    if not candidates:
        return Consumed(False, None, notes)
    try:
        pub = signing.load_public(home)
    except signing.SigningError as exc:
        raise ApprovalError(str(exc)) from exc
    for rec in candidates:
        rid = str(rec["id"])
        if "signature" not in rec:
            notes.append(f"approval {rid} ignored: legacy HMAC approval, no longer valid (approve again)")
            continue
        if pub is None:
            notes.append(f"approval {rid} ignored: no approval key (run `qrh approve --init` in your terminal)")
            continue
        why = _problem(rec, pub, fingerprint, session_id, now)
        if why is not None:
            notes.append(f"approval {rid} ignored: {why}")
            continue
        try:  # exclusive creation is the single-use guarantee, also against a racing hook process
            fd = os.open(approvals_dir(home) / f"{rid}.used", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            notes.append(f"approval {rid} ignored: already used")
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(now.isoformat())
        return Consumed(True, rec, notes)
    return Consumed(False, None, notes)
