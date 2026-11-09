"""The hook table: name, event, matcher, timeout (seconds) and fail mode. Single source for settings output."""

from __future__ import annotations

from qrh.hooks.runner import HookSpec

_EDIT = "Edit|Write|MultiEdit"
_GUARDED = "Bash|Read|Edit|Write|MultiEdit|NotebookEdit|Grep|Glob|WebFetch|mcp__.*"  # everything that can touch the approval state

HOOKS: dict[str, HookSpec] = {
    s.name: s
    for s in (
        HookSpec("session-context", "SessionStart", "startup|resume|clear|compact", 10, False, "qrh.hooks.session_context"),
        HookSpec("prior-attempts", "UserPromptSubmit", None, 10, False, "qrh.hooks.prior_attempts"),
        HookSpec("lookahead-lint", "PostToolUse", _EDIT, 15, True, "qrh.hooks.lookahead_lint"),
        HookSpec("approval-guard", "PreToolUse", _GUARDED, 10, True, "qrh.hooks.approval_guard",
                 approvable=False, needs_config=False),
        HookSpec("data-gate", "PreToolUse", "Bash", 15, True, "qrh.hooks.data_gate"),
        HookSpec("data-guard", "PreToolUse", "Read|Bash|Grep", 10, True, "qrh.hooks.data_guard",
                 risk="reading a private dataset sends licensed data to the cloud model, which cannot be taken back"),
        HookSpec("impact-check", "PreToolUse", _EDIT, 10, False, "qrh.hooks.impact_check"),
        HookSpec("pre-restart-guard", "PreToolUse", "Bash", 20, True, "qrh.hooks.pre_restart_guard"),
        HookSpec("publish-guard", "PreToolUse", "Bash|Write|Edit|MultiEdit", 10, True, "qrh.hooks.publish_guard",
                 risk="publishing code or weakening the checks that stop secrets and private material from "
                      "leaving this machine cannot be taken back"),
        HookSpec("runaway-guard", "PreToolUse", None, 10, True, "qrh.hooks.runaway"),
        HookSpec("runaway-failure", "PostToolUseFailure", None, 10, False, "qrh.hooks.runaway_failure"),
        HookSpec("claim-check", "Stop", None, 15, True, "qrh.hooks.claim_check"),
        HookSpec("receipt-required", "Stop", None, 15, True, "qrh.hooks.receipt_required"),
    )
}
