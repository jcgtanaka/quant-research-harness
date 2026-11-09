"""The banner every block that needs a HUMAN decision starts with (standing rule, 2026-10-23).

Used by approval-requiring hook blocks, runaway-guard trips, config-integrity blocks and the CLI gates that wait
for an approval (``qrh methods promote|demote|deprecate``, ``qrh pack add`` and ``approve-lesson``).
"""

from __future__ import annotations

BANNER = (
    "##################\n"
    "WARNING: DECISION TO BE MADE BY YOU!\n"
    "##################\n"
    "Content of the warning:\n"
)

RELAY = (
    "Agent: relay this whole banner to the user verbatim, then STOP. Do not retry the action, do not run the "
    "command above yourself, and do not work around this block; wait for the user's decision.\n"
)


def decision_banner(details: str) -> str:
    """BANNER, then the details (hook, action, why, risk, the human's command), then the relay instruction."""
    return BANNER + details.rstrip("\n") + "\n" + RELAY
