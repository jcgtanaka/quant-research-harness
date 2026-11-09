"""Claude Code hooks: thin entry points (``qrh hook <name>``) over library code.

One module per hook, plus ``common`` (payload, config, redaction, logging)
and ``runner`` (dispatch, fail-closed policy, human approvals and log handling).
"""
