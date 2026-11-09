"""A deterministic fake model for tests and examples (no network, no randomness).

The answer is a pure function of the prompt text. Prompts are expected to look like::

    Decision date: 2021-01-05
    Closes, oldest first: 100, 101, 103
    Flat threshold (fraction): 0.01

The default rule is a trend rule on those closes (``UP``, ``DOWN`` or ``FLAT``). With ``oracle`` and
``oracle_until`` the stub "knows the future" before a date: for a decision date earlier than ``oracle_until`` it
returns the true next-bar answer from ``oracle`` (decision date -> answer). That models a language model whose
training data covers the period (see ``examples/08_llm_contamination``).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

_DATE = re.compile(r"Decision date:\s*(\d{4}-\d{2}-\d{2})")
_CLOSES = re.compile(r"Closes[^:\n]*:\s*([^\n]+)")
_THRESHOLD = re.compile(r"Flat threshold[^:\n]*:\s*([0-9.eE+-]+)")


class StubBackend:
    def __init__(self, *, oracle: Mapping[str, str] | None = None, oracle_until: str | None = None):
        if (oracle is None) != (oracle_until is None):
            raise ValueError("oracle and oracle_until go together")
        self.oracle = dict(oracle or {})
        self.oracle_until = oracle_until

    def __call__(self, model_id: str, prompt: str, params: dict[str, Any]) -> str:
        date = _DATE.search(prompt)
        if date and self.oracle_until is not None and date[1] < self.oracle_until and date[1] in self.oracle:
            return self.oracle[date[1]]
        closes = _CLOSES.search(prompt)
        if closes:
            try:
                values = [float(x) for x in closes[1].split(",") if x.strip()]
            except ValueError:
                values = []
            if len(values) >= 2 and values[0] != 0:
                flat = _THRESHOLD.search(prompt)
                threshold = float(flat[1]) if flat else 0.0
                change = values[-1] / values[0] - 1.0
                return "FLAT" if abs(change) < threshold else ("UP" if change > 0 else "DOWN")
        return "UP" if hashlib.sha256(prompt.encode("utf-8")).digest()[0] % 2 == 0 else "DOWN"
