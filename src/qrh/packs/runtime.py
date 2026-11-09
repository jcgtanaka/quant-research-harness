"""The active packs, resolved for one run. Everything the pipeline needs from packs goes through this view."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qrh.hashing import hash_json
from qrh.packs.model import Account, Lesson, LintRule, Pack, Question
from qrh.packs.store import active_packs, pack_record


class PackView:
    def __init__(self, packs: list[Pack], home: str | Path | None = None):
        self.packs = packs
        self._home = home

    @classmethod
    def load(cls, home: str | Path) -> "PackView":
        """Raises PackError when a registered pack cannot be loaded (callers fail closed)."""
        return cls(active_packs(home), home)

    @classmethod
    def empty(cls) -> "PackView":
        return cls([])

    def __bool__(self) -> bool:
        return bool(self.packs)

    def record(self) -> dict[str, dict[str, Any]]:
        return pack_record(self.packs, self._home) if self.packs and self._home is not None else {}

    def questions(self) -> list[Question]:
        """The merged bank. The first pack to define an id supplies it; any pack may mark it critical."""
        merged: dict[str, Question] = {}
        for pack in self.packs:
            for q in pack.questions:
                if q.id not in merged:
                    merged[q.id] = q
                elif q.critical and not merged[q.id].critical:
                    merged[q.id] = merged[q.id].model_copy(update={"critical": True})
        return list(merged.values())

    def matched_lessons(self, trial: dict[str, Any]) -> list[tuple[str, Lesson]]:
        out: list[tuple[str, Lesson]] = []
        for pack in self.packs:
            if not pack.manifest.applies_to.matches(trial):
                continue
            for lesson in pack.lessons:
                if lesson.applies_to.matches(trial) and lesson.trigger.matches(trial):
                    out.append((pack.name, lesson))
        return out

    def required_questions(self, trial: dict[str, Any]) -> set[str]:
        return {str(les.requirement["interview_question"]) for _p, les in self.matched_lessons(trial)
                if les.action == "require" and les.requirement and "interview_question" in les.requirement}

    def account(self) -> tuple[str, Account] | None:
        """The account of the last pack that has one (private packs come after public ones)."""
        found = [(p.name, p.account) for p in self.packs if p.account is not None]
        return found[-1] if found else None

    def account_hash(self) -> str:
        acct = self.account()
        return hash_json(acct[1].model_dump()) if acct else ""

    def lint_rules(self) -> list[LintRule]:
        return [r for p in self.packs for r in p.lint_rules]
