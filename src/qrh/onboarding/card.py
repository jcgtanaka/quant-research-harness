"""The S1 "dataset card" and onboarding table of a dataset, as one JSON-safe document (stored as an S1 artifact).

Only the stored profile (aggregates, never rows) and the onboarding model are used; claims and evidence are plain text.
"""

from __future__ import annotations

from typing import Any

from qrh.onboarding import store, workflow
from qrh.onboarding.profile import profile_hash

MAX_COLUMNS = 40
MAX_RELATIONSHIPS = 12


def _relationship_text(r: dict[str, Any]) -> str:
    kind = r["kind"]
    if kind == "a_ge_b":
        return f"{r['a']} >= {r['b']} on every row"
    if kind == "inside":
        return f"{r['value']} always between {r['low']} and {r['high']}"
    if kind == "close":
        return f"{r['a']} within {r['rel_tol']:g} of {r['b']}" + (" (identical)" if r.get("identical") else "")
    if kind == "return_like":
        return f"{r['column']} looks like a return series" + (f" (median |x| {r['median_abs']:g})" if "median_abs" in r else "")
    if kind == "percent_like":
        return f"{r['column']} looks like a percent (0 to 100)"
    if kind == "rate_like":
        return f"{r['column']} looks like a rate ({r['scale']})"
    return kind


def profile_card(profile: dict[str, Any]) -> dict[str, Any]:
    ti = profile["time_index"]
    cols = []
    for name, c in list(profile["columns"].items())[:MAX_COLUMNS]:
        cols.append({
            "name": name, "kind": c["kind"], "dtype": c["dtype"], "null_rate": c["null_rate"], "distinct": c["distinct"],
            "min": c.get("min"), "max": c.get("max"),
            "note": ("shape " + c["shapes"][0]["shape"]) if c["kind"] == "string" and c.get("shapes") else "",
        })
    rels = [_relationship_text(r) for r in profile["relationships"]]
    return {
        "rows": profile["rows"], "no_values": profile["no_values"], "profile_hash": profile_hash(profile),
        "time": {k: ti.get(k) for k in ("column", "timezone", "inferred_frequency", "gap_count", "duplicates",
                                         "monotonic_increasing", "first", "last", "observed_density")},
        "columns": cols, "columns_total": len(profile["columns"]),
        "relationships": rels[:MAX_RELATIONSHIPS], "relationships_total": len(rels),
        "hints": [f"{h['column']} ~ {h['hint']}" for h in profile["hints"]],
    }


def dataset_doc(cat: Any, name: str) -> dict[str, Any] | None:
    """The card document of a dataset, or None when it has neither a stored profile nor an onboarding for its file."""
    rec = cat.get_dataset(name)
    if rec is None:
        return None
    stored = store.get_profile(cat.conn, rec.get("content_hash"))
    desc = workflow.describe(cat, name)
    if stored is None and desc["version"] is None:
        return None
    return {
        "name": name, "privacy": desc["privacy"], "state": desc["state"], "version": desc["version"],
        "content_hash": rec.get("content_hash"), "model_hash": desc["model_hash"],
        "card": profile_card(stored[0]) if stored else None,
        "fields": desc["fields"],
        "pending_questions": [{"id": q["id"], "critical": q["critical"], "status": q["status"]}
                              for q in desc["pending_questions"]],
    }


def receipt_record(cat: Any, name: str) -> dict[str, Any]:
    return store.dataset_state(cat.conn, name)
