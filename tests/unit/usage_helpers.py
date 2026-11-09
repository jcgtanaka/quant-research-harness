"""Synthetic Claude Code transcripts for the usage tests. Nothing here is real data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SECRET = "SECRET-PROMPT-TEXT"
SECRET_PATH = "/private/never/shown/file.py"


def usage(inp=0, out=0, cr=0, cw5=0, cw1=0, split=True) -> dict[str, Any]:
    u: dict[str, Any] = {
        "input_tokens": inp, "output_tokens": out, "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cw5 + cw1, "service_tier": "standard", "speed": "standard",
    }
    if split:
        u["cache_creation"] = {"ephemeral_5m_input_tokens": cw5, "ephemeral_1h_input_tokens": cw1}
    return u


def assistant(
    mid: str | None, model: str, ts: str, *, inp=0, out=0, cr=0, cw5=0, cw1=0, tools: tuple[str, ...] = (),
    tool_ids: tuple[str, ...] = (), sidechain=False, cwd="/work/projA", split=True, repeat=1,
    with_usage=True, stop: Any = "unset",
) -> list[dict[str, Any]]:
    """``repeat`` records for one message id, one content block each (as Claude Code writes them)."""
    blocks: list[dict[str, Any]] = [{"type": "text", "text": SECRET}]
    for i, name in enumerate(tools):
        tid = tool_ids[i] if i < len(tool_ids) else f"tu_{mid}_{i}"
        blocks.append({"type": "tool_use", "id": tid, "name": name, "input": {"file_path": SECRET_PATH}})
    out_records = []
    for i in range(max(repeat, len(blocks))):
        msg: dict[str, Any] = {"model": model, "role": "assistant", "content": [blocks[i % len(blocks)]]}
        if mid is not None:
            msg["id"] = mid
        if stop != "unset":  # an explicit None is a message that never completed (stream interrupted)
            msg["stop_reason"] = stop
        if with_usage:
            msg["usage"] = usage(inp, out, cr, cw5, cw1, split)
        out_records.append({
            "type": "assistant", "timestamp": ts, "isSidechain": sidechain, "cwd": cwd, "sessionId": "x",
            "message": msg,
        })
    return out_records


def user(ts: str, cwd="/work/projA") -> dict[str, Any]:
    return {"type": "user", "timestamp": ts, "isSidechain": False, "cwd": cwd,
            "message": {"role": "user", "content": SECRET}}


def compact(ts: str) -> dict[str, Any]:
    return {"type": "system", "subtype": "compact_boundary", "timestamp": ts, "cwd": "/work/projA"}


def cost_state(total: float, models: dict[str, dict[str, float]]) -> dict[str, Any]:
    """models: name -> dict(inp, out, cr, cw, cost)."""
    return {
        "type": "cost-state", "totalCostUSD": total, "totalAPIDuration": 1000, "totalDuration": 5000,
        "totalLinesAdded": 3, "totalLinesRemoved": 1,
        "modelUsage": {
            m: {"inputTokens": v.get("inp", 0), "outputTokens": v.get("out", 0),
                "cacheReadInputTokens": v.get("cr", 0), "cacheCreationInputTokens": v.get("cw", 0),
                "thinkingTokens": 0, "webSearchRequests": 0, "costUSD": v.get("cost", 0.0)}
            for m, v in models.items()
        },
    }


def write_jsonl(path: Path, records: list[Any], raw_lines: tuple[str, ...] = ()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records] + list(raw_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def flat(*groups: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for g in groups:
        out.extend(g if isinstance(g, list) else [g])
    return out


def default_thresholds() -> dict[str, Any]:
    return {"cache_cold_pct": 60, "context_high_tokens": 300000, "mechanical_tool_share": 0.8,
            "mechanical_min_tools": 5}


def default_prices() -> dict[str, Any]:
    return {"models": {
        "model-big": {"input": 10.0, "output": 50.0, "cache_read": 1.0, "cache_write_5m": 12.5,
                      "cache_write_1h": 20.0, "tier": "frontier"},
        "model-small": {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write_5m": 1.25,
                        "cache_write_1h": 2.0, "tier": "small"},
    }}


def write_settings(home: Path, thresholds: Any = "default", prices: Any = "default") -> None:
    d = home / "usage"
    d.mkdir(parents=True, exist_ok=True)
    if thresholds is not None:
        t = default_thresholds() if thresholds == "default" else thresholds
        (d / "thresholds.json").write_text(json.dumps(t), encoding="utf-8")
    if prices is not None:
        p = default_prices() if prices == "default" else prices
        (d / "prices.json").write_text(json.dumps(p), encoding="utf-8")


T = "2026-03-01T10:{:02d}:00.000Z"


def build_claude(root: Path) -> Path:
    proj = root / "projects" / "-work-projA"
    main = flat(
        user(T.format(0)),
        assistant("r1", "model-big", T.format(1), inp=100, out=10, cw5=900, tools=("Read", "Grep"), repeat=2),
        assistant("r2", "model-big", T.format(2), inp=10, out=5, cr=1000),
        assistant("r3", "model-big", T.format(3), inp=500, out=5, cr=100, cw1=400, tools=("Edit",)),
        compact(T.format(4)),
        assistant("r4", "model-big", T.format(5), inp=800, out=5, cw5=200),
        assistant("r5", "model-big", T.format(6), inp=900, out=5, cr=50, cw5=50),
        assistant("sc", "model-small", T.format(7), inp=7, out=3, sidechain=True),
        cost_state(3.5, {"model-big": {"inp": 2310, "out": 30, "cr": 1150, "cw": 1550, "cost": 3.0},
                         "model-small": {"inp": 27, "out": 13, "cost": 0.5}}),
    )
    write_jsonl(proj / "s1.jsonl", main, raw_lines=("{broken",))
    write_jsonl(proj / "s1" / "subagents" / "agent-a1.jsonl",
                flat(assistant("x1", "model-small", T.format(2), inp=20, out=10, repeat=2, sidechain=True)))
    write_jsonl(proj / "s2.jsonl", flat(
        assistant("e1", "model-big", T.format(10), inp=1_000_000, out=100_000)))
    write_jsonl(proj / "s2" / "subagents" / "agent-b.jsonl",
                flat(assistant("e2", "model-small", T.format(11), inp=1_000_000, sidechain=True)))
    write_jsonl(proj / "s3.jsonl", flat(assistant("u1", "mystery", T.format(20), inp=5, out=5)))
    return root


def prompt(ts: str, text: str = SECRET, cwd="/work/projA") -> dict[str, Any]:
    """A typed user prompt (a turn boundary)."""
    return {"type": "user", "timestamp": ts, "isSidechain": False, "cwd": cwd,
            "message": {"role": "user", "content": text}}


def tool_call(mid: str, model: str, ts: str, name: str, tid: str, tool_input: dict[str, Any], *,
              result: str = "ok", error: bool = False, **usage_kw: Any) -> list[dict[str, Any]]:
    """An assistant record that calls one tool, followed by the user record carrying its result."""
    use = assistant(mid, model, ts, **usage_kw)[0]
    use["message"]["content"] = [{"type": "tool_use", "id": tid, "name": name, "input": tool_input}]
    res = {"type": "user", "timestamp": ts, "isSidechain": False, "cwd": "/work/projA",
           "message": {"role": "user", "content": [
               {"type": "tool_result", "tool_use_id": tid, "content": result, "is_error": error}]}}
    return [use, res]
