"""Write the bench reports beside the receipt: ``report.txt``, ``report.pdf`` and ``chart.json``.

The text is built from the receipt; the chart JSON is built from the receipt; the PDF is drawn from the chart
JSON content only (one-source rule). Output is deterministic: identical receipts give identical files.
"""

from __future__ import annotations

import json
from pathlib import Path

from qrh.bench.chartdata import SCHEMA_VERSION, build_figures
from qrh.bench.charts import render_pdf
from qrh.bench.receipt import BenchReceipt
from qrh.bench.textreport import format_receipt


def render_bench(receipt: BenchReceipt, folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    figures = build_figures(receipt)
    doc = {
        "schema_version": SCHEMA_VERSION, "bench_id": receipt.bench_id(), "receipt_hash": receipt.receipt_hash(),
        "figures": figures,
    }
    chart = folder / "chart.json"
    chart.write_text(json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    text = folder / "report.txt"
    text.write_text(format_receipt(receipt), encoding="utf-8")
    pdf = folder / "report.pdf"
    loaded = json.loads(chart.read_text(encoding="utf-8"))  # the PDF is drawn from what is on disk
    render_pdf(pdf, loaded["figures"], loaded["bench_id"], loaded["receipt_hash"])
    return [text, pdf, chart]
