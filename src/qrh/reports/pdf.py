"""PDF assembly. Output is deterministic: no timestamps, fixed producer, pinned font."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from matplotlib import rc_context
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from qrh.reports import theme

PRODUCER = "quant-research-harness"


def write_pdf(path: str | Path, figures: Sequence[Figure], *, title: str, subject: str, keywords: str = "") -> None:
    """Write one page per figure. ``subject`` carries the full receipt hash(es) in the PDF metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    metadata = {
        "Title": title, "Subject": subject, "Keywords": keywords or "synthetic, research",
        "Creator": PRODUCER, "Producer": PRODUCER, "CreationDate": None, "ModDate": None,
    }
    try:
        with rc_context(theme.RC), PdfPages(tmp, metadata=metadata) as pdf:
            for fig in figures:
                pdf.savefig(fig)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
