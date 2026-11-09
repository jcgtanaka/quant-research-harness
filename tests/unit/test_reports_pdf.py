import re

from qrh.reports.catalog import figure_specs
from qrh.reports.pdf import write_pdf


def figures(make_data, stage="S4"):
    data = make_data(stage)
    return data, [spec.chart() for spec in figure_specs(data)]


def pages(blob):
    return len(re.findall(rb"/Type /Page\b(?!s)", blob))


def test_pdf_has_one_page_per_figure_and_full_receipt_hash_in_metadata(tmp_path, make_data):
    data, figs = figures(make_data)
    out = tmp_path / "s4.pdf"
    write_pdf(out, figs, title="S4 report", subject=f"trial {data.trial_id} stage S4 receipt {data.receipt_hash}")
    blob = out.read_bytes()
    assert blob.startswith(b"%PDF") and pages(blob) == len(figs) == 3
    assert data.receipt_hash.encode() in blob


def test_pdf_has_no_timestamps(tmp_path, make_data):
    data, figs = figures(make_data)
    out = tmp_path / "s4.pdf"
    write_pdf(out, figs, title="t", subject="s")
    assert b"/CreationDate" not in out.read_bytes() and b"/ModDate" not in out.read_bytes()


def test_identical_artifacts_give_byte_identical_pdfs(tmp_path, make_data):
    blobs = []
    for i in range(2):
        data, figs = figures(make_data, "S6")
        out = tmp_path / f"{i}.pdf"
        write_pdf(out, figs, title="t", subject=data.receipt_hash)
        blobs.append(out.read_bytes())
    assert blobs[0] == blobs[1]


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path, make_data):
    data, figs = figures(make_data)
    write_pdf(tmp_path / "x.pdf", figs, title="t", subject="s")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["x.pdf"]


def test_pdf_embeds_only_the_pinned_font(tmp_path, make_data):
    data, figs = figures(make_data)
    out = tmp_path / "f.pdf"
    write_pdf(out, figs, title="t", subject="s")
    names = set(re.findall(rb"/BaseFont /(\S+)", out.read_bytes()))
    assert names and all(b"DejaVuSans" in n for n in names), names
