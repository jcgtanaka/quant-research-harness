import pytest


@pytest.fixture(autouse=True)
def reports_off_by_default(monkeypatch):
    """Rendering PDFs after every stage costs seconds per pipeline run; only the report tests turn it on."""
    monkeypatch.setenv("QRH_REPORTS", "0")


@pytest.fixture(autouse=True)
def fast_approval_keys(monkeypatch):
    """Approval keys use a cheap scrypt cost and no wrong-passphrase delay in tests (a code seam, never an env var)."""
    from qrh import signing

    monkeypatch.setattr(signing, "KDF_N", 2 ** 4)
    monkeypatch.setattr(signing, "_sleep", lambda seconds: None)
