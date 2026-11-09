"""The dashboard's static page: served at / and /static/<file>, nothing else, fully local."""

from __future__ import annotations

import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from qrh.ui.server import make_server

STATIC = Path(__file__).resolve().parents[2] / "src" / "qrh" / "ui" / "static"


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("QRH_HOME", str(tmp_path))
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def test_index_is_html(base):
    status, ctype, body = _get(base + "/")
    assert status == 200
    assert ctype.startswith("text/html")
    assert b"<html" in body


def test_app_js_is_javascript(base):
    status, ctype, body = _get(base + "/static/app.js")
    assert status == 200
    assert "javascript" in ctype
    assert body


def test_app_css_is_css(base):
    status, ctype, _ = _get(base + "/static/app.css")
    assert status == 200
    assert ctype.startswith("text/css")


@pytest.mark.parametrize("path", [
    "/static/../server.py", "/static/..%2Fserver.py", "/static/%2e%2e/server.py",
    "/static/", "/static/other.txt", "/static/app.js/x", "/static/%2e%2e%2fdata.py", "/index.html",
    "/static/app.js%00.py",
])
def test_everything_else_is_404(base, path):
    status, _, body = _get(base + path)
    assert status == 404
    assert b"import " not in body


def test_static_is_get_only(base):
    req = urllib.request.Request(base + "/", data=b"{}", method="POST")
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=5)
    assert ei.value.code == 405


def test_html_has_no_external_url():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for m in re.finditer(r"""(?:src|href)\s*=\s*["']([^"']*)["']""", html):
        assert not re.match(r"(?i)(https?:)?//", m.group(1)), m.group(1)
    assert "http://" not in html and "https://" not in html


def test_css_and_js_have_no_external_url():
    for name in ("app.css", "app.js"):
        text = (STATIC / name).read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, name
        assert "@import" not in text, name


def test_css_defines_tokens_on_root_with_dark_mode():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    root = re.search(r":root\s*\{([^}]*)\}", css)
    assert root
    for token in ("--bg", "--text", "--pass", "--fail", "--amber", "--pass-conditional", "--missing"):
        assert re.search(rf"{re.escape(token)}\s*:", root.group(1)), token
    assert "prefers-color-scheme: dark" in css
    assert 'data-theme="dark"' in css and 'data-theme="light"' in css
    assert re.search(r"body\s*\{[^}]*background", css)


def test_js_uses_textcontent_not_innerhtml():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js
    assert "outerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js
    assert "textContent" in js


def test_js_reads_only_api_and_never_writes():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "/api/trials" in js
    for method in ("POST", "PUT", "DELETE", "PATCH"):
        assert method not in js


def test_html_has_bench_and_warnings_tabs():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(r"<nav[^>]*>", html)
    for label, target in (("Trials", "#/"), ("Bench", "#/bench"), ("Warnings", "#/warnings")):
        assert re.search(rf'<a[^>]*href="{re.escape(target)}"[^>]*>\s*{label}\s*</a>', html), label


def test_html_has_checks_tab_link():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert re.search(r'<a[^>]*href="#/checks"[^>]*>\s*Checks\s*</a>', html)


def test_js_calls_only_bench_and_trials_api_paths():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '"/api/bench"' in js
    assert "/api/trials/" in js and "/warnings" in js
    paths = re.findall(r"""["'](/api[^"']*)["']""", js)
    assert paths
    for p in paths:
        assert re.match(r"^/api/(bench|trials|parity|data-checks)(/|$)", p), p


def test_js_bench_and_warnings_views_have_required_text_and_no_html_injection():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for text in ("No bench results yet.", "No warnings for this trial."):
        assert text in js, text
    assert "innerHTML" not in js and "outerHTML" not in js and "insertAdjacentHTML" not in js
    assert "<select" not in js or "createElement" in js
    # trial id is chosen from the list, never typed
    assert "<input" not in js and '"input"' not in js and "'input'" not in js
    assert '"select"' in js
    assert "fetch(" in js and "method: \"GET\"" in js


def test_css_has_bench_and_warnings_styles_using_tokens_only():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    for cls in (".tabs", ".bench-card", ".metrics-table", ".sev-high"):
        assert cls in css, cls
    body = css.split("* { box-sizing", 1)[1]
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", body), "colours must come from :root tokens"
    assert "overflow-wrap" in css


def _assert_only_refresh_button(html):
    """No write controls. The one allowed button is the read-only Refresh button (id "refresh")."""
    assert "<form" not in html and "<input" not in html
    buttons = re.findall(r"<button\b([^>]*)>(.*?)</button>", html, flags=re.S)
    assert html.count("<button") == len(buttons) == 1
    attrs, text = buttons[0]
    assert re.search(r'\bid="refresh"', attrs)
    assert not re.search(r"""type\s*=\s*["']?submit""", attrs, flags=re.I)
    assert not re.search(r"(?i)start|run|save|approve|submit", attrs + " " + text)


def test_html_shows_no_write_controls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    _assert_only_refresh_button(html)


# ---- "Start here" front page ----
def _first_tab_and_js():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    return html, js


def test_start_here_is_first_tab_and_default_page():
    html, js = _first_tab_and_js()
    links = re.findall(r'<a[^>]*href="([^"]*)"[^>]*>\s*([^<]*?)\s*</a>', html.split("</nav>")[0])
    assert links[0] == ("#/start", "Start here")
    assert 'id="tab-start"' in html
    # no hash at all opens the Start here page; "#/" still opens Trials
    assert re.search(r"window\.location\.hash\s*\|\|\s*\"#/start\"", js) or "showStart" in js
    assert "hash === \"#/start\"" in js and "showStart()" in js
    assert re.search(r"if\s*\(\s*!window\.location\.hash[^)]*\)\s*\{[^}]*showStart\(\)", js)


def test_start_here_has_section_headings():
    _, js = _first_tab_and_js()
    for heading in (
        "What the harness is", "What it does not do", "The steps to follow",
        "How a research run is started today", "What each tab is for", "Glossary", "A note on safety",
    ):
        assert heading in js, heading


def test_start_here_says_there_is_no_start_button_yet():
    _, js = _first_tab_and_js()
    assert "there is no start button yet" in js.lower()
    assert "Ask the AI assistant in VS Code to run the research for your trial. The dashboard shows the results as they are recorded." in js


def test_start_here_glossary_has_eight_terms():
    _, js = _first_tab_and_js()
    for term in ("Stage", "Verdict", "PASS", "FAIL", "Warning", "Backtest", "Paper trading", "Live trading"):
        assert f'term: "{term}"' in js, term
    assert len(re.findall(r'term:\s*"', js)) == 8


def test_start_here_steps_name_their_tabs_and_six_steps():
    _, js = _first_tab_and_js()
    assert len(re.findall(r'step:\s*"', js)) == 6
    for tab in ("Trials", "Warnings", "Checks"):
        assert tab in js


def test_start_here_page_stays_read_only_and_plain():
    html, js = _first_tab_and_js()
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js
    _assert_only_refresh_button(html)
    assert "<form" not in js and '"button"' not in js and '"form"' not in js
    assert ".start-section" in css
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", css.split("* { box-sizing", 1)[1])


# ---- design brief: layout, checklist, empty states, freshness, themes ----
def _all_static():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    return html, js, css


def test_start_page_has_four_step_checklist_with_done_or_to_do():
    _, js, _ = _all_static()
    assert len(re.findall(r'\{\s*check:\s*"', js)) == 4
    assert '"Done"' in js and '"To do"' in js
    for title in ("Describe your idea", "Provide data", "Backtest and check", "Read the verdict"):
        assert title in js, title


def test_sidebar_nav_with_sticky_header_and_narrow_layout():
    html, _, css = _all_static()
    assert re.search(r'<nav[^>]*class="sidebar[ "]', html)
    assert re.search(r"<header[^>]*>", html) and "position: sticky" in css
    assert "220px" in css
    assert "@media (max-width: 760px)" in css
    assert "min-height: 44px" in css
    _assert_only_refresh_button(html)


def test_empty_state_wording_pattern_and_dashed_box():
    _, js, css = _all_static()
    assert "empty-state" in js
    # title names what is missing; the second line says how to fill it
    assert 'emptyState("No trials yet.",' in js
    assert "Ask the assistant in VS Code to start a research run; results appear here." in js
    assert len(re.findall(r"emptyState\(", js)) >= 4
    assert re.search(r"\.empty-state\s*\{[^}]*dashed", css)


def test_freshness_line_wording():
    _, js, css = _all_static()
    assert "Data through " in js
    assert "freshness" in css


def test_activity_list_discloses_cap_and_end():
    _, js, _ = _all_static()
    assert "Showing the latest " in js
    assert "End of list." in js


def test_theme_choice_saved_in_localstorage_inside_try_catch():
    html, js, css = _all_static()
    assert re.search(r"try\s*\{[^}]*localStorage\.setItem[^}]*\}\s*catch", js)
    assert re.search(r"try\s*\{[^}]*localStorage\.getItem[^}]*\}\s*catch", js)
    assert "data-theme" in js
    assert 'id="theme-select"' in html


def test_typography_rules_no_caps_no_tracking_tabular_numbers():
    html, js, css = _all_static()
    assert "text-transform" not in css
    assert "letter-spacing" not in css
    assert "tabular-nums" in css
    assert "toUpperCase()" not in js.replace("charAt(0).toUpperCase()", "")


def test_no_inner_html_and_no_inline_handlers():
    html, js, _ = _all_static()
    assert "innerHTML" not in js
    assert not re.search(r"\son[a-z]+\s*=", html)
    assert "<script>" not in html


# ---- Refresh button: read-only, header only ----
def test_header_has_one_refresh_button():
    html, _, _ = _all_static()
    header = re.search(r"<header\b.*?</header>", html, flags=re.S).group(0)
    assert re.search(r'<button\b[^>]*\bid="refresh"[^>]*>\s*Refresh\s*</button>', header)
    assert html.count("<button") == 1
    assert 'id="updated-at"' in header
    assert not re.search(r"(?i)<button[^>]*(start|run|save|approve|submit)", html)


def test_refresh_uses_only_existing_get_helper_and_no_new_paths():
    _, js, _ = _all_static()
    assert 'getElementById("refresh")' in js
    assert js.count("fetch(") == 1  # only inside fetchJson, which sends GET
    assert 'method: "GET"' in js
    paths = set(re.findall(r"""["'](/api[^"']*)["']""", js))
    assert paths <= {"/api/trials", "/api/trials/", "/api/bench", "/api/parity", "/api/data-checks"}, paths


def test_refresh_label_and_updated_at_use_textcontent():
    _, js, _ = _all_static()
    assert '"Refreshing..."' in js and "disabled" in js
    assert re.search(r'updated-at"\)', js)
    assert re.search(r'\.textContent\s*=\s*"Updated at "', js)
    assert "innerHTML" not in js


def test_refresh_button_css_uses_tokens_min_height_and_focus_ring():
    _, _, css = _all_static()
    block = re.search(r"\.refresh-button\s*\{([^}]*)\}", css)
    assert block and "min-height: 44px" in block.group(1) and "var(--" in block.group(1)
    assert re.search(r"\.refresh-button:focus-visible\s*\{[^}]*outline:\s*3px", css)
