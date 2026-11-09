/* Read-only dashboard. It only reads the /api endpoints and shows what they return.
   It never computes a verdict. All server text is inserted with textContent. */
(function () {
  "use strict";

  var STAGES = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"];

  // Display wording only: how a verdict string the API returned is shown. Unknown values are shown as received.
  var LIGHTS = {
    PASS: { css: "pass", label: "Passed" },
    FAIL: { css: "fail", label: "Failed" },
    BLOCKED: { css: "amber", label: "Blocked" },
    INCONCLUSIVE: { css: "amber", label: "Inconclusive" },
    PASS_CONDITIONAL: { css: "pass-conditional", label: "Passed with conditions" }
  };
  var MISSING = { css: "missing", label: "Not run yet" };

  var app = document.getElementById("app");

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function clear(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  function sentence(text) {
    var s = String(text);
    return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
  }

  /* An empty state names what is missing, then says how to fill it. */
  function emptyState(title, howTo) {
    var box = el("div", "empty-state");
    box.appendChild(el("p", "empty-title", title));
    box.appendChild(el("p", "muted", howTo));
    return box;
  }

  /* A status card: number, label and a status word beside the colour. */
  function statCard(number, label, css, word, through) {
    var card = el("li", "stat-card");
    card.appendChild(el("p", "stat-number", number));
    card.appendChild(el("p", "stat-label", label));
    var pill = el("p", "light " + css);
    pill.appendChild(el("span", "dot"));
    pill.appendChild(el("span", null, word));
    card.appendChild(pill);
    if (through) { card.appendChild(el("p", "freshness muted", "Data through " + String(through))); }
    return card;
  }

  /* ---- theme: saved choice is optional; storage may be blocked ---- */
  function applyTheme(choice) {
    var root = document.documentElement;
    if (choice === "light" || choice === "dark") { root.setAttribute("data-theme", choice); }
    else { root.removeAttribute("data-theme"); }
  }

  function initTheme() {
    var saved = "auto";
    try { saved = window.localStorage.getItem("qrh-theme") || "auto"; } catch (e) { saved = "auto"; }
    applyTheme(saved);
    var select = document.getElementById("theme-select");
    if (!select) { return; }
    select.value = (saved === "light" || saved === "dark") ? saved : "auto";
    select.addEventListener("change", function () {
      applyTheme(select.value);
      try { window.localStorage.setItem("qrh-theme", select.value); } catch (e) { saved = "auto"; }
    });
  }

  function lightFor(verdict) {
    if (verdict === undefined || verdict === null) { return MISSING; }
    return LIGHTS[verdict] || { css: "amber", label: "Unrecognised result: " + verdict };
  }

  // Every read made through fetchJson is remembered here, so Refresh can tell when the page has finished loading.
  var pendingReads = [];

  function fetchJson(path) {
    var read = fetch(path, { method: "GET", headers: { "Accept": "application/json" } }).then(function (resp) {
      return resp.text().then(function (text) {
        var body = null;
        try { body = JSON.parse(text); } catch (e) { body = null; }
        if (!resp.ok) {
          var why = body && body.error ? body.error : "the server answered with status " + resp.status;
          throw new Error(why);
        }
        if (body === null) { throw new Error("the server sent an answer that is not readable"); }
        return body;
      });
    });
    pendingReads.push(read.then(function () {}, function () {}));
    return read;
  }

  function showError(prefix, err) {
    var box = el("div", "error");
    box.setAttribute("role", "alert");
    box.appendChild(el("strong", null, prefix));
    box.appendChild(el("p", null, err && err.message ? err.message : String(err)));
    return box;
  }

  function renderLights(verdicts) {
    var list = el("ul", "lights");
    list.setAttribute("aria-label", "Stage results");
    STAGES.forEach(function (stage) {
      var info = lightFor(verdicts ? verdicts[stage] : undefined);
      var item = el("li", "light " + info.css);
      item.appendChild(el("span", "dot"));
      item.appendChild(el("span", null, stage + ": " + info.label));
      list.appendChild(item);
    });
    return list;
  }

  function describe(t) {
    var parts = [t.strategy_family, t.asset_class, t.frequency].filter(function (x) { return x; });
    return parts.join(" / ");
  }

  /* ---- list view ---- */
  function showList() {
    clear(app);
    app.appendChild(el("h2", null, "Trials"));
    app.appendChild(el("p", "muted",
      "Each trial moves through stages S0 to S7. Green means passed, red means failed, " +
      "amber means blocked or not conclusive, grey means the stage has not run yet."));
    var holder = el("div");
    holder.appendChild(el("p", "muted", "Loading trials..."));
    app.appendChild(holder);
    fetchJson("/api/trials").then(function (body) {
      clear(holder);
      var trials = body.trials || [];
      if (!trials.length) {
        holder.appendChild(emptyState("No trials yet.",
          "Ask the assistant in VS Code to start a research run; results appear here."));
        return;
      }
      var stats = el("ul", "trial-list stat-list");
      stats.appendChild(statCard(trials.length, trials.length === 1 ? "Trial recorded" : "Trials recorded", "pass", "Recorded"));
      holder.appendChild(stats);
      var list = el("ul", "trial-list");
      trials.forEach(function (t) {
        var card = el("li", "trial-card");
        var link = el("a", "trial-link", t.trial_id);
        link.setAttribute("href", "#/trial/" + encodeURIComponent(t.trial_id));
        card.appendChild(link);
        card.appendChild(el("p", "muted", describe(t)));
        card.appendChild(renderLights(t.stage_verdicts));
        list.appendChild(card);
      });
      holder.appendChild(list);
    }).catch(function (err) {
      clear(holder);
      holder.appendChild(showError("The trial list could not be loaded.", err));
    });
  }

  /* ---- detail view ---- */
  function verdictSentence(w) {
    if (!w) { return null; }
    var text = "Overall result: " + String(w.verdict);
    if (w.failing_stage) { text += ". It stopped at stage " + String(w.failing_stage) + "."; }
    return text;
  }

  function renderStages(detail) {
    var list = el("ul", "stage-list");
    STAGES.forEach(function (stage) {
      var s = detail.stages ? detail.stages[stage] : undefined;
      var item = el("li", "stage-item");
      if (!s) {
        item.appendChild(el("h3", null, stage + ": " + MISSING.label));
        list.appendChild(item);
        return;
      }
      if (s.error) {
        item.appendChild(el("h3", null, stage + ": result could not be verified"));
        item.appendChild(el("p", "error", s.error));
        list.appendChild(item);
        return;
      }
      var info = lightFor(s.verdict);
      var head = el("h3", "light " + info.css);
      head.appendChild(el("span", "dot"));
      head.appendChild(el("span", null, stage + ": " + info.label));
      item.appendChild(head);
      if (s.reasons && s.reasons.length) {
        var ul = el("ul");
        s.reasons.forEach(function (r) { ul.appendChild(el("li", null, r)); });
        item.appendChild(ul);
      } else {
        item.appendChild(el("p", "muted", "No reasons were recorded."));
      }
      list.appendChild(item);
    });
    return list;
  }

  function renderWarnings(w) {
    var wrap = el("div");
    var items = (w && w.warnings) || [];
    if (!items.length) {
      wrap.appendChild(el("p", null, "No warnings for this trial."));
      return wrap;
    }
    var list = el("ul", "warning-list");
    items.forEach(function (x) {
      var item = el("li", "warning-item");
      var head = el("p");
      head.appendChild(el("span", "severity", sentence(x.severity)));
      head.appendChild(document.createTextNode(" (stage " + String(x.stage) + ")"));
      item.appendChild(head);
      item.appendChild(el("p", null, x.message));
      var res = el("p", "resolution");
      res.appendChild(el("strong", null, "What to do: "));
      res.appendChild(document.createTextNode(String(x.resolution)));
      item.appendChild(res);
      list.appendChild(item);
    });
    wrap.appendChild(list);
    return wrap;
  }

  function addRow(dl, name, value) {
    if (value === undefined || value === null || value === "") { return; }
    dl.appendChild(el("dt", null, name));
    dl.appendChild(el("dd", null, value));
  }

  function renderDetails(detail, w) {
    var box = el("details");
    box.appendChild(el("summary", null, "Details (ids and hashes)"));
    var dl = el("dl");
    addRow(dl, "Trial id", detail.trial_id);
    addRow(dl, "Family", detail.family);
    addRow(dl, "Created at", detail.created_at);
    ((w && w.warnings) || []).forEach(function (x, i) {
      var n = "Warning " + (i + 1) + " ";
      addRow(dl, n + "id", x.warning_id);
      addRow(dl, n + "code", x.code);
      addRow(dl, n + "scope", x.scope);
      addRow(dl, n + "receipt hash", x.receipt_hash);
      addRow(dl, n + "receipt path", x.receipt_path);
    });
    box.appendChild(dl);
    return box;
  }

  var COST_SOURCES = { broker_record: "Broker record", declared: "Declared by you" };

  function renderRunSettings(detail) {
    var wrap = el("div", "check-card");
    var cs = detail.cost_source;
    if (cs) {
      var text = COST_SOURCES[cs.source] || (cs.source ? "Source: " + String(cs.source) : "Source not recorded");
      if (cs.broker) { text += " (broker: " + String(cs.broker) + ")"; }
      if (cs.status === "BLOCKED") { text = "Blocked. " + text; }
      wrap.appendChild(el("p", null, "Cost source: " + text));
      if (cs.reason) { wrap.appendChild(el("p", "muted", cs.reason)); }
    } else {
      wrap.appendChild(el("p", "muted", "Cost source: none recorded for this run."));
    }
    var adj = detail.adjustment;
    var cols = adj ? Object.keys(adj).sort() : [];
    if (cols.length) {
      cols.forEach(function (c) {
        wrap.appendChild(el("p", null, "Price adjustment for " + c + ": " + String(adj[c])));
      });
    } else {
      wrap.appendChild(el("p", "muted", "Price adjustment: none recorded for this run."));
    }
    return wrap;
  }

  function showTrial(id) {
    clear(app);
    var back = el("a", null, "Back to all trials");
    back.setAttribute("href", "#/");
    app.appendChild(back);
    app.appendChild(el("h2", null, "Trial " + id));
    var holder = el("div");
    holder.appendChild(el("p", "muted", "Loading trial..."));
    app.appendChild(holder);

    var path = "/api/trials/" + encodeURIComponent(id);
    Promise.all([
      fetchJson(path),
      fetchJson(path + "/warnings").then(function (w) { return { ok: w }; },
                                         function (e) { return { err: e }; })
    ]).then(function (res) {
      var detail = res[0];
      var wr = res[1];
      var w = wr.ok || null;
      clear(holder);

      var info = lightFor(null);
      if (w && LIGHTS[w.verdict]) { info = LIGHTS[w.verdict]; }
      else if (w) { info = { css: "amber", label: "" }; }
      var vb = el("div", "verdict-box " + info.css);
      vb.appendChild(el("h3", null, "Verdict"));
      if (w) {
        vb.appendChild(el("p", null, verdictSentence(w)));
        if (info.label) { vb.appendChild(el("p", "muted", "In plain words: " + info.label.toLowerCase() + ".")); }
      } else {
        vb.appendChild(showError("The verdict could not be loaded.", wr.err));
      }
      holder.appendChild(vb);

      var meta = describe(detail);
      if (meta) { holder.appendChild(el("p", "muted", meta)); }
      if (detail.hypothesis) {
        holder.appendChild(el("h2", null, "Hypothesis"));
        holder.appendChild(el("p", null, detail.hypothesis));
      }

      holder.appendChild(el("h2", null, "Cost and adjustment"));
      holder.appendChild(renderRunSettings(detail));

      holder.appendChild(el("h2", null, "Stage results"));
      holder.appendChild(renderStages(detail));

      holder.appendChild(el("h2", null, "Warnings"));
      if (w) { holder.appendChild(renderWarnings(w)); }
      else { holder.appendChild(showError("The warnings could not be loaded.", wr.err)); }

      holder.appendChild(renderDetails(detail, w));
    }).catch(function (err) {
      clear(holder);
      holder.appendChild(showError("This trial could not be loaded.", err));
    });
  }

  /* ---- bench view ---- */
  function renderMetrics(metrics) {
    var names = Object.keys(metrics || {});
    if (!names.length) { return el("p", "muted", "No metrics were recorded."); }
    var table = el("table", "metrics-table");
    var head = el("tr");
    head.appendChild(el("th", null, "Metric"));
    head.appendChild(el("th", null, "Value"));
    var thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    var tbody = el("tbody");
    names.sort().forEach(function (name) {
      var row = el("tr");
      row.appendChild(el("td", null, name));
      row.appendChild(el("td", "num", metrics[name]));
      tbody.appendChild(row);
    });
    table.appendChild(tbody);
    return table;
  }

  var BENCH_CAP = 20;

  function showBench() {
    clear(app);
    app.appendChild(el("h2", null, "Bench results"));
    app.appendChild(el("p", "muted", "Each entry is one recorded benchmark run, with its result and its measured values."));
    var holder = el("div");
    holder.appendChild(el("p", "muted", "Loading bench results..."));
    app.appendChild(holder);
    fetchJson("/api/bench").then(function (body) {
      clear(holder);
      var benches = body.benches || [];
      if (!benches.length) {
        holder.appendChild(emptyState("No bench results yet.",
          "Ask the assistant in VS Code to run the benchmarks; results appear here."));
        return;
      }
      // Newest first, shown as a timestamped activity list with a visible cap.
      var sorted = benches.slice().sort(function (a, b) {
        return String(b.created_at) < String(a.created_at) ? -1 : String(b.created_at) > String(a.created_at) ? 1 : 0;
      });
      var shown = sorted.slice(0, BENCH_CAP);
      if (sorted.length > BENCH_CAP) {
        holder.appendChild(el("p", "muted", "Showing the latest " + BENCH_CAP + " of " + sorted.length + " bench results."));
      }
      var list = el("ul", "trial-list activity-list");
      shown.forEach(function (b) {
        var info = lightFor(b.verdict);
        var card = el("li", "bench-card");
        card.appendChild(el("h3", null, b.suite));
        card.appendChild(el("p", "muted", "Recorded at " + String(b.created_at) + " (bench id " + String(b.bench_id) + ")"));
        var pill = el("p", "light " + info.css);
        pill.appendChild(el("span", "dot"));
        pill.appendChild(el("span", null, "Result: " + info.label));
        card.appendChild(pill);
        card.appendChild(renderMetrics(b.metrics));
        list.appendChild(card);
      });
      holder.appendChild(list);
      holder.appendChild(el("p", "muted end-note", "End of list."));
    }).catch(function (err) {
      clear(holder);
      holder.appendChild(showError("The bench results could not be loaded.", err));
    });
  }

  /* ---- warnings view ---- */
  var SEVERITY_ORDER = ["high", "medium", "low"];

  function renderGroupedWarnings(w) {
    var items = (w && w.warnings) || [];
    var wrap = el("div");
    if (!items.length) {
      wrap.appendChild(el("p", null, "No warnings for this trial."));
      return wrap;
    }
    var groups = {};
    var order = SEVERITY_ORDER.slice();
    items.forEach(function (x) {
      var key = String(x.severity).toLowerCase();
      if (!groups[key]) { groups[key] = []; }
      groups[key].push(x);
      if (order.indexOf(key) === -1) { order.push(key); }
    });
    order.forEach(function (key) {
      if (!groups[key]) { return; }
      var css = SEVERITY_ORDER.indexOf(key) === -1 ? "sev-other" : "sev-" + key;
      var section = el("section", "sev-group " + css);
      section.appendChild(el("h3", null, sentence(key) + " severity (" + groups[key].length + ")"));
      var list = el("ul", "warning-list");
      groups[key].forEach(function (x) {
        var item = el("li", "warning-item");
        item.appendChild(el("p", null, x.message));
        var res = el("p", "resolution");
        res.appendChild(el("strong", null, "What to do: "));
        res.appendChild(document.createTextNode(String(x.resolution)));
        item.appendChild(res);
        list.appendChild(item);
      });
      section.appendChild(list);
      wrap.appendChild(section);
    });
    return wrap;
  }

  function showWarnings() {
    clear(app);
    app.appendChild(el("h2", null, "Warnings"));
    app.appendChild(el("p", "muted", "Pick a trial to see its warnings and what to do about them."));
    var holder = el("div");
    holder.appendChild(el("p", "muted", "Loading trials..."));
    app.appendChild(holder);
    fetchJson("/api/trials").then(function (body) {
      clear(holder);
      var trials = body.trials || [];
      if (!trials.length) {
        holder.appendChild(emptyState("No trials yet.",
          "Ask the assistant in VS Code to start a research run; results appear here."));
        return;
      }
      var label = el("label", null, "Trial: ");
      label.setAttribute("for", "warn-trial");
      var select = el("select");
      select.id = "warn-trial";
      var first = el("option", null, "Choose a trial...");
      first.value = "";
      select.appendChild(first);
      trials.forEach(function (t) {
        var o = el("option", null, t.trial_id);
        o.value = t.trial_id;
        select.appendChild(o);
      });
      var out = el("div");
      out.setAttribute("aria-live", "polite");
      select.addEventListener("change", function () {
        clear(out);
        if (!select.value) { return; }
        out.appendChild(el("p", "muted", "Loading warnings..."));
        var chosen = select.value;
        fetchJson("/api/trials/" + encodeURIComponent(chosen) + "/warnings").then(function (w) {
          if (select.value !== chosen) { return; }
          clear(out);
          out.appendChild(renderGroupedWarnings(w));
        }).catch(function (err) {
          if (select.value !== chosen) { return; }
          clear(out);
          out.appendChild(showError("The warnings could not be loaded.", err));
        });
      });
      holder.appendChild(label);
      holder.appendChild(select);
      holder.appendChild(out);
    }).catch(function (err) {
      clear(holder);
      holder.appendChild(showError("The trial list could not be loaded.", err));
    });
  }

  /* ---- checks view: parity reports and data checks, shown as the API returns them ---- */
  var PARITY_LABELS = {
    MATCH: "Match: the backtest and live trading agree",
    DIFFERENT: "Different: the backtest and live trading do not agree",
    MISSING_LIVE_DATA: "Missing live data: no comparison was made"
  };
  var CHECK_LABELS = {
    PASS: "Pass: no problem found",
    WARN: "Warning: look at this data",
    FAIL: "Fail: a data rule found a problem"
  };

  function cardPill(css, text) {
    var pill = el("p", "light " + css);
    pill.appendChild(el("span", "dot"));
    pill.appendChild(el("span", null, text));
    return pill;
  }

  function listOf(texts) {
    var ul = el("ul", "check-notes");
    texts.forEach(function (t) { ul.appendChild(el("li", null, t)); });
    return ul;
  }

  function parityCard(item) {
    var card = el("li", "check-card");
    if (item.error) {
      card.appendChild(el("h3", null, String(item.file)));
      card.appendChild(el("p", "error", item.error));
      return card;
    }
    card.appendChild(el("h3", null, "Trial " + String(item.trial_id)));
    if (item.strategy_name) { card.appendChild(el("p", "muted", String(item.strategy_name))); }
    var css = item.status === "MATCH" ? "pass" : item.status === "DIFFERENT" ? "fail" : "amber";
    card.appendChild(cardPill(css, "Parity: " + (PARITY_LABELS[item.status] || "Unrecognised status: " + String(item.status))));
    if (item.start_date || item.end_date) {
      card.appendChild(el("p", "muted", "Days compared: " + String(item.days_compared) +
        " (from " + String(item.start_date) + " to " + String(item.end_date) + ")"));
    }
    if (item.end_date) { card.appendChild(el("p", "freshness muted", "Data through " + String(item.end_date))); }
    if (item.explanation) { card.appendChild(el("p", null, item.explanation)); }
    var diffs = item.differing_days || [];
    if (diffs.length) {
      card.appendChild(el("p", null, "Days that differ (" + diffs.length + "):"));
      card.appendChild(listOf(diffs.map(function (d) {
        return String(d.date) + " " + String(d.instrument);
      })));
    }
    var warns = item.warnings || [];
    if (warns.length) {
      card.appendChild(el("p", null, "Warnings (" + warns.length + "):"));
      card.appendChild(listOf(warns.map(String)));
    } else {
      card.appendChild(el("p", "muted", "No warnings."));
    }
    return card;
  }

  function dataCheckCard(item) {
    var card = el("li", "check-card");
    if (item.error) {
      card.appendChild(el("h3", null, String(item.file)));
      card.appendChild(el("p", "error", item.error));
      return card;
    }
    card.appendChild(el("h3", null, "Dataset " + String(item.dataset)));
    var css = item.verdict === "PASS" ? "pass" : item.verdict === "FAIL" ? "fail" : "amber";
    card.appendChild(cardPill(css, "Data check: " + (CHECK_LABELS[item.verdict] || "Unrecognised result: " + String(item.verdict))));
    var age = item.age_days === null || item.age_days === undefined ? "unknown" : String(item.age_days) + " days";
    if (item.last_date) { card.appendChild(el("p", "freshness muted", "Data through " + String(item.last_date))); }
    card.appendChild(el("p", "muted", "Last date in the data: " + String(item.last_date || "unknown") + ". Age: " + age +
      ". Checked on " + String(item.checked_on) + "."));
    var found = item.findings || [];
    if (found.length) { card.appendChild(listOf(found.map(String))); }
    else { card.appendChild(el("p", "muted", "Nothing to do.")); }
    return card;
  }

  function loadCards(path, emptyTitle, emptyHow, failText, makeCard, into) {
    var holder = el("div");
    holder.appendChild(el("p", "muted", "Loading..."));
    into.appendChild(holder);
    fetchJson(path).then(function (body) {
      clear(holder);
      var items = body.items || [];
      if (!items.length) { holder.appendChild(emptyState(emptyTitle, emptyHow)); return; }
      var list = el("ul", "trial-list");
      items.forEach(function (it) { list.appendChild(makeCard(it)); });
      holder.appendChild(list);
    }).catch(function (err) {
      clear(holder);
      holder.appendChild(showError(failText, err));
    });
  }

  function showChecks() {
    clear(app);
    app.appendChild(el("h2", null, "Checks"));
    app.appendChild(el("p", "muted",
      "What the harness has recorded about parity and data. This page only shows it; every result carries a text label."));
    app.appendChild(el("h2", null, "Parity reports"));
    app.appendChild(el("p", "muted", "Parity compares the backtest with live trading, day by day."));
    loadCards("/api/parity", "No parity report yet.", "Ask the assistant in VS Code to build one for a trial; it appears here.",
      "The parity reports could not be loaded.", parityCard, app);
    app.appendChild(el("h2", null, "Data checks"));
    app.appendChild(el("p", "muted", "Each dataset is checked against its data rules and its age."));
    loadCards("/api/data-checks", "No data checks yet.", "Ask the assistant in VS Code to check a dataset; the result appears here.",
      "The data checks could not be loaded.", dataCheckCard, app);
    app.appendChild(el("h2", null, "Cost source, adjustment and bench results"));
    app.appendChild(el("p", "muted",
      "The cost source and the adjustment type of each run are on that trial's page (Trials tab). " +
      "Bench results are on the Bench tab."));
  }

  /* ---- start here: fixed plain-English text, no API calls ---- */
  var START_IS = [
    "The harness tests one trading strategy, or one portfolio of assets, through the whole workflow: idea, backtest, paper trading, live trading.",
    "It gives you recommendations and reports.",
    "The decision to use a strategy is always yours."
  ];
  var START_IS_NOT = [
    "It never trades for you.",
    "It never stops your work.",
    "It never decides for you."
  ];
  var CHECKLIST = [
    { check: "Describe your idea", detail: "Tell the assistant what you want to test and where you trade from." },
    { check: "Provide data or let the assistant get it", detail: "Provide data, or let the assistant obtain it. The Checks tab shows data checks." },
    { check: "Backtest and check", detail: "The assistant runs the backtest and the checks. Stage results show on the Trials tab." },
    { check: "Read the verdict", detail: "Open a trial to read its verdict and warnings. The decision is yours." }
  ];
  var START_STEPS = [
    { step: "Describe your trading idea and where you trade from.",
      detail: "These are the intake questions. The AI assistant asks them in the chat.",
      tab: "Result: your answers are recorded and shown on the Trials tab." },
    { step: "Provide your data, or let the assistant obtain it.",
      detail: "The assistant tells you what data it needs and where it came from.",
      tab: "Result: the data checks are on the Checks tab." },
    { step: "The assistant builds or checks the backtest and explains every number.",
      detail: "Ask it to explain any number you do not understand.",
      tab: "Result: your research results appear on the Trials tab, where each trial shows its internal checks S0 to S7." },
    { step: "Run the research.",
      detail: "Ask the assistant to run it (see the next section).",
      tab: "Result: new stage results appear on the Trials tab." },
    { step: "Read the verdict, the warnings and the report.",
      detail: "Open a trial to see its verdict. Each warning says what to check and why.",
      tab: "Result: see the Trials, Warnings and Checks tabs." },
    { step: "Decide what to do: paper trade or stop.",
      detail: "This choice is yours. The harness only informs it.",
      tab: "Result: nothing to look at here. You decide, then tell the assistant." }
  ];
  var START_TABS = [
    { name: "Trials", text: "Each strategy's results. Under each trial are its internal checks S0 to S7 (the harness's automatic stages). Green means passed, red means failed, amber means blocked or not conclusive, grey means not run yet." },
    { name: "Bench", text: "Tests of the harness itself, to show that the tool works." },
    { name: "Warnings", text: "What to check and why, for each trial." },
    { name: "Checks", text: "Parity checks (backtest against live trading) and data checks." }
  ];
  var START_GLOSSARY = [
    { term: "Stage", means: "One automatic check inside the harness's testing pipeline. There are eight, S0 to S7. They are internal checks, separate from the six steps you follow." },
    { term: "Verdict", means: "The overall result for a trial, taken from its stages." },
    { term: "PASS", means: "The stage met its rules." },
    { term: "FAIL", means: "The stage broke a rule, so the strategy did not get through it." },
    { term: "Warning", means: "Something worth checking. It does not stop anything." },
    { term: "Backtest", means: "Running the strategy on past data to see how it would have done." },
    { term: "Paper trading", means: "Trading with pretend money in real time, to test the strategy safely." },
    { term: "Live trading", means: "Trading with real money. It is always your decision." }
  ];

  function startSection(title) {
    var s = el("section", "start-section");
    s.appendChild(el("h2", null, title));
    return s;
  }

  function startList(texts, ordered) {
    var ul = el(ordered ? "ol" : "ul", "start-list");
    texts.forEach(function (t) { ul.appendChild(el("li", null, t)); });
    return ul;
  }

  function showStart() {
    clear(app);
    app.appendChild(el("h2", "start-title", "Start here"));
    app.appendChild(el("p", "muted", "A short guide for a beginner. This page only explains; it never changes anything."));

    var cl = startSection("Your checklist");
    var clList = el("ol", "checklist");
    var statusWords = [];
    CHECKLIST.forEach(function (c) {
      var li = el("li", "checklist-item");
      li.appendChild(el("strong", null, c.check));
      li.appendChild(el("p", "muted", c.detail));
      var word = el("p", "checklist-status", "To do");
      statusWords.push(word);
      li.appendChild(word);
      clList.appendChild(li);
    });
    cl.appendChild(clList);
    cl.appendChild(el("p", "muted", "A step shows Done when the harness has recorded something for it."));
    app.appendChild(cl);
    // Read-only: reuse the existing list endpoints to mark steps Done. A failed call leaves "To do".
    var optional = function (path) {
      return fetchJson(path).then(function (b) { return b; }, function () { return {}; });
    };
    Promise.all([optional("/api/trials"), optional("/api/data-checks")]).then(function (res) {
      var trials = res[0].trials || [];
      var checks = res[1].items || [];
      var anyStage = false;
      var anyLast = false;
      trials.forEach(function (t) {
        var v = t.stage_verdicts || {};
        STAGES.forEach(function (s) { if (v[s] !== undefined && v[s] !== null) { anyStage = true; } });
        if (v.S7 !== undefined && v.S7 !== null) { anyLast = true; }
      });
      var done = [trials.length > 0, checks.length > 0, anyStage, anyLast];
      done.forEach(function (d, i) {
        if (!d) { return; }
        statusWords[i].textContent = "Done";
        statusWords[i].className = "checklist-status done";
      });
    });

    var a = startSection("What the harness is");
    a.appendChild(startList(START_IS, false));
    app.appendChild(a);

    var b = startSection("What it does not do");
    b.appendChild(startList(START_IS_NOT, false));
    app.appendChild(b);

    var c = startSection("The steps to follow");
    var steps = el("ol", "start-list");
    START_STEPS.forEach(function (s) {
      var li = el("li");
      li.appendChild(el("strong", null, s.step));
      li.appendChild(el("p", "muted", s.detail));
      li.appendChild(el("p", null, s.tab));
      steps.appendChild(li);
    });
    c.appendChild(steps);
    app.appendChild(c);

    var d = startSection("How a research run is started today");
    d.appendChild(el("p", null,
      "Ask the AI assistant in VS Code to run the research for your trial. The dashboard shows the results as they are recorded."));
    d.appendChild(el("p", null, "There is no start button yet. This dashboard cannot start a run."));
    app.appendChild(d);

    var e = startSection("What each tab is for");
    var tabs = el("dl", "start-dl");
    START_TABS.forEach(function (t) {
      tabs.appendChild(el("dt", null, t.name));
      tabs.appendChild(el("dd", null, t.text));
    });
    e.appendChild(tabs);
    app.appendChild(e);

    var f = startSection("Glossary");
    var gl = el("dl", "start-dl");
    START_GLOSSARY.forEach(function (g) {
      gl.appendChild(el("dt", null, g.term));
      gl.appendChild(el("dd", null, g.means));
    });
    f.appendChild(gl);
    app.appendChild(f);

    var g = startSection("A note on safety");
    g.appendChild(el("p", null, "Results are information. Nothing on this page changes a strategy."));
    app.appendChild(g);
  }

  /* ---- routing ---- */
  function markTab(name) {
    ["start", "trials", "bench", "warnings", "checks"].forEach(function (t) {
      var a = document.getElementById("tab-" + t);
      if (!a) { return; }
      if (t === name) { a.setAttribute("aria-current", "page"); }
      else { a.removeAttribute("aria-current"); }
    });
  }

  function route() {
    if (!window.location.hash) { markTab("start"); showStart(); return; }
    var hash = window.location.hash;
    if (hash === "#/start") { markTab("start"); showStart(); return; }
    if (hash === "#/bench") { markTab("bench"); showBench(); return; }
    if (hash === "#/checks") { markTab("checks"); showChecks(); return; }
    if (hash === "#/warnings") { markTab("warnings"); showWarnings(); return; }
    markTab("trials");
    var m = /^#\/trial\/(.+)$/.exec(hash);
    if (m) {
      var id;
      try { id = decodeURIComponent(m[1]); } catch (e) { id = m[1]; }
      showTrial(id);
    } else {
      showList();
    }
  }

  /* ---- refresh: draw the current page again. Read-only: it only repeats the GET reads of that page. ---- */
  function two(n) { return (n < 10 ? "0" : "") + n; }

  function waitForReads(seen) {
    var batch = pendingReads.slice(seen);
    return Promise.all(batch).then(function () {
      if (pendingReads.length > seen + batch.length) { return waitForReads(seen + batch.length); }
    });
  }

  function initRefresh() {
    var button = document.getElementById("refresh");
    var stamp = document.getElementById("updated-at");
    if (!button) { return; }
    button.addEventListener("click", function () {
      if (button.disabled) { return; }
      button.disabled = true;
      button.textContent = "Refreshing...";
      var seen = pendingReads.length;
      var finish = function () {
        var now = new Date();
        if (stamp) {
          stamp.textContent = "Updated at " + two(now.getHours()) + ":" + two(now.getMinutes()) + ":" + two(now.getSeconds());
        }
        button.textContent = "Refresh";
        button.disabled = false;
      };
      try { route(); } catch (e) { finish(); return; }
      waitForReads(seen).then(finish, finish);
    });
  }

  initTheme();
  initRefresh();
  window.addEventListener("hashchange", route);
  route();
})();
