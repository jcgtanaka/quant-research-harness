"""Rule 2 in the registry: variants are logged with their prompt hash, and a changed prompt leaves a notice."""

import pytest

from qrh.memory.registry import HypothesisRecord, Registry

BASE = dict(text="t", strategy_family="trend", asset_class="synthetic", frequency="1D", parameter_space_size=4,
            thresholds={"a": 1}, difference_statement="d")


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "r.sqlite3")
    yield r
    r.close()


def new(reg, **kw):
    return reg.register_hypothesis(HypothesisRecord(**{**BASE, **kw}))


def variant(name, phash):
    return {"name": name, "models": ["m1"], "prompt_files": ["p.txt"], "params": {}, "prompt_hash": phash}


def test_each_evaluated_parameters_and_variant_is_logged_with_the_variant_name_and_prompt_hash(reg):
    t = new(reg)
    reg.record_trial(t, {"k": 1, "llm_variant": variant("terse", "aa")}, {"m": 1.0})
    reg.record_trial(t, {"k": 1, "llm_variant": variant("verbose", "bb")}, {"m": 1.0})
    reg.record_trial(t, {"k": 2}, {"m": 1.0})  # a strategy with no llm: both columns stay empty
    rows = reg.conn.execute("SELECT variant, prompt_hash FROM trials WHERE trial_id=? ORDER BY id", (t,)).fetchall()
    assert [tuple(r) for r in rows] == [("terse", "aa"), ("verbose", "bb"), (None, None)]


def test_variants_count_as_distinct_combinations_for_pbo_and_dsr(reg):
    t = new(reg)
    for name, h in (("a", "1"), ("b", "2")):
        for k in (1, 2):
            reg.record_trial(t, {"k": k, "llm_variant": variant(name, h)}, {"m": 1.0})
    assert reg.trial_count(t) == 4  # 2 parameter points x 2 variants
    assert reg.family_trial_count(reg.get_trial(t)["family_id"]) == 4


def test_the_same_prompt_in_two_trials_is_still_two_trials(reg):
    a, b = new(reg), new(reg)
    for t in (a, b):
        reg.record_trial(t, {"k": 1, "llm_variant": variant("a", "1")}, {"m": 1.0})
    assert reg.family_trial_count(reg.get_trial(a)["family_id"]) == 2


def test_a_fingerprint_is_stored_once_per_trial_with_its_notes(reg):
    t = new(reg)
    reg.record_llm_fingerprint(t, reg.get_trial(t)["family_id"], "ph", "fp", ["a note"])
    got = reg.get_llm_fingerprint(t)
    assert got == {"trial_id": t, "family_id": "trend|synthetic|1D", "prompt_hash": "ph", "fingerprint": "fp",
                   "notes": ["a note"]}
    assert reg.get_llm_fingerprint("nope") is None


def test_trials_of_the_family_with_a_different_fingerprint_are_found_newest_first(reg):
    ids = [new(reg) for _ in range(4)]
    fam = reg.get_trial(ids[0])["family_id"]
    other = new(reg, strategy_family="other")
    for t, fp in zip(ids, ("x", "y", "y", "x")):
        reg.record_llm_fingerprint(t, fam, "p" + fp, fp, [])
    reg.record_llm_fingerprint(other, "other|synthetic|1D", "pz", "z", [])
    plain = new(reg)  # a trial with no llm has no fingerprint and never matches
    assert plain not in reg.llm_trials_differing(fam, "newcomer", "x")
    assert reg.llm_trials_differing(fam, "newcomer", "x") == [ids[2], ids[1]]
    assert reg.llm_trials_differing(fam, ids[3], "x") == [ids[2], ids[1]]  # the trial itself is never listed
    assert reg.llm_trials_differing(fam, "newcomer", "none-match") == [ids[3], ids[2], ids[1], ids[0]]


def test_the_declared_family_count_is_what_the_count_becomes_once_every_registered_trial_ran(reg):
    a, b = new(reg), new(reg)  # two registered trials of the family, nothing evaluated yet
    fam = reg.get_trial(a)["family_id"]
    assert reg.family_trial_count(fam) == 0 and reg.family_declared_count(fam) == 8
    for t in (a, b):
        for k in range(4):
            reg.record_trial(t, {"k": k}, {"m": 1.0})
    assert reg.family_trial_count(fam) == reg.family_declared_count(fam) == 8
