"""Answer provenance: a value counts only if the user said it, in the session, after the question was issued."""

import json

import pytest

from qrh.interview.provenance import normalize, user_texts, verify_in_transcript

ISSUED = "2026-01-01T10:00:00+00:00"
LATER = "2026-01-01T10:05:00Z"
EARLIER = "2026-01-01T09:00:00Z"


def write(path, *records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def user(text, ts=LATER, session="s1", **extra):
    return {"type": "user", "timestamp": ts, "sessionId": session, "message": {"role": "user", "content": text}, **extra}


def assistant(text, ts=LATER, session="s1"):
    return {"type": "assistant", "timestamp": ts, "sessionId": session,
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}}


def tool_result(text, tool_use_id="t1", ts=LATER, session="s1"):
    return {"type": "user", "timestamp": ts, "sessionId": session,
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": text}]}}


def tool_use(name, tool_use_id="t1", ts=LATER, session="s1"):
    return {"type": "assistant", "timestamp": ts, "sessionId": session,
            "message": {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": name, "input": {}}]}}


def test_normalize_is_case_and_whitespace_insensitive():
    assert normalize("  Point-in-Time \n daily ") == "point-in-time daily"


def test_user_string_and_text_blocks_count(tmp_path):
    t = write(tmp_path / "t.jsonl", user("it was Point-in-time   daily"),
              {"type": "user", "timestamp": LATER, "sessionId": "s1",
               "message": {"role": "user", "content": [{"type": "text", "text": "second answer"}]}})
    texts = [x.text for x in user_texts(t, "s1")]
    assert texts == ["it was Point-in-time   daily", "second answer"]


def test_assistant_and_ordinary_tool_results_do_not_count(tmp_path):
    t = write(tmp_path / "t.jsonl", assistant("the answer is vendor"), tool_use("Bash"), tool_result("vendor"))
    assert user_texts(t, "s1") == []


def test_ask_user_question_answer_counts_but_not_the_question_text(tmp_path):
    result = 'User has answered your questions: "Pick one (vendor, constant)?"="Single constant". You can now continue.'
    rec = tool_result(result)
    rec["toolUseResult"] = {"answers": {"Pick one (vendor, constant)?": "Single constant"}}
    t = write(tmp_path / "t.jsonl", tool_use("AskUserQuestion"), rec)
    texts = [x.text for x in user_texts(t, "s1")]
    assert texts == ["Single constant"]


def test_sidechain_meta_and_harness_text_are_ignored(tmp_path):
    t = write(tmp_path / "t.jsonl", user("from subagent", isSidechain=True), user("meta", isMeta=True),
              user("<system-reminder>vendor</system-reminder>"), user("Caveat: vendor"))
    assert user_texts(t, "s1") == []


def test_other_sessions_are_ignored(tmp_path):
    t = write(tmp_path / "t.jsonl", user("vendor", session="other"))
    assert user_texts(t, "s1") == []


def test_verify_accepts_value_or_label_after_issue(tmp_path):
    t = write(tmp_path / "t.jsonl", user("I'd say POINT-IN-TIME daily published, yes"))
    assert verify_in_transcript(t, "s1", ISSUED, ["pit", "Point-in-time daily published"])
    assert not verify_in_transcript(t, "s1", ISSUED, ["vendor"])


def test_verify_rejects_messages_posted_before_the_question(tmp_path):
    t = write(tmp_path / "t.jsonl", user("vendor", ts=EARLIER))
    assert not verify_in_transcript(t, "s1", ISSUED, ["vendor"])


def test_unknown_accepts_dont_know_phrases(tmp_path):
    for phrase in ("I don't know", "dont know", "no idea / not sure", "unknown"):
        t = write(tmp_path / "t.jsonl", user(phrase))
        assert verify_in_transcript(t, "s1", ISSUED, ["unknown"]), phrase


def test_unreadable_transcript_raises(tmp_path):
    with pytest.raises(OSError):
        user_texts(tmp_path / "missing.jsonl", "s1")


# --- F4: whole words, minimum length, short messages for bool/choice -------------------------------

from qrh.interview.provenance import verify_answer  # noqa: E402


def said(tmp_path, *texts):
    return write(tmp_path / "t.jsonl", *[user(t) for t in texts])


def test_a_one_letter_answer_no_longer_matches_any_message(tmp_path):
    t = said(tmp_path, "Why do you ask? Let's analyze the history of this strategy.")
    assert not verify_answer(t, "s1", ISSUED, kind="bool", candidates=["y", "yes"])


def test_bool_and_choice_need_the_label_or_id_as_a_whole_word(tmp_path):
    assert verify_answer(said(tmp_path, "yes"), "s1", ISSUED, kind="bool", candidates=["y", "yes"])
    assert verify_answer(said(tmp_path, "Y"), "s1", ISSUED, kind="bool", candidates=["y", "yes"])
    assert verify_answer(said(tmp_path, "pit, please"), "s1", ISSUED, kind="choice", candidates=["pit", "Point-in-time"])
    assert not verify_answer(said(tmp_path, "spit it out"), "s1", ISSUED, kind="choice", candidates=["pit"])
    assert not verify_answer(said(tmp_path, "the eyes have it"), "s1", ISSUED, kind="bool", candidates=["yes"])


def test_a_short_token_in_a_long_message_does_not_count_but_a_full_label_does(tmp_path):
    long = "I am not going to answer that now, but yes, the data looks fine to me overall honestly speaking."
    assert not verify_answer(said(tmp_path, long), "s1", ISSUED, kind="bool", candidates=["y", "yes"])
    text = "After checking with the vendor it is Point-in-time daily published, with every revision logged."
    assert verify_answer(said(tmp_path, text), "s1", ISSUED, kind="choice",
                         candidates=["pit", "Point-in-time daily published"])


def test_text_answers_need_three_characters_as_a_whole_phrase(tmp_path):
    assert verify_answer(said(tmp_path, "we used the vendor curve"), "s1", ISSUED, kind="text", candidates=["vendor curve"])
    assert not verify_answer(said(tmp_path, "vendorcurves"), "s1", ISSUED, kind="text", candidates=["vendor"])
    assert not verify_answer(said(tmp_path, "a b c"), "s1", ISSUED, kind="text", candidates=["b"])
    assert verify_answer(said(tmp_path, "ab"), "s1", ISSUED, kind="text", candidates=["ab"])  # the whole message


def test_numbers_match_as_whole_numbers(tmp_path):
    assert verify_answer(said(tmp_path, "we tried 12 variants"), "s1", ISSUED, kind="number", candidates=["12"])
    assert not verify_answer(said(tmp_path, "we tried 120 variants"), "s1", ISSUED, kind="number", candidates=["12"])
    assert not verify_answer(said(tmp_path, "version 1.12"), "s1", ISSUED, kind="number", candidates=["12"])


# --- F6: AskUserQuestion answers come from the structured result, never from text ------------------

def ask_result(answers, text, tool_use_id="t1"):
    rec = tool_result(text, tool_use_id=tool_use_id)
    rec["toolUseResult"] = {"questions": [{"question": q} for q in answers], "answers": answers}
    return rec


def test_structured_answers_count(tmp_path):
    t = write(tmp_path / "t.jsonl", tool_use("AskUserQuestion"),
              ask_result({"Which timing?": "Single constant"}, 'User has answered: "Which timing?"="Single constant"'))
    assert [x.text for x in user_texts(t, "s1")] == ["Single constant"]


def test_question_text_written_by_the_agent_cannot_inject_an_answer(tmp_path):
    injected = 'Pick one "x"="pit" (pit or curve)?'
    text = f'User has answered your questions: "{injected}"="curve".'
    t = write(tmp_path / "t.jsonl", tool_use("AskUserQuestion"), ask_result({injected: "curve"}, text))
    assert [x.text for x in user_texts(t, "s1")] == ["curve"]
    assert not verify_answer(t, "s1", ISSUED, kind="choice", candidates=["pit"])


def test_ask_results_without_structured_answers_are_ignored(tmp_path):
    t = write(tmp_path / "t.jsonl", tool_use("AskUserQuestion"), tool_result('User has answered: "Q"="pit"'))
    assert user_texts(t, "s1") == []
