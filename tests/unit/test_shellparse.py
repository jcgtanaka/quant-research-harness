"""shellparse: a small, quote-aware split of a shell command into simple commands (used by the guards)."""

from qrh.hooks.shellparse import parse


def words(command):
    return [c.words for c in parse(command)]


def test_splits_on_control_operators_newlines_and_substitutions():
    assert words("a b && c || d; e | f\ng $(h i) `j k`") == [
        ["a", "b"], ["c"], ["d"], ["e"], ["f"], ["g", "$"], ["h", "i"], ["j", "k"]]


def test_quoted_text_stays_one_word():
    assert words('git commit -m "fix git push bug; really"') == [["git", "commit", "-m", "fix git push bug; really"]]


def test_redirect_targets_are_collected_and_removed_from_words():
    (cmd,) = parse("echo x >> out.txt 2>&1 > other.txt")
    assert cmd.redirects == ["out.txt", "other.txt"]
    assert "out.txt" not in cmd.words


def test_shell_dash_c_scripts_are_parsed_too():
    flat = [w for c in parse("bash -c 'git push origin main; ls'") for w in c.words]
    assert "push" in flat and "ls" in flat
    flat = [w for c in parse('sudo sh -lc "cd /x && git push"') for w in c.words]
    assert "push" in flat


def test_heredoc_body_is_dropped_unless_a_shell_may_run_it():
    assert all("push" not in c.words for c in parse("git commit -F - <<'EOF'\nrun git push later\nEOF"))
    flat = [w for c in parse("cat <<EOF | bash\ngit push\nEOF") for w in c.words]
    assert "push" in flat


def test_unbalanced_quotes_still_parse_best_effort():
    cmds = parse("echo don't; git push")
    assert any(c.words[:2] == ["git", "push"] for c in cmds)
