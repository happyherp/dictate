import pytest

from dictate import config
from dictate.llm import interpret, is_intentional_dictation

pytestmark = pytest.mark.skipif(
    not config.OPENROUTER_API_KEY,
    reason="requires OPENROUTER_API_KEY to make a real OpenRouter call",
)


def test_interpret_routes_command_to_cmd():
    action, value = interpret("open a terminal")
    assert action == "cmd"
    assert "terminal" in value.lower()


def test_interpret_routes_shortcut_to_key():
    action, value = interpret("select all")
    assert action == "key"
    assert value.lower() == "ctrl+a"


def test_interpret_falls_back_to_type():
    action, value = interpret("hello world")
    assert action == "type"
    assert value.strip().lower() == "hello world"


def test_is_intentional_dictation_accepts_real_sentence():
    assert is_intentional_dictation(
        "Please remember to pick up milk on the way home from work."
    )


def test_is_intentional_dictation_rejects_disfluent_fragment():
    assert not is_intentional_dictation("the the the the")
