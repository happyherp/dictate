from unittest.mock import patch

from dictate.output import type_text


def test_type_text_uses_minimal_delay():
    with patch("dictate.output.subprocess.run") as mock_run:
        type_text("hello world")

    args = mock_run.call_args.args[0]
    assert args[0] == "xdotool"
    assert "--delay" in args
    assert args[args.index("--delay") + 1] == "1"
    assert args[-1] == "hello world"
