from unittest.mock import patch

from dictate import config
from dictate.output import press_key, run_command, type_text


@patch("dictate.output.subprocess.run")
def test_type_text_invokes_xdotool_type(mock_run):
    type_text("hello world")
    args, kwargs = mock_run.call_args
    assert args[0] == ["xdotool", "type", "--clearmodifiers", "--", "hello world"]
    assert kwargs["env"]["DISPLAY"] == config.DISPLAY


@patch("dictate.output.subprocess.run")
def test_press_key_invokes_xdotool_key(mock_run):
    press_key("ctrl+a")
    args, kwargs = mock_run.call_args
    assert args[0] == ["xdotool", "key", "--clearmodifiers", "ctrl+a"]
    assert kwargs["env"]["DISPLAY"] == config.DISPLAY


@patch("dictate.output.subprocess.Popen")
def test_run_command_invokes_shell(mock_popen):
    run_command("echo hi")
    args, kwargs = mock_popen.call_args
    assert args[0] == "echo hi"
    assert kwargs["shell"] is True
    assert kwargs["env"]["DISPLAY"] == config.DISPLAY
