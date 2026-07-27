from unittest.mock import MagicMock, patch

from dictate.__main__ import on_utterance


@patch("dictate.__main__.run_command")
@patch("dictate.__main__.press_key")
@patch("dictate.__main__.type_text")
@patch("dictate.__main__.interpret")
@patch("dictate.__main__.is_hallucination", return_value=False)
@patch("dictate.__main__.transcribe")
def test_plain_speech_is_typed_directly(
    mock_transcribe, mock_is_hallucination, mock_interpret, mock_type_text, mock_press_key, mock_run_command
):
    mock_transcribe.return_value = "hello world"

    on_utterance(MagicMock(), MagicMock(), b"audio")

    mock_type_text.assert_called_once_with("hello world")
    mock_interpret.assert_not_called()
    mock_press_key.assert_not_called()
    mock_run_command.assert_not_called()


@patch("dictate.__main__.run_command")
@patch("dictate.__main__.press_key")
@patch("dictate.__main__.type_text")
@patch("dictate.__main__.interpret")
@patch("dictate.__main__.is_hallucination", return_value=False)
@patch("dictate.__main__.transcribe")
def test_hallucination_is_filtered_out(
    mock_transcribe, mock_is_hallucination, mock_interpret, mock_type_text, mock_press_key, mock_run_command
):
    mock_transcribe.return_value = "thanks for watching"
    mock_is_hallucination.return_value = True

    on_utterance(MagicMock(), MagicMock(), b"audio")

    mock_type_text.assert_not_called()
    mock_interpret.assert_not_called()
    mock_press_key.assert_not_called()
    mock_run_command.assert_not_called()


@patch("dictate.__main__.run_command")
@patch("dictate.__main__.press_key")
@patch("dictate.__main__.type_text")
@patch("dictate.__main__.interpret")
@patch("dictate.__main__.is_hallucination", return_value=False)
@patch("dictate.__main__.transcribe")
def test_command_prefix_routes_to_llm_and_types(
    mock_transcribe, mock_is_hallucination, mock_interpret, mock_type_text, mock_press_key, mock_run_command
):
    mock_transcribe.return_value = "computer, hello there"
    mock_interpret.return_value = ("type", "hello there")

    on_utterance(MagicMock(), MagicMock(), b"audio")

    mock_interpret.assert_called_once_with("hello there")
    mock_type_text.assert_called_once_with("hello there")


@patch("dictate.__main__.run_command")
@patch("dictate.__main__.press_key")
@patch("dictate.__main__.type_text")
@patch("dictate.__main__.interpret")
@patch("dictate.__main__.is_hallucination", return_value=False)
@patch("dictate.__main__.transcribe")
def test_command_prefix_routes_to_key_action(
    mock_transcribe, mock_is_hallucination, mock_interpret, mock_type_text, mock_press_key, mock_run_command
):
    mock_transcribe.return_value = "computer select all"
    mock_interpret.return_value = ("key", "ctrl+a")

    on_utterance(MagicMock(), MagicMock(), b"audio")

    mock_interpret.assert_called_once_with("select all")
    mock_press_key.assert_called_once_with("ctrl+a")
    mock_type_text.assert_not_called()


@patch("dictate.__main__.run_command")
@patch("dictate.__main__.press_key")
@patch("dictate.__main__.type_text")
@patch("dictate.__main__.interpret")
@patch("dictate.__main__.is_hallucination", return_value=False)
@patch("dictate.__main__.transcribe")
def test_command_prefix_routes_to_cmd_action(
    mock_transcribe, mock_is_hallucination, mock_interpret, mock_type_text, mock_press_key, mock_run_command
):
    mock_transcribe.return_value = "shooter open a terminal"
    mock_interpret.return_value = ("cmd", "gnome-terminal")

    on_utterance(MagicMock(), MagicMock(), b"audio")

    mock_interpret.assert_called_once_with("open a terminal")
    mock_run_command.assert_called_once_with("gnome-terminal")
    mock_type_text.assert_not_called()
    mock_press_key.assert_not_called()
