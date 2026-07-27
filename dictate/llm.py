import anthropic
from . import config

SYSTEM_PROMPT = """You are a voice command interpreter for a Linux desktop.
Given transcribed speech, respond with exactly one of:
  TYPE: <text to type>
  CMD: <shell command to run>
  KEY: <xdotool key name>

Examples:
  "open a terminal" -> CMD: gnome-terminal
  "switch window" -> KEY: alt+Tab
  "close this window" -> KEY: alt+F4
  "volume up" -> CMD: pactl set-sink-volume @DEFAULT_SINK@ +10%
  "volume down" -> CMD: pactl set-sink-volume @DEFAULT_SINK@ -10%
  "go to desktop" -> KEY: super+d
  "take a screenshot" -> KEY: Print
  "select all" -> KEY: ctrl+a
  "copy" -> KEY: ctrl+c
  "paste" -> KEY: ctrl+v
  "undo" -> KEY: ctrl+z
  "hello world" -> TYPE: hello world

Only respond with TYPE/CMD/KEY and nothing else."""


def interpret(text: str) -> tuple[str, str]:
    """Returns (action_type, value) where action_type is 'type'/'cmd'/'key'."""
    client = anthropic.Anthropic(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api",
    )
    message = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    response = message.content[0].text.strip()
    if response.startswith("CMD: "):
        return "cmd", response[5:]
    if response.startswith("KEY: "):
        return "key", response[5:]
    if response.startswith("TYPE: "):
        return "type", response[6:]
    # Fallback: treat as plain text
    return "type", text
