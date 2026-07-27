import os
import subprocess
from . import config

_ENV = {**os.environ, "DISPLAY": config.DISPLAY}


def type_text(text: str):
    subprocess.run(
        ["xdotool", "type", "--clearmodifiers", "--delay", "1", "--", text], env=_ENV
    )


def press_key(key: str):
    subprocess.run(["xdotool", "key", "--clearmodifiers", key], env=_ENV)


def run_command(cmd: str):
    subprocess.Popen(cmd, shell=True, env=_ENV)
