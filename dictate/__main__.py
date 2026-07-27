from . import config
from .transcribe import load_model, transcribe, is_hallucination
from .audio import AudioListener
from .output import type_text, press_key, run_command
from .llm import interpret, is_intentional_dictation


def on_utterance(processor, model, audio_bytes: bytes):
    text = transcribe(processor, model, audio_bytes)

    if is_hallucination(text):
        print(f"  ✗ filtered: {text!r}")
        return

    print(f"  heard: {text!r}")

    low = text.lower().strip(" .,")

    # Command mode: "computer, ..." / "shooter, ..." routes to the LLM interpreter.
    matched_prefix = next((p for p in config.COMMAND_PREFIXES if low.startswith(p)), None)
    if matched_prefix:
        command_text = text[len(matched_prefix):].strip(" .,")
        print(f"  → LLM routing: {command_text!r}")
        action, value = interpret(command_text)
        print(f"  → {action}: {value!r}")
        if action == "type":
            type_text(value)
        elif action == "key":
            press_key(value)
        elif action == "cmd":
            run_command(value)
        return

    # Plain dictation: gated by a trigger word if configured, else the LLM judges intent.
    if config.DICTATION_TRIGGERS:
        matched_trigger = next((t for t in config.DICTATION_TRIGGERS if low.startswith(t)), None)
        if matched_trigger:
            type_text(text[len(matched_trigger):].strip(" .,"))
        else:
            print(f"  ✗ no dictation trigger, ignored: {text!r}")
    elif is_intentional_dictation(text):
        type_text(text)
    else:
        print(f"  ✗ LLM filtered as unintentional: {text!r}")


def main():
    processor, model = load_model()
    listener = AudioListener(
        on_utterance=lambda audio: on_utterance(processor, model, audio)
    )
    listener.run()


if __name__ == "__main__":
    main()
