"""Builds ai_task_build/task.py from kaggle/ai_task.py, workflow inputs and the skill.

Inputs arrive as environment variables (never interpolated into the shell).
"""
import base64
import json
import os
import pathlib


def env_int(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


cfg = {
    "title": os.environ.get("INPUT_TITLE", "Untitled"),
    "instructions": os.environ.get("INPUT_INSTRUCTIONS", ""),
    "language": os.environ.get("INPUT_LANGUAGE", "Hebrew"),
    "words": env_int("INPUT_WORDS", 1500),
    "skill": pathlib.Path("skills/story-craft.md").read_text(encoding="utf-8"),
}

out = pathlib.Path("ai_task_build")
out.mkdir(exist_ok=True)

template = pathlib.Path("kaggle/ai_task.py").read_text(encoding="utf-8")
b64 = base64.b64encode(json.dumps(cfg, ensure_ascii=False).encode("utf-8")).decode("ascii")
(out / "task.py").write_text(template.replace("__CONFIG_B64__", b64), encoding="utf-8")
print("task file ready:", out / "task.py")
