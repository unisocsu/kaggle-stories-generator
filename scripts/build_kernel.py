"""Builds kernel_build/ (main.py + kernel-metadata.json) from workflow inputs.

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


user = os.environ["KAGGLE_USERNAME"]
run_id = os.environ.get("GITHUB_RUN_ID", "local")

cfg = {
    "title": os.environ.get("INPUT_TITLE", "Untitled"),
    "instructions": os.environ.get("INPUT_INSTRUCTIONS", ""),
    "language": os.environ.get("INPUT_LANGUAGE", "Hebrew"),
    "hours": env_int("INPUT_HOURS", 5),
    "chapter_tokens": env_int("INPUT_CHAPTER_TOKENS", 4000),
    "context_tokens": env_int("INPUT_CONTEXT_TOKENS", 24000),
    "model_id": os.environ["INPUT_MODEL_ID"],
    "quant": os.environ.get("INPUT_QUANT", "bnb4"),
    "skill": pathlib.Path("skills/story-craft.md").read_text(encoding="utf-8"),
}

out = pathlib.Path("kernel_build")
out.mkdir(exist_ok=True)

template = pathlib.Path("kaggle/main.py").read_text(encoding="utf-8")
b64 = base64.b64encode(json.dumps(cfg, ensure_ascii=False).encode("utf-8")).decode("ascii")
(out / "main.py").write_text(template.replace("__CONFIG_B64__", b64), encoding="utf-8")

slug = f"book-run-{run_id}"
meta = {
    "id": f"{user}/{slug}",
    "title": slug,
    "code_file": "main.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": "true",
    "enable_gpu": "true",
    "enable_internet": "true",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

with open(os.environ.get("GITHUB_OUTPUT", os.devnull), "a", encoding="utf-8") as f:
    f.write(f"slug={slug}\n")

print("kernel folder ready:", meta["id"])
