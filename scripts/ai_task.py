# %%
# Kaggle Benchmarks task: writes notes + chapter 1 of a story with a hosted model.
# Runs on Kaggle's servers (no GPU) and uses the AI quota, not the GPU quota.
# The config (including the story-craft skill text) is injected by
# scripts/build_ai_task.py at the placeholder below.
import base64
import json
import re

import kaggle_benchmarks as kbench

CFG = json.loads(base64.b64decode("__CONFIG_B64__").decode("utf-8"))


# %%
def parse_skill(text):
    """Split the markdown skill into {lowercased '## heading': body}."""
    sections, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


SK = parse_skill(CFG["skill"])
NAMES = [
    "core principles",
    "planning the book",
    "writing a chapter",
    "avoid machine-sounding prose",
    "language notes",
]
GUIDE = "\n\n".join(f"### {n.title()}\n{SK[n]}" for n in NAMES if SK.get(n))
LANG = CFG["language"]
WORDS = CFG["words"]

SYSTEM = (
    f"You are an award-winning novelist and a careful editor. Write in {LANG}. "
    "Never add meta commentary and never mention being an AI. "
    "Follow the craft guide below.\n\n# CRAFT GUIDE\n" + GUIDE
)

HEAD = f"BOOK TITLE: {CFG['title']}\n\nAUTHOR'S INSTRUCTIONS:\n{CFG['instructions']}\n\n"

NOTES_PROMPT = (
    HEAD
    + "Write concise story notes (max 500 words): premise and core question, setting, "
    "tone and narrative voice (point of view and tense), main characters with goals, "
    "flaws and the false belief each must confront, the opposing force, the intended "
    "ending, and a one-paragraph plan for chapter 1."
)

CHAPTER_PROMPT = (
    f"Now write chapter 1 of the book, about {WORDS} words, following the notes and "
    "the craft guide. Start with the chapter heading, then the chapter text. Open on "
    "the day something changes and end on a turn, not a summary."
)


# %%
@kbench.task(name="story-chapter-test")
def story_chapter_test(llm) -> dict:
    kbench.system.send(SYSTEM)
    notes = llm.prompt(NOTES_PROMPT, temperature=0.7)
    chapter = llm.prompt(CHAPTER_PROMPT, temperature=0.8)

    kbench.assertions.assert_not_empty(chapter, expectation="The chapter must not be empty.")
    words = len(chapter.split())
    kbench.assertions.assert_true(
        words >= 300, expectation="The chapter should have at least 300 words."
    )
    return {"words": words, "notes": notes, "chapter": chapter}


# %%
story_chapter_test.run(kbench.llm)
