# -*- coding: utf-8 -*-
"""Single-model book writer for a Kaggle GPU script kernel.

The config (including the story-craft skill text) is injected by
scripts/build_kernel.py at the placeholder below.

Flow:
  1. story notes (premise, characters, world, ending)
  2. outline: the model decides how many chapters the story needs, within a
     limit computed from the measured generation speed and the time budget
  3. chapter by chapter, each followed by a short summary for continuity
Output goes to /kaggle/working (book.md, chapter files, notes, outline).
"""
import base64
import json
import os
import re
import subprocess
import sys
import time

START = time.time()
CFG = json.loads(base64.b64decode("__CONFIG_B64__").decode("utf-8"))
OUT = "/kaggle/working"
os.environ.setdefault("HF_HOME", "/kaggle/temp/hf")  # bigger disk than /kaggle/working
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Leave 10 minutes of margin before the requested number of hours.
BUDGET = CFG["hours"] * 3600 - 600


def log(*a):
    print(f"[{int(time.time() - START):>6}s]", *a, flush=True)


def remaining():
    return BUDGET - (time.time() - START)


subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers", "accelerate", "bitsandbytes", "sentencepiece"],
    check=True,
)
subprocess.run("nvidia-smi", shell=True)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoTokenizer, BitsAndBytesConfig  # noqa: E402

# ---------------------------------------------------------------- model load
log("loading", CFG["model_id"])
tok = AutoTokenizer.from_pretrained(CFG["model_id"])

# "balanced" spreads the weights evenly so each GPU keeps free room for prefill
kw = dict(device_map="balanced", dtype=torch.float16)
n_gpu = torch.cuda.device_count()
if n_gpu:
    # leave headroom on each card for the KV cache
    kw["max_memory"] = {i: "13GiB" for i in range(n_gpu)}
if CFG["quant"] == "bnb4":
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

cls = getattr(transformers, "AutoModelForMultimodalLM", transformers.AutoModelForCausalLM)
try:
    model = cls.from_pretrained(CFG["model_id"], **kw)
except Exception as e:  # fall back to the plain causal-LM class
    log("falling back to AutoModelForCausalLM:", repr(e)[:300])
    model = transformers.AutoModelForCausalLM.from_pretrained(CFG["model_id"], **kw)
model.eval()
log("model loaded")

# -------------------------------------------------------------------- skill
CTX = CFG["context_tokens"]
LANG = CFG["language"]
MAX_NEW = CFG["chapter_tokens"]
WORDS = max(300, MAX_NEW // 3)


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


SK = parse_skill(CFG.get("skill", ""))
if not SK:
    log("WARNING: skill is empty, writing without the craft guide")

STAGES = {
    "plan": ["core principles", "planning the book", "language notes"],
    "write": ["core principles", "writing a chapter",
              "avoid machine-sounding prose", "language notes"],
    "sum": ["continuity"],
}
BASE = (
    f"You are an award-winning novelist and a careful editor. Write in {LANG}. "
    "Never add meta commentary and never mention being an AI. "
    "Follow the craft guide below."
)


def system_for(stage):
    body = "\n\n".join(f"### {n.title()}\n{SK[n]}" for n in STAGES[stage] if SK.get(n))
    return BASE + ("\n\n# CRAFT GUIDE\n" + body if body else "")


SYS = {s: system_for(s) for s in STAGES}

# ------------------------------------------------------------------- helpers
STATS = {"tok": 0, "sec": 0.0}


def tps():
    return STATS["tok"] / STATS["sec"] if STATS["sec"] > 0 else 3.0


def render(system, user):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except Exception:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def ntok(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])


CHUNK = 1024  # prefill is processed in chunks of this many tokens
USE_CHUNK = True


def cleanup():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _generate(enc, max_new, temperature):
    global USE_CHUNK
    extra = {"prefill_chunk_size": CHUNK} if USE_CHUNK else {}
    try:
        return model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.05,
            **extra,
        )
    except torch.OutOfMemoryError:
        raise
    except Exception as e:
        if not USE_CHUNK:
            raise
        log("chunked prefill not usable, disabling it:", repr(e)[:200])
        USE_CHUNK = False
        cleanup()
        return _generate(enc, max_new, temperature)


def chat(system, user, max_new, temperature=0.8):
    text = render(system, user)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_in = enc["input_ids"].shape[-1]
    t0 = time.time()
    with torch.inference_mode():
        out = _generate(enc, max_new, temperature)
    STATS["sec"] += time.time() - t0
    STATS["tok"] += out.shape[-1] - n_in
    txt = tok.decode(out[0][n_in:], skip_special_tokens=True)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
    del enc, out
    cleanup()
    return txt


def write(name, text):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)


chapters = []


def save_book():
    body = f"# {CFG['title']}\n\n" + "\n\n".join(chapters)
    write("book.md", body)


def parse_outline(raw):
    return [l.strip() for l in raw.splitlines() if re.match(r"^\s*\d+[\.\)]\s*\S", l)]


# ---------------------------------------------------------------------- main
status = {"state": "started"}
try:
    head = f"BOOK TITLE: {CFG['title']}\n\nAUTHOR'S INSTRUCTIONS:\n{CFG['instructions']}\n\n"

    # 1) story notes
    log("writing story notes")
    notes = chat(
        SYS["plan"],
        head + "Write concise story notes (max 700 words): premise and core question, "
        "setting, tone and narrative voice (point of view and tense), main characters "
        "with goals, flaws and the false belief each must confront, the opposing force, "
        "key rules of the world, and the intended ending.",
        max_new=1500,
        temperature=0.7,
    )
    write("notes.md", notes)

    # 2) chapter limit from measured speed, then let the model choose
    per_chapter = (MAX_NEW + 300) / tps()  # generation + summary, in seconds
    usable = remaining() * 0.92 - 2500 / tps()
    capacity = max(3, int(usable // per_chapter))
    log(f"speed {tps():.2f} tok/s -> time budget allows at most {capacity} chapters")

    outline_prompt = (
        head + f"STORY NOTES:\n{notes}\n\n"
        "Plan the chapters. Decide for yourself how many chapters this story needs: "
        "let the shape of the story decide, not a round number. "
        f"Hard limits: at least 3 and at most {capacity} chapters; each chapter will be "
        f"about {WORDS} words, so the whole story must fit. "
        "Output the first line exactly as `CHAPTERS: <number>`, then one chapter per "
        "line in the format `N. Chapter title - one sentence about what happens and "
        "how the situation turns`. The last chapter must deliver the ending. "
        "Output nothing else."
    )
    log("writing outline")
    outline_raw = chat(SYS["plan"], outline_prompt, max_new=2500, temperature=0.7)
    lines = parse_outline(outline_raw)
    if len(lines) > capacity:
        log(f"outline has {len(lines)} chapters, limit is {capacity}; asking for a tighter one")
        outline_raw = chat(
            SYS["plan"],
            outline_prompt + f"\n\nYour previous outline had {len(lines)} chapters, which is "
            f"too many. Rewrite it with at most {capacity} chapters, merging or cutting "
            "material while keeping the ending.",
            max_new=2500,
            temperature=0.6,
        )
        lines = parse_outline(outline_raw)
    if not lines:
        lines = [f"{i}. Chapter {i}" for i in range(1, min(capacity, 12) + 1)]
    lines = lines[:capacity]
    outline_text = "\n".join(lines)
    write("outline.md", outline_text)
    log(f"the model planned {len(lines)} chapters")

    # 3) chapters
    summaries = []
    tail = "(none)"
    durations = []

    for i, line in enumerate(lines, start=1):
        est = (sum(durations) / len(durations) * 1.25) if durations else 0
        if remaining() < max(est, 300):
            log("time budget reached, stopping before chapter", i)
            break
        t0 = time.time()

        def build_user(limit, tail_chars):
            s = list(summaries)
            while True:
                prev = "\n".join(s) if s else "(this is the first chapter)"
                final_note = (
                    " This is the final chapter: bring the story to its ending."
                    if i == len(lines)
                    else " Do not wrap up the whole book in this chapter."
                )
                user = (
                    head
                    + f"STORY NOTES:\n{notes}\n\nFULL OUTLINE:\n{outline_text}\n\n"
                    + f"STORY SO FAR (chapter summaries):\n{prev}\n\n"
                    + f"END OF THE PREVIOUS CHAPTER:\n{tail[-tail_chars:]}\n\n"
                    + f"Now write chapter {line}\n\nTarget length: about {WORDS} words. "
                    "Start with the chapter heading, then the chapter text." + final_note
                )
                if ntok(SYS["write"] + user) <= limit - MAX_NEW or len(s) <= 1:
                    return user
                s.pop(0)  # drop the oldest summary to stay inside the context budget

        log(f"writing chapter {i}/{len(lines)}")
        text = None
        for scale in (1.0, 0.7, 0.5):
            user = build_user(int(CTX * scale), int(1500 * scale))
            try:
                text = chat(SYS["write"], user, max_new=MAX_NEW)
            except torch.OutOfMemoryError:
                text = None
            if text is not None:
                break
            log(f"out of memory at context scale {scale}, retrying with a smaller prompt")
            cleanup()
        if text is None:
            raise RuntimeError("out of memory even with a reduced context")
        chapters.append(text)
        write(f"chapter_{i:02d}.md", text)
        tail = text[-1500:]

        summ = None
        body = text
        for _ in range(2):
            try:
                summ = chat(
                    SYS["sum"],
                    "Summarize this chapter in about 150 words as a continuity record. "
                    f"Output only the summary.\n\n{body}",
                    max_new=450,
                    temperature=0.3,
                )
            except torch.OutOfMemoryError:
                summ = None
            if summ is not None:
                break
            cleanup()
            body = body[-int(len(body) * 0.6):]
        if summ is None:
            summ = "(summary unavailable) " + text[:300]
        summaries.append(f"Chapter {i}: {summ}")
        write("summaries.md", "\n".join(summaries))
        save_book()  # partial book is always available

        durations.append(time.time() - t0)
        log(f"chapter {i} done in {int(durations[-1])}s")

    status = {
        "state": "finished",
        "chapters_written": len(chapters),
        "chapters_planned": len(lines),
        "chapter_limit_from_time": capacity,
        "tokens_per_second": round(tps(), 2),
        "chunked_prefill": USE_CHUNK,
    }
except Exception as e:  # keep whatever was written
    status = {"state": "error", "error": repr(e)[:500], "chapters_written": len(chapters)}
    log("ERROR", repr(e))
    raise
finally:
    save_book()
    write("status.json", json.dumps(status, ensure_ascii=False, indent=2))
    log("status:", status)
