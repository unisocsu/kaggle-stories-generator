"""Finds the chapter in the downloaded *.run.json files and writes chapter_extracted.md.

The exact layout of run files is not documented here, so the search is defensive:
first a value under the key "chapter", otherwise the longest string in the file.
"""
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k, v
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def maybe_json(s):
    s = s.strip()
    if s[:1] in "{[":
        try:
            return json.loads(s)
        except ValueError:
            return None
    return None


def find_chapter(data):
    # 1) a "chapter" key, possibly inside a JSON-encoded string
    stack = [data]
    while stack:
        cur = stack.pop()
        for k, v in walk(cur):
            if k == "chapter" and isinstance(v, str) and len(v) > 200:
                return v
            if isinstance(v, str) and v.lstrip()[:1] == "{":
                inner = maybe_json(v)
                if inner is not None:
                    stack.append(inner)
    # 2) fallback: longest string anywhere
    best = ""
    for _, v in walk(data):
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best


files = sorted(root.rglob("*.run.json"))
print(f"found {len(files)} run file(s) under {root}")
if not files:
    print("no run files: check the status and log steps above")
    sys.exit(0)

lines = []
for i, f in enumerate(files, start=1):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except ValueError as e:
        print("could not parse", f, e)
        continue
    text = find_chapter(data)
    out = f.with_name(f"chapter_extracted_{i}.md")
    out.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"{f}: {words} words -> {out}")
    lines.append(f"### {f.parent.name} ({words} words)\n\n{text[:1200]}\n")

summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary and lines:
    with open(summary, "a", encoding="utf-8") as s:
        s.write("## Chapter preview\n\n" + "\n".join(lines))
