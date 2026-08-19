#!/usr/bin/env python3
"""
update_readme.py

Reads every content/day-XXX/metadata.json file and regenerates:
  - the Daily Log table
  - the Progress line
in README.md, between the AUTO:LOG / AUTO:PROGRESS markers.

No manual README editing required — this is meant to run:
  - locally before a commit, OR
  - inside a GitHub Action (see .github/workflows/update-readme.yml)

Usage:
    python automation/scripts/update_readme.py
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTENT_DIR = REPO_ROOT / "content"
README_PATH = REPO_ROOT / "README.md"
TOTAL_DAYS = 90  # the 90-day journey


def load_all_metadata():
    entries = []
    if not CONTENT_DIR.exists():
        return entries
    for day_folder in sorted(CONTENT_DIR.glob("day-*")):
        meta_path = day_folder / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                entries.append(json.load(f))
    entries.sort(key=lambda e: e.get("day", 0))
    return entries


def build_log_table(entries):
    lines = ["| Day | Topic | Category | Difficulty | Link |", "|---|---|---|---|---|"]
    for e in entries:
        github_path = e.get("github_path") or ""
        link = f"[notes]({github_path})" if github_path else "—"
        lines.append(
            f"| {e.get('day', '?')} | {e.get('title', '')} | "
            f"{e.get('category', '')} | {e.get('difficulty', '')} | {link} |"
        )
    return "\n".join(lines)


def build_progress_line(entries):
    current_day = max((e.get("day", 0) for e in entries), default=0)
    percent = round((current_day / TOTAL_DAYS) * 100)
    return f"**Day {current_day} of {TOTAL_DAYS}** · {percent}% complete"


def replace_between_markers(text, start_marker, end_marker, new_content):
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL
    )
    replacement = f"{start_marker}\n{new_content}\n{end_marker}"
    if not pattern.search(text):
        raise ValueError(f"Markers not found in README: {start_marker} / {end_marker}")
    return pattern.sub(replacement, text)


def main():
    entries = load_all_metadata()
    if not entries:
        print("No metadata.json files found under content/. Nothing to update.")
        return

    readme_text = README_PATH.read_text(encoding="utf-8")

    log_table = build_log_table(entries)
    readme_text = replace_between_markers(
        readme_text, "<!-- AUTO:LOG:START -->", "<!-- AUTO:LOG:END -->", log_table
    )

    progress_line = build_progress_line(entries)
    readme_text = replace_between_markers(
        readme_text, "<!-- AUTO:PROGRESS:START -->", "<!-- AUTO:PROGRESS:END -->", progress_line
    )

    README_PATH.write_text(readme_text, encoding="utf-8")
    print(f"README.md updated. {len(entries)} day(s) logged.")


if __name__ == "__main__":
    main()
