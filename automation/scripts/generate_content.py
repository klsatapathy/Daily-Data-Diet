"""
The Data Diet - daily content generator.

Reads topics.json + state.json, calls the free Gemini API TWICE per run
(one call for the GitHub technical note, one independent call for the
LinkedIn story post - matching automation/prompts/generate_content.md,
which specifies the two should never influence each other), and writes:

  content/day-XXX/github.md
  content/day-XXX/linkedin.md
  content/day-XXX/metadata.json

Then advances state.json to the next topic in the round-robin rotation.
update_readme.py (triggered separately by the metadata.json push) picks
this up automatically once the PR is merged.
"""

import json
import os
import re
import sys
import time
import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]


def gemini_url(model):
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={GEMINI_API_KEY}"
    )

TOPICS_FILE = "topics.json"
STATE_FILE = "state.json"
CONTENT_DIR = "content"

DIFFICULTY_DEFAULT = "Beginner"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def slugify(text):
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"\s+", "-", text)


def call_gemini(prompt, max_retries=4):
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY env var is not set.")
        sys.exit(1)
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for model in GEMINI_MODELS:
        url = gemini_url(model)
        for attempt in range(max_retries):
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code in (429, 503) and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                print(f"{model} -> {resp.status_code}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code in (429, 503):
                print(f"{model} still failing after retries, trying next model...")
                break
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    print("ERROR: all Gemini models failed (persistent 429/503).")
    sys.exit(1)


def github_prompt(pillar, topic, day_number, tags):
    return f"""You are writing a technical reference note for a public Data
Science learning repository called "The Data Diet".

Topic: {topic}
Pillar/Category: {pillar}
Audience: beginners preparing for data roles, revising for interviews.

Write the BODY ONLY (no title line - that gets added separately) in this
exact markdown structure, using these exact headers:

## Definition
(precise, no analogies)

## Rule of Thumb
(one memorable line)

## Technical Comparison
(a markdown table, only if genuinely applicable to this topic - otherwise
a short bulleted list of key facts instead)

## Code Example
(a minimal, runnable Python or SQL code block, whichever fits the topic)

## Best Practices
(3-5 bullet points)

## Common Interview Question
**Q:** ...
**A:** ...

## Summary
(2-3 sentences)

Tone: technical, clear, documentation-style. No storytelling, no
first-person narrative, no emojis, no hooks. This is a reference note,
not a social post. Output ONLY the markdown body starting from "## Definition".
"""


def linkedin_prompt(pillar, topic, day_number, tags):
    return f"""You are writing a LinkedIn post for a personal brand called
"The Data Diet" (🥗), about someone learning Data Science in public, one
concept a day, for a 90-day journey.

Topic: {topic}
Pillar/Category: {pillar}
Day number: {day_number}
Audience: professionals and learners on LinkedIn.

Write the post with this structure (don't label the sections, just flow
naturally):
1. Hook - first line must stop the scroll (a surprising fact, a personal
   moment, or a bold claim)
2. A short personal story or moment tied to learning this topic
3. An analogy that makes the concept intuitive to a non-technical reader
4. A simple, jargon-free explanation
5. Mention "Day {day_number} of 90" somewhere naturally
6. A call to action - a question inviting comments/discussion
7. End with 4-6 relevant hashtags including #TheDataDiet

Tone: human, first-person, conversational, a little playful. Never read
like documentation. No tables, no code blocks, no formal structure.
Keep it 250-400 words. Output ONLY the finished post text, nothing else.
"""


def main():
    topics = load_json(TOPICS_FILE)
    state = load_json(STATE_FILE)

    pillars = topics["pillars"]
    idx = state["current_index"]
    day_number = state["day_number"]

    pillar = pillars[idx % len(pillars)]
    topic_list = topics["topics_by_pillar"][pillar]
    topic_idx = idx // len(pillars)

    if topic_idx >= len(topic_list):
        print(f"All topics exhausted for pillar '{pillar}'. Add more topics to topics.json.")
        sys.exit(1)

    topic = topic_list[topic_idx]
    tags = [slugify(pillar), slugify(topic).split("-")[0]]

    github_body = call_gemini(github_prompt(pillar, topic, day_number, tags))
    linkedin_body = call_gemini(linkedin_prompt(pillar, topic, day_number, tags))

    day_folder = f"{CONTENT_DIR}/day-{day_number:03d}"
    os.makedirs(day_folder, exist_ok=True)

    github_content = (
        f"# Day {day_number} — {topic}\n\n"
        f"**Category:** {pillar} · **Difficulty:** {DIFFICULTY_DEFAULT} · "
        f"**Tags:** {', '.join(tags)}\n\n---\n\n{github_body}\n"
    )
    with open(f"{day_folder}/github.md", "w", encoding="utf-8") as f:
        f.write(github_content)

    linkedin_content = (
        f"<!--\nLinkedIn Draft — Day {day_number} — {topic}\n"
        f"Status: draft (requires manual approval before publish)\n-->\n\n"
        f"{linkedin_body}\n"
    )
    with open(f"{day_folder}/linkedin.md", "w", encoding="utf-8") as f:
        f.write(linkedin_content)

    metadata = {
        "day": day_number,
        "title": topic,
        "category": pillar,
        "difficulty": DIFFICULTY_DEFAULT,
        "tags": tags,
        "linkedin_status": "draft",
        "github_status": "draft",
        "linkedin_url": "",
        "github_path": f"{day_folder}/github.md",
        "date": "",
    }
    with open(f"{day_folder}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open("pr_title.txt", "w", encoding="utf-8") as f:
        f.write(f"Day {day_number}: {topic} ({pillar})")

    with open("telegram_message.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Data Diet - Day {day_number} draft ready ({pillar})\n\n"
            f"{linkedin_body}\n\n---\nGitHub notes: {day_folder}/github.md"
        )

    state["current_index"] = idx + 1
    state["day_number"] = day_number + 1
    save_json(STATE_FILE, state)

    print(f"Generated Day {day_number}: {topic} ({pillar}) -> {day_folder}/")


if __name__ == "__main__":
    main()
