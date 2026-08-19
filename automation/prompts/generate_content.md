# Content Generation Prompts (Phase 3 & 4)

These are the two prompts an AI step (OpenAI API, Claude API, etc.) will use inside
the n8n workflow. Same topic in, two independent outputs out. Neither prompt
sees the other's output.

---

## GitHub Technical Note Prompt

```
You are writing a technical reference note for a public Data Science learning repository.

Topic: {{topic}}
Audience: beginners preparing for data roles, revising for interviews.

Write in this exact structure:
1. Definition (precise, no analogies)
2. Rule of Thumb (one memorable line)
3. Technical Comparison (table, if applicable)
4. Code Example (minimal, runnable)
5. Best Practices (bulleted)
6. Common Interview Question (Q&A)
7. Summary (2-3 sentences)

Tone: technical, clear, documentation-style. No storytelling, no first-person
narrative, no emojis, no hooks. This is a reference, not a post.
```

---

## LinkedIn Story Prompt

```
You are writing a LinkedIn post for a personal brand called "The Data Diet,"
about someone learning Data Science in public, one concept a day.

Topic: {{topic}}
Audience: professionals and learners on LinkedIn.

Write in this exact structure:
1. Hook (first line must stop the scroll — a surprising fact, a personal
   moment, or a bold claim)
2. A short personal story or moment tied to learning this topic
3. An analogy that makes the concept intuitive to a non-technical reader
4. A simple, jargon-free explanation
5. A call to action (a question inviting comments/discussion)
6. 4-6 relevant hashtags

Tone: human, first-person, conversational. Never read like documentation.
Do not include tables, code blocks, or formal technical structure.
```

---

## Notes

- Both prompts should be called with the SAME research notes as input, but
  generate completely independently — do not let one output influence the other.
- Output from each prompt maps directly into `content/day-XXX/github.md` and
  `content/day-XXX/linkedin.md`.
- `metadata.json` is generated separately (day number, title, category, tags,
  difficulty) and used to trigger the README automation.
