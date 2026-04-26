# SnapKV Notebook Storyline

This project is a storytelling-first notebook built around **one primary paper**:

- **SnapKV: LLM Knows What You are Looking for Before Generation**
  Paper: https://arxiv.org/abs/2404.14469

The goal is not to reproduce the entire paper benchmark-by-benchmark. The goal is to make the core idea behind KV cache optimization feel **explainable, intuitive, interactive, and beautiful** for a notebook competition setting.

## Positioning

The notebook should feel like:

**What should an LLM remember?**

That framing is stronger than presenting SnapKV as just another systems optimization. It lets us:

- start from the reader's intuition
- introduce KV cache naturally
- use real-world memory analogies
- add game-like demos
- make SnapKV feel inevitable rather than abrupt

## Citation Strategy

Only **SnapKV** should be treated as the main cited paper for the competition entry.

Other KV-cache papers can still influence the notebook, but only in a light way:

- brief theory mentions
- rough historical context
- no heavy comparative citation structure

In other words:

- **formal anchor**: SnapKV
- **background inspiration**: a few earlier KV-cache ideas mentioned informally in theory sections

## Storyline

The notebook should progress as a story rather than as a survey.

### 1. Why do LLMs need memory?

Start from a human example:

When we respond in a conversation, we do not re-read the entire chat from the beginning every second. We keep a working memory of what matters. LLMs do something similar during generation.

This is the entry point to:

- autoregressive generation
- attention over previous tokens
- why recomputing everything repeatedly is wasteful

### 2. What is KV cache?

Introduce KV cache as the model's reusable short-term memory during generation.

Key idea:

- once keys and values are computed for earlier tokens, they are cached
- future generated tokens reuse them instead of recomputing them

This section should be clear, visual, and lightweight.

### 3. Why does KV cache become a problem?

This is the tension point in the story.

As prompts get longer, KV cache grows. That means:

- more memory usage
- more bandwidth pressure
- more difficulty serving long-context tasks efficiently

This is where the reader should feel the problem before seeing the solution.

### 4. Naive solutions are not enough

Before SnapKV, walk the reader through natural but imperfect ideas:

- keep everything
- keep only the most recent tokens
- drop tokens randomly

This creates a perfect interactive section:

> If you had limited memory, what would you keep?

### 5. Why SnapKV exists

This is the notebook's central turn.

SnapKV's insight is beautiful:

> The end of the prompt often reveals what the model will need next.

So instead of keeping the entire KV cache, SnapKV uses an **observation window** near the end of the prompt to estimate which earlier tokens will matter most during decoding.

This makes the story feel much more intelligent than "just compression."

### 6. What SnapKV does

The reader should understand this pipeline clearly:

1. Look at the last part of the prompt.
2. Measure which earlier tokens this part attends to.
3. Score earlier positions by importance.
4. Keep only the important KV entries under a fixed budget.
5. Decode using this compressed memory.

### 7. Why this feels intuitive

Bring it back to human memory:

- recency matters sometimes
- repeated importance matters sometimes
- but current intent matters a lot

SnapKV feels intuitive because it says:

> "What I am trying to do right now tells you what I should remember."

### 8. What comes after SnapKV

Close with a very light future-looking section:

- memory can be selected more intelligently
- memory can be compressed differently
- memory budgets may vary across layers or tasks

This should remain light and non-citation-heavy.

## Real-World Examples

Yes, the notebook should be rooted in **real-world and intuitive examples**, not just abstract matrices.

These examples should appear throughout the notebook:

### Conversation memory

Example:

- a long chat contains many side topics
- the final user question asks about one specific earlier detail
- the notebook shows that not every previous token matters equally

Why it works:

- instantly relatable
- mirrors real assistant behavior
- naturally motivates targeted memory retention

### Shopping or planning memory

Example:

- a person discusses clothes, budget, shoe type, trip length, weather
- the final question is: "Which running shoes under my budget fit the trip?"
- the current goal determines which prior details matter

Why it works:

- very human
- easy to visualize
- maps naturally onto selective memory

### Studying for an exam

Example:

- many notes are seen over time
- only a few become relevant for the current question
- memory is not about storing everything equally

Why it works:

- intuitive "triage" analogy
- strong bridge into eviction/compression ideas

### Needle-in-a-haystack style retrieval

Example:

- hide one fact in a long passage
- ask a question that depends on it later
- compare naive retention with targeted retention

Why it works:

- game-like
- visually satisfying
- easy to make interactive

## Interactive Philosophy

The notebook should be friendly to both readers and tinkerers.

That means a few deliberate design choices:

- simple editable values
- visible outputs after each change
- visual feedback instead of dense text alone
- small, self-contained code cells

The notebook should not force users to understand all implementation details before they can experiment.

## Recommended Editable Cells

Add clear cells labeled something like:

**Try changing these values**

Examples:

```python
context_length = 128
kv_budget = 24
observation_window = 16
needle_position = 67
policy = "snapkv"
```

These are ideal because they let the user interact with the notebook without touching deeper logic.

## Code Block Direction

The code should focus on intuition first.

### Good code blocks

- tiny attention visualizations
- token importance heatmaps
- memory budget games
- side-by-side policy comparison
- human memory analogy simulators

### Bad early code blocks

- huge framework patching
- complex kernel-level optimization
- long implementation details before intuition lands

## Draft Intuition Block

One of the most important sections should be a "daily life memory" block.

### Theory text

Imagine you are preparing to answer a question from a long conversation. You do not keep every sentence equally active in your mind. You tend to remember:

- things that are recent
- things that repeated
- things that are relevant to the current goal

SnapKV mirrors this third behavior especially well: current intent helps decide what to keep.

### Draft code

```python
import numpy as np
import matplotlib.pyplot as plt

events = [
    "saw a shoe ad",
    "friend mentioned sneakers",
    "walked past a sports store",
    "read laptop review",
    "checked running routes",
    "opened shoe size chart",
    "searched: best running shoes"
]

recent_weight = 0.35
repeat_weight = 0.25
goal_weight = 0.40

repeat_score = np.array([0.3, 0.7, 0.6, 0.2, 0.5, 0.8, 1.0])
goal_score = np.array([0.4, 0.7, 0.8, 0.1, 0.9, 0.95, 1.0])
recent_score = np.linspace(0.2, 1.0, len(events))

final_score = (
    recent_weight * recent_score
    + repeat_weight * repeat_score
    + goal_weight * goal_score
)

top_k = 3
keep_idx = np.argsort(final_score)[-top_k:][::-1]

plt.figure(figsize=(10, 4))
bars = plt.bar(range(len(events)), final_score, color="#c7dcef")
for i in keep_idx:
    bars[i].set_color("#2b6cb0")

plt.xticks(range(len(events)), events, rotation=35, ha="right")
plt.ylabel("importance")
plt.title("Human memory triage: keep what matters most")
plt.tight_layout()
plt.show()

print("Memories kept:")
for i in keep_idx:
    print("-", events[i])
```

Why this block matters:

- it is editable
- it is visual
- it is non-technical at first glance
- it gives a human explanation for selective memory

## Suggested Notebook Sections

This is the recommended high-level structure:

1. Title and promise
2. Why LLMs need memory
3. What KV cache is
4. Why KV cache becomes expensive
5. A game: what would you keep?
6. Daily-life memory analogy
7. SnapKV intuition
8. SnapKV mechanism
9. Visualizing the observation window
10. Interactive KV-budget experiments
11. A small "build your own memory policy" section
12. Light closing section on future directions

## Visual Style Notes

The notebook should feel polished and welcoming:

- crisp section dividers
- clear headings
- attractive but not noisy plots
- a few highlighted interactive controls
- cells that invite editing

The reader should feel like they are exploring an exhibit, not reading lecture notes.

## Proposed Big Interactive Hook

The strongest audience-facing feature would be a small game or sandbox:

### Memory DJ

The user pastes a long prompt or uses a toy prompt. Then they:

- choose a memory budget
- choose a retention policy
- ask a final question
- see which tokens are preserved

Policies can include:

- full cache
- recent only
- random
- simple heuristic
- SnapKV-style selection

This gives the notebook a strong "play with the idea" component.

## What This Project Should Optimize For

Priority order:

1. Explainability
2. Intuition
3. Interactivity
4. Beauty
5. Faithfulness to the core SnapKV idea
6. Only then deeper implementation detail

## Summary

This notebook should not read like:

> "Here is a paper and here are its results."

It should read like:

> "Here is a hard memory problem in LLMs, here is why naive solutions fail, and here is a clever idea called SnapKV that feels surprisingly human once you see it."

That is the tone and direction for the project.
