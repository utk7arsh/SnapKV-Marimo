# Competition Guidelines

**Bring Research to Life: molab Notebook Competition**
alphaXiv × marimo

## Challenge

Turn a research result into something tangible, reproducible, and interactive — a marimo notebook.

Implement a key contribution from a curated AlphaXiv paper with code, UI elements, and explanatory text. Others who work through the notebook should walk away with an intuitive understanding of the paper's core contribution and how to apply it.

Bonus points for adding your own spin: an extension or variant of the algorithm, or applying it to different datasets.

**Constraint: CPU compute only.** No GPUs available in molab.

## Judging Criteria

Reviewed by 4 panelists (2 from alphaXiv, 2 from marimo). The jury favors:

1. **Intuitive understanding** — core idea explained through code, UI, and text
2. **Custom extensions** — improve performance or provide additional insight
3. **Interactivity** — the reader can experiment, not just read

## Prizes

| Place | Prize |
|-------|-------|
| 1st | Mac Mini + $500 gift cards + marimo swag |
| 2nd | $500 gift cards + marimo swag |
| 3rd | $250 gift cards + marimo swag |

All places receive shoutouts on alphaXiv & marimo socials.

## Submission

Submit a molab link. Individual or team submissions welcome.
Up to 3 entries per person/team. Each entry can be from a different team.

## Our Paper

**SnapKV: LLM Knows What You Are Looking for Before Generation**
- Yuhong Li et al. · UIUC + Cohere + Princeton · NeurIPS 2024
- https://arxiv.org/abs/2404.14469

## Our Extension

**Entropy-guided adaptive observation window.**

The original SnapKV uses a fixed window size `w` for all attention heads. We propose
sizing the window per-head based on attention entropy:
- High-entropy heads (diffuse, uncertain) → larger observation window
- Low-entropy heads (focused, specialized) → smaller window

This mirrors Ada-KV's per-head budget idea but applies it to the *observation stage*
rather than the cache budget itself.
