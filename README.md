# What Should an LLM Remember?

**molab × alphaXiv Competition Submission**

An interactive marimo notebook that turns [SnapKV (NeurIPS 2024)](https://arxiv.org/abs/2404.14469) into a story — from the first intuition about memory all the way to agentic systems, with six interactive games along the way.

## Paper

> **SnapKV: LLM Knows What You Are Looking for Before Generation**  
> Yuhong Li et al. · UIUC + Cohere + Princeton · NeurIPS 2024  
> https://arxiv.org/abs/2404.14469

## What's in the notebook

| # | Section | What you'll see |
|---|---|---|
| 1 | Why LLMs Need Memory | Autoregressive generation intuition; interactive KV cache memory equation |
| 2 | What is KV Cache? | Library analogy for K/V/Q; O(T²) vs O(T) compute chart |
| 3 | The Problem: Memory Grows Linearly | Memory footprint across methods; why linear growth breaks GPU serving |
| **G1** | **Game: What Would You Keep?** | **Try naive eviction strategies — each one has a fatal flaw** |
| 5 | How Humans Handle This | Editable memory-triage chart: recency vs repetition vs goal relevance |
| 6 | SnapKV's Key Insight | Attention consistency; Figure 2 embedded from the original paper |
| 7 | How SnapKV Works | 4-step algorithm walkthrough; interactive vote + cluster pipeline |
| **G2** | **Game: Memory DJ** | **Paste any prompt, pick a method and budget, watch token selection live** |
| **G3** | **Game: Needle in a Haystack** | **Hide a fact at early/middle/late position; see which methods preserve it** |
| 9 | Per-Head Specialization | Per-head attention heatmap; why global budgets destroy head diversity |
| 10 | Our Extension | Entropy-guided adaptive observation window per head |
| 11 | Budget vs Quality | LongBench-style tradeoff curves across methods |
| **G4** | **Game: Build Your Own Policy** | **Mix recency + frequency + attention weights; converge on SnapKV yourself** |
| 12 | Competitors | Method comparison table and capability matrix |
| **G5** | **Game: Method Picker** | **Answer a few questions about your use-case; get a recommendation** |
| 13 | The Bigger Picture | Sparse attention vs KV eviction — two orthogonal axes |
| 14 | From KV Cache to Agentic Memory | Hot/warm/cold memory hierarchy; SnapKV concepts mapped to agent turns |
| **G6** | **Game: Agent Memory Over Turns** | **Simulate a multi-turn agent loop under different memory strategies** |

## Our extension

The original paper uses a fixed observation window size `w` across all heads. We propose **entropy-guided adaptive windowing**: each head's observation window is sized proportionally to its attention entropy.

- High-entropy heads (diffuse, uncertain attention) get a **larger** window — they need more tokens before their selection stabilises.
- Low-entropy heads (focused, specialised attention) get a **smaller** window — a few tokens already tell you what they care about.

This applies Ada-KV's per-head adaptation idea one stage earlier — to the *observation window* rather than the cache budget — and requires no additional parameters or training.

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/snapkv-notebook
cd snapkv-notebook
pip install -r requirements.txt
marimo edit notebooks/walkthrough.py
```
