# SnapKV: Interactive Notebook

**molab x alphaXiv Competition Submission**

An interactive marimo notebook that brings [SnapKV (NeurIPS 2024)](https://arxiv.org/abs/2404.14469) to life — from the core insight to a live token-level demo and a novel adaptive extension.

## Paper

> **SnapKV: LLM Knows What You Are Looking for Before Generation**  
> Yuhong Li et al. · UIUC + Cohere + Princeton · NeurIPS 2024  
> https://arxiv.org/abs/2404.14469

## What's in the notebook

| Section | What you'll see |
|---|---|
| 1 · The problem | KV cache memory growth across methods |
| 2 · The insight | Attention consistency between obs. window and full sequence |
| 3 · The algorithm | Interactive step-through of voting + clustering |
| 4 · Live demo | Type a prompt — watch tokens get kept/evicted per method |
| 5 · Head-level analysis | Per-head attention specialization heatmap |
| 6 · Extension | Adaptive observation window via attention entropy |
| 7 · Budget vs quality | SnapKV vs H2O vs StreamingLLM tradeoff curves |

## Our extension

The original paper uses a fixed observation window size `w` across all heads. We propose **entropy-guided adaptive windowing**: heads with high attention entropy (diffuse, uncertain attention) receive a larger observation window, while focused heads need fewer tokens. This mirrors Ada-KV's per-head budget idea but applies it to the *observation stage* rather than the cache budget.

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/snapkv-notebook
cd snapkv-notebook
pip install -r requirements.txt
marimo edit notebooks/walkthrough.py
```
