import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.visualizations import (
        plot_adaptive_window,
        plot_attention_compute,
        plot_attention_consistency,
        plot_budget_quality,
        plot_entropy_intuition,
        plot_human_memory,
        plot_kv_growth,
        plot_memory_breakdown,
        plot_naive_strategy,
        plot_per_head_heatmap,
        plot_vote_cluster,
        render_algo_step,
        run_demo,
    )

    return (
        plot_adaptive_window,
        plot_attention_compute,
        plot_attention_consistency,
        plot_budget_quality,
        plot_entropy_intuition,
        plot_human_memory,
        plot_kv_growth,
        plot_memory_breakdown,
        plot_naive_strategy,
        plot_per_head_heatmap,
        plot_vote_cluster,
        render_algo_step,
        run_demo,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What Should an LLM Remember?

    ### An interactive exploration of SnapKV · NeurIPS 2024

    ---

    Think about the last long conversation you had.

    When you replied, you didn't re-read every message from the beginning. You held a working
    memory of what mattered — recent things, repeated things, things relevant to what
    you were about to say.

    Large language models face exactly this problem, at scale. Every token they generate
    depends on all previous tokens. As conversations grow longer, so does the memory
    they must carry — and the cost of carrying it.

    **SnapKV** is a 2024 paper from UIUC, Cohere, and Princeton that solves this with a
    surprisingly human-like insight:

    > *The model already knows which tokens matter before it starts generating.*

    In this notebook you'll build intuition for why that's true, how SnapKV exploits it,
    and interactively explore what happens when you compress the cache.

    <small>Li et al., "SnapKV: LLM Knows What You Are Looking for Before Generation", NeurIPS 2024.
    [arxiv.org/abs/2404.14469](https://arxiv.org/abs/2404.14469)</small>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1 · Why LLMs Need Memory

    LLMs generate text one token at a time. To generate token $t$, the model runs
    **attention** over all previous tokens — it looks back at everything to decide
    what comes next.

    This is called **autoregressive generation**. It's powerful but expensive:
    the cost of generating each new token scales with the length of everything
    that came before it.

    The naive fix is to cache the intermediate representations so we don't
    recompute them. That's the **KV cache** — a store of Key and Value vectors
    for every past token, reused at each generation step.

    But here's the catch:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The memory equation, visualised

    The KV cache for one token is the product of six factors:

    $$\text{bytes/token} \;=\; \underbrace{2}_{K+V} \cdot L \cdot H \cdot d_{\text{head}} \cdot \text{bytes}_{\text{dtype}}$$

    Drag the sliders below to see how each factor compounds. The chart's y-axis is
    **logarithmic** — small bumps in any factor blow up the total.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    layers_s   = mo.ui.slider(start=8,  stop=80,  step=4,  value=32,
                              label="Layers (L)")
    heads_s    = mo.ui.slider(start=8,  stop=64,  step=4,  value=32,
                              label="Attention heads (H)")
    headdim_s  = mo.ui.slider(start=32, stop=256, step=32, value=128,
                              label="Head dimension (d)")
    seqmem_s   = mo.ui.slider(start=1024, stop=131072, step=1024, value=8192,
                              label="Sequence length (tokens)")
    dtype_s    = mo.ui.radio(options={"fp16 (2B)": 2, "fp32 (4B)": 4, "int8 (1B)": 1},
                             value="fp16 (2B)", label="Data type", inline=True)
    mo.vstack([
        mo.md("**Try changing these values:**"),
        layers_s, heads_s, headdim_s, seqmem_s, dtype_s,
    ])
    return dtype_s, headdim_s, heads_s, layers_s, seqmem_s


@app.cell
def _(dtype_s, headdim_s, heads_s, layers_s, plot_memory_breakdown, seqmem_s):
    plot_memory_breakdown(
        n_layers=layers_s.value,
        n_heads=heads_s.value,
        head_dim=headdim_s.value,
        seq_len=seqmem_s.value,
        dtype_bytes=dtype_s.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the default 7B-style config (32 / 32 / 128 / fp16): **~0.5 MB per token**.
    At 32K tokens that's **~16 GB**. At 128K (GPT-4 context) it's **~64 GB** —
    just for the KV cache, before the model weights themselves.

    This is the bottleneck that kills long-context inference at scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2 · What is KV Cache?

    Transformers compute attention using three projections of each token: **Query (Q)**,
    **Key (K)**, and **Value (V)**.

    When generating token $t$:
    1. Compute $Q_t$ from the current token
    2. Compare $Q_t$ against all previous $K_i$ to get attention weights
    3. Use the weights to combine all previous $V_i$

    Without caching, steps 2 and 3 require recomputing $K_i$ and $V_i$ for every
    previous token at every generation step. That's $O(T^2)$ work per step.

    **The KV cache** stores each $K_i$ and $V_i$ after it's first computed and
    reuses them — reducing work to $O(T)$ per step.

    The chart below makes that gap concrete: as $T$ grows, the red curve
    (no cache) explodes while the green curve (with cache) grows linearly.
    """)
    return


@app.cell
def _(plot_attention_compute):
    plot_attention_compute()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The trade is: we save *time* but spend *memory*. The cache grows by one row
    (one K, one V) per generated token, multiplied across every layer and every head.
    That's exactly what the next section shows.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3 · The Problem: Memory Grows Linearly

    Below is the KV cache memory footprint at different context lengths,
    comparing the full cache against the compressed approaches we'll explore.
    """)
    return


@app.cell
def _(plot_kv_growth):
    plot_kv_growth()
    return


@app.cell(hide_code=True)
def _(mo):
    seq_len_slider = mo.ui.slider(
        start=1024, stop=32768, step=1024, value=8192,
        label="Sequence length (tokens)"
    )
    mo.vstack([
        mo.md("Zoom in on a specific context length:"),
        seq_len_slider,
    ])
    return (seq_len_slider,)


@app.cell
def _(mo, plot_kv_growth, seq_len_slider):
    s = seq_len_slider.value
    n_layers, n_heads, head_dim, dtype_bytes = 32, 32, 128, 2
    full_mb = 2 * n_layers * n_heads * s * head_dim * dtype_bytes / 1e6
    snap_mb = full_mb * 0.25
    mo.vstack([
        plot_kv_growth(seq_lengths=[s]),
        mo.md(f"""
        At **{s:,} tokens**: full cache = **{full_mb:.0f} MB** ·
        SnapKV (25% budget) = **{snap_mb:.0f} MB**
        · savings = **{full_mb - snap_mb:.0f} MB**
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4 · A Game: What Would You Keep?

    Suppose you must throw away some tokens to fit within a memory budget.
    You don't yet know exactly what the model will need to generate next.

    What would you keep?

    Here are the natural strategies people reach for first.
    Each one has a fatal flaw.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    naive_strategy = mo.ui.radio(
        options=["Keep Everything", "Recent Only", "Random Drop", "Uniform Stride"],
        value="Keep Everything",
        label="Eviction strategy",
        inline=True,
    )
    naive_budget = mo.ui.slider(
        start=0.1, stop=0.9, step=0.05, value=0.35,
        label="Cache budget (fraction of tokens to keep)"
    )
    mo.vstack([
        mo.md("**Choose a strategy and a budget:**"),
        naive_strategy,
        naive_budget,
    ])
    return naive_budget, naive_strategy


@app.cell
def _(naive_budget, naive_strategy, plot_naive_strategy):
    plot_naive_strategy(naive_strategy.value, n_tokens=40, budget_pct=naive_budget.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    None of these strategies use the **current intent** to decide what to keep.
    They're all static — they don't know what the model is about to do.

    That's the missing ingredient.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 5 · How Humans Handle This

    When you're about to answer a specific question, your brain doesn't
    keep all past experiences equally accessible. It prioritizes based on:

    - **Recency** — things that happened recently are easier to recall
    - **Repetition** — things that came up multiple times stick around
    - **Goal relevance** — things relevant to what you're doing *right now*

    SnapKV mirrors the third factor especially well.

    Here's a concrete example. You've been thinking about a trip and running gear
    throughout the day. Now you're about to answer: *"Which running shoes fit my
    trip under budget?"*

    Which memories should you keep?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Try changing these weights to see how the importance ranking shifts:**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    recent_w  = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.35, label="Recency weight")
    repeat_w  = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.25, label="Repetition weight")
    goal_w    = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.40, label="Goal relevance weight")
    topk_mem  = mo.ui.slider(start=1, stop=7, step=1, value=3, label="Memories to keep (top-k)")
    mo.vstack([recent_w, repeat_w, goal_w, topk_mem])
    return goal_w, recent_w, repeat_w, topk_mem


@app.cell
def _(goal_w, plot_human_memory, recent_w, repeat_w, topk_mem):
    plot_human_memory(
        recent_weight=recent_w.value,
        repeat_weight=repeat_w.value,
        goal_weight=goal_w.value,
        top_k=topk_mem.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that when **goal relevance** is high, the chart naturally selects
    "checked running routes", "opened shoe size chart", and "searched: best running shoes" —
    exactly the events relevant to the current question.

    **This is the core intuition behind SnapKV.**

    The model's current query — the last part of the prompt, just before it starts
    generating — is the signal we should use to decide what to keep.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 6 · SnapKV's Key Insight

    Here is the central observation from the paper:

    > **The end of the prompt often reveals what the model will need next.**

    Before the model generates a single token, it runs a full "prefill" pass over
    the entire prompt. During that pass, attention patterns are computed for every
    token position.

    SnapKV's authors noticed that the attention pattern in the **last few tokens
    of the prompt** (the "observation window") is highly consistent with the
    attention pattern throughout subsequent generation.

    In other words: the model already "decided" what it cares about during prefill —
    before generation starts. We can use those decisions to compress the cache.

    Instead of keeping every KV entry, we can:
    1. Look at which tokens the observation window attends to
    2. Keep only those (plus the window itself)
    3. Decode normally with the compressed cache

    The attention pattern is **consistent** — the selections made during prefill
    remain valid throughout the generation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Attention consistency: observation window vs full-sequence attention
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    obs_window_slider = mo.ui.slider(
        start=4, stop=64, step=4, value=16,
        label="Observation window size (w)"
    )
    head_selector = mo.ui.dropdown(
        options={"Head 0": 0, "Head 1": 1, "Head 2": 2, "Head 3": 3},
        value="Head 0",
        label="Attention head",
    )
    mo.vstack([
        mo.md("The dashed line shows what SnapKV *predicts* from the observation window. "
              "The solid line is the true full-sequence attention. Try different window sizes."),
        obs_window_slider,
        head_selector,
    ])
    return head_selector, obs_window_slider


@app.cell
def _(head_selector, obs_window_slider, plot_attention_consistency):
    plot_attention_consistency(obs_window_slider.value, head_selector.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even with a small observation window, the prediction closely tracks the true
    attention pattern. The high-importance tokens (the "heavy hitters") are
    consistently identified — which is all we need to compress the cache faithfully.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 7 · How SnapKV Works

    The algorithm runs in two stages after the prefill pass:

    **Stage 1 — Vote:** For each attention head, use the observation window
    (last $w$ tokens of the prompt) to score every earlier token by how much
    attention it received. Select the top-$k$ tokens per head.

    **Stage 2 — Cluster:** Don't just keep the exact winners. Keep their
    neighbors too, using a max-pooling kernel of size $k_{pool}$. This preserves
    local context around important tokens and avoids fragmentation artifacts.

    The result: a compressed cache of size $k + w$ (selected prefix + observation
    window) used for all subsequent generation.

    Step through it below:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    algo_step = mo.ui.slider(
        start=1, stop=4, step=1, value=1,
        label="Algorithm step (drag or click)"
    )
    algo_step
    return (algo_step,)


@app.cell
def _(algo_step, render_algo_step):
    render_algo_step(algo_step.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Watch it run on a toy sequence

    Below: a 32-token prefix gets scored by a 6-token observation window.
    Purple bars are the raw votes. The orange line is what max-pooling does to
    them — it spreads each peak over its neighbours so we keep local context,
    not just isolated winners. Green bars are the tokens SnapKV ultimately keeps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pool_kernel_s = mo.ui.slider(start=1, stop=9, step=2, value=3,
                                 label="Pooling kernel size (kₚₒₒₗ)")
    toy_budget_s  = mo.ui.slider(start=4, stop=20, step=1, value=10,
                                 label="Cache budget (k + w)")
    mo.vstack([pool_kernel_s, toy_budget_s])
    return pool_kernel_s, toy_budget_s


@app.cell
def _(plot_vote_cluster, pool_kernel_s, toy_budget_s):
    plot_vote_cluster(
        seq_len=32,
        window_size=6,
        budget=toy_budget_s.value,
        kernel_size=pool_kernel_s.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pseudocode

    ```python
    def snapkv_compress(keys, values, window_size=16, budget=256, kernel_size=5):
        prefix_keys,   window_keys   = keys[:-window_size],   keys[-window_size:]
        prefix_values, window_values = values[:-window_size], values[-window_size:]

        # Stage 1 — Vote
        scale = head_dim ** -0.5
        attn  = softmax(window_keys @ prefix_keys.T * scale)  # (w, S-w)
        scores = attn.mean(axis=0)                             # (S-w,)

        # Stage 2 — Cluster
        scores = max_pool1d(scores, kernel_size)               # smooth
        top_k_indices = argsort(scores)[-budget:]

        compressed_keys   = concat([prefix_keys[top_k_indices],   window_keys])
        compressed_values = concat([prefix_values[top_k_indices], window_values])
        return compressed_keys, compressed_values
    ```

    The code in `src/snapkv.py` implements this exactly, operating on tensors
    of shape `(batch, heads, seq_len, head_dim)`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 8 · Memory DJ: Live Demo

    Paste any prompt below. Choose an eviction method and a cache budget.
    Watch which tokens each method keeps — and which it throws away.

    The **observation window** (purple) is always kept by SnapKV and used
    to vote on the prefix. **Kept** tokens (green) survive compression.
    **Evicted** tokens (coral) are dropped.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    prompt_input = mo.ui.text_area(
        value=(
            "The Eiffel Tower is 324 metres tall, about the same height as an 81-storey building "
            "and the tallest structure in Paris. Its base is square, measuring 125 metres on each "
            "side. During its construction, the Eiffel Tower surpassed the Washington Monument to "
            "become the tallest man-made structure in the world, a title it held for 41 years "
            "until the Chrysler Building in New York City was finished in 1930. "
            "What is the height of the Eiffel Tower?"
        ),
        label="Input prompt",
        rows=5,
        full_width=True,
    )
    budget_slider = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.3,
        label="Cache budget (fraction of tokens to keep)"
    )
    method_select = mo.ui.radio(
        options=["SnapKV", "H2O", "StreamingLLM", "Full Cache"],
        value="SnapKV",
        label="Eviction method",
        inline=True,
    )
    mo.vstack([prompt_input, budget_slider, method_select])
    return budget_slider, method_select, prompt_input


@app.cell
def _(budget_slider, method_select, prompt_input, run_demo):
    run_demo(prompt_input.value, budget_slider.value, method_select.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try the **needle-in-a-haystack** pattern: put a key fact early in the prompt,
    fill the middle with unrelated text, and put the question at the end.

    SnapKV should keep the needle because the question (observation window) attends
    to it. "Recent Only" will drop it. "Random Drop" is unpredictable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 9 · Per-Head Specialization

    SnapKV operates **independently per attention head** — each head keeps the
    tokens *it* cares about, not a global average. This matters because heads
    specialize: some focus on recent tokens (positional), some on semantically
    important tokens, some on syntax.

    The chart below shows simulated per-head attention over the tokens in your
    prompt from the demo above.
    """)
    return


@app.cell
def _(plot_per_head_heatmap, prompt_input):
    plot_per_head_heatmap(prompt_input.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Heads 0–1 (positional) have smooth decay curves — they weight recent tokens heavily.
    Heads 2–3 (semantic) show sparse spikes — they find a few high-value tokens and
    attend to them almost exclusively. Heads 4–7 are mixed.

    Per-head budgets mean each head gets to keep what it specialises in,
    rather than being forced to share a single global selection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 10 · Our Extension: Adaptive Observation Window

    The original SnapKV paper uses a **fixed** observation window size $w$ across
    all heads. But is that optimal?

    **Our extension:** measure **attention entropy** across heads to adaptively
    size the observation window.

    - **High-entropy head** (diffuse attention, uncertain what matters)
      → benefit from a *larger* observation window
    - **Low-entropy head** (focused attention, confident about what matters)
      → a *small* window is enough

    This mirrors Ada-KV's per-head budget idea but applies it to the
    *observation stage* rather than the cache budget.

    The key equation:

    $$w_h = \text{clip}\left(w_{\min} + (w_{\max} - w_{\min}) \cdot \frac{H(a_h)}{H_{\max}}, \; w_{\min}, \; w_{\max}\right)$$

    where $H(a_h)$ is the Shannon entropy of head $h$'s attention distribution
    and $H_{\max} = \log(S)$ is the maximum possible entropy.

    **What does entropy look like?** Two heads from the same model, side by side:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    focus_temp_s   = mo.ui.slider(start=0.2, stop=1.5, step=0.1, value=0.4,
                                  label="Focused-head sharpness (low → spikier)")
    diffuse_temp_s = mo.ui.slider(start=1.5, stop=8.0, step=0.5, value=4.0,
                                  label="Diffuse-head sharpness (high → flatter)")
    mo.vstack([focus_temp_s, diffuse_temp_s])
    return diffuse_temp_s, focus_temp_s


@app.cell
def _(diffuse_temp_s, focus_temp_s, plot_entropy_intuition):
    plot_entropy_intuition(
        focus_temp=focus_temp_s.value,
        diffuse_temp=diffuse_temp_s.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The green head's attention concentrates on a handful of positions — its
    entropy is far below the maximum, and a small observation window already
    sees what matters. The orange head spreads weight everywhere — high entropy,
    needs a wider window to capture the right tokens.

    Now apply that idea per head across the model:
    """)
    return


@app.cell
def _(plot_adaptive_window, prompt_input):
    plot_adaptive_window(prompt_input.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Focused heads (low entropy, left side of the entropy line) use a window as
    small as 8 tokens. Diffuse heads (high entropy) scale up to 32 tokens.

    In practice this yields a **smaller total observation cost** than using $w=32$
    uniformly, while maintaining the same prediction quality for focused heads.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 11 · Budget vs Quality Tradeoff

    How much quality do we lose as we compress more aggressively?
    The chart below shows the qualitative shape of results on LongBench —
    a suite of long-context comprehension tasks.
    """)
    return


@app.cell
def _(plot_budget_quality):
    plot_budget_quality()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Key observations:

    - **Full cache** is the ceiling — but it's unaffordable at long context
    - **StreamingLLM** (sink + recent) degrades sharply below 30% budget
    - **H2O** (cumulative attention) is competitive but starts losing ground below 20%
    - **SnapKV** maintains near-full quality down to ~15–20% budget, then drops off gracefully

    The gap widens as context grows: SnapKV's advantage is most pronounced
    on the longest inputs, precisely where cache compression matters most.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    | Method | Core idea | Budget type | Long-context quality |
    |---|---|---|---|
    | Full cache | Keep everything | 100% | Baseline |
    | StreamingLLM | Sink tokens + recent window | Fixed | Degrades |
    | H2O | Greedy top-k by cumulative attention | Fixed | Good |
    | **SnapKV** | **Obs. window voting + clustering** | **Fixed per head** | **Best** |
    | *Adaptive SnapKV (ours)* | *Entropy-guided window sizing* | *Dynamic per head* | *≥ SnapKV* |

    ---

    ### The one-sentence version

    > SnapKV works because the model's **current intent** — expressed in the last
    > few tokens of the prompt — reliably predicts which past tokens it will need
    > during generation. So we look there first, and throw the rest away.

    ---

    **Paper:** Li et al., "SnapKV: LLM Knows What You Are Looking for Before Generation",
    NeurIPS 2024. [arxiv.org/abs/2404.14469](https://arxiv.org/abs/2404.14469)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
