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
        plot_agent_mapping,
        plot_attention_compute,
        plot_attention_consistency,
        plot_budget_quality,
        plot_capability_matrix,
        plot_competitors_table,
        plot_entropy_intuition,
        plot_human_memory,
        plot_kv_growth,
        plot_memory_breakdown,
        plot_memory_hierarchy,
        plot_memory_types,
        plot_naive_strategy,
        plot_per_head_heatmap,
        plot_two_axes,
        plot_vote_cluster,
        render_algo_step,
        run_custom_policy,
        run_demo,
        run_method_picker,
        run_needle_demo,
        simulate_agent_loop,
    )

    return (
        plot_adaptive_window,
        plot_agent_mapping,
        plot_attention_compute,
        plot_attention_consistency,
        plot_budget_quality,
        plot_capability_matrix,
        plot_competitors_table,
        plot_entropy_intuition,
        plot_human_memory,
        plot_kv_growth,
        plot_memory_breakdown,
        plot_memory_hierarchy,
        plot_memory_types,
        plot_naive_strategy,
        plot_per_head_heatmap,
        plot_two_axes,
        plot_vote_cluster,
        render_algo_step,
        run_custom_policy,
        run_demo,
        run_method_picker,
        run_needle_demo,
        simulate_agent_loop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What Should an LLM Remember?

    ### An interactive exploration of SnapKV: LLM Knows What You Are Looking for Before Generation · NeurIPS 2024

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
    ### Intuition first: why does generation need memory at all?

    Suppose an LLM is writing the word after *"The Eiffel Tower was built in"*.
    To pick the right word — *1889* — it needs to have "read" the earlier tokens
    and formed a sense of what the sentence is about.

    Now imagine doing that for every single word in a long reply, one at a time.
    Each new word requires looking back at everything written so far. That's
    what **autoregressive generation** means: the model generates one token,
    appends it, then generates the next — always attending to the full history.

    Without any optimisation, this means recomputing the same representations
    over and over. The **KV cache** avoids that by storing the result of each
    token's computation the first time and reusing it on every subsequent step.
    It trades computation for memory — and as context grows longer, the memory
    bill compounds fast.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Intuition: the library analogy

    Think of the KV cache as a reference library the model builds while reading
    your prompt.

    - **Keys** are like the index cards in that library — a compact summary of
      what each past token is "about", used to decide if it's relevant.
    - **Values** are the actual content on those cards — the rich representation
      the model reads once it decides a token matters.
    - **Queries** are the search terms the model issues for the current token —
      "what do I need from my history to predict what comes next?"

    At each generation step the model issues a query, scans every key in the
    library to compute a relevance score, then blends the corresponding values
    proportionally. The library never shrinks — every generated token adds a new
    card — which is exactly why memory grows linearly with context length.
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
    ### Why does linear growth hurt so much?

    "Linear" sounds manageable — but the constant factor is brutal. Each token's
    KV entry is multiplied across every layer and every head before it even reaches
    the sequence length. On a standard 7B model that's ×32 layers × 32 heads, so
    a single new token adds roughly **0.5 MB** to the cache.

    The practical consequences stack up fast:

    - **GPU memory ceiling.** A high-end A100 has 80 GB of VRAM. At 32K tokens
      the KV cache alone consumes 16 GB — 20% of the card — leaving less room
      for model weights and activations. At 128K tokens it needs 64 GB, nearly
      the entire card just for cache.
    - **Batch size collapses.** Serving many users at once means fitting multiple
      KV caches in parallel. As each grows, fewer fit, which means lower
      throughput and higher cost per query — a direct hit to serving economics.
    - **Memory bandwidth pressure.** Even when a cache fits, *reading* it takes
      time. Every decode step streams the full cache through the GPU's memory
      bus, so a larger cache means slower tokens-per-second regardless of
      compute.

    The result: without compression, long-context models are either very slow,
    very expensive, or both. That is the problem SnapKV was built to solve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## GAME 1 · What Would You Keep?

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
    mo.Html("""
    <div style="margin:16px 0">
      <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
                  color:var(--color-text-secondary);margin-bottom:8px">
        Figure 2 · directly from the paper
      </div>
      <img src="https://arxiv.org/html/2404.14469v2/x2.png"
           alt="Figure 2 from the SnapKV paper: overlap rates for observation windows within prompt tokens, showing that the last window achieves ~70–100% overlap with generation-time attention across all layers"
           style="max-width:100%;border-radius:8px;border:1px solid var(--color-border, #e5e5e5)" />
      <div style="font-size:12px;color:var(--color-text-secondary);margin-top:8px;line-height:1.6">
        Each line is a transformer layer. The x-axis is which part of the prompt the
        observation window was taken from; the y-axis is how much its attention selection
        overlaps with what the model actually attends to during generation. The last window
        (right edge) consistently hits <strong>70–100% overlap across all 32 layers</strong>
        — the empirical foundation for SnapKV's core assumption.
      </div>
    </div>
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
    ## GAME 2 · Live Demo — Memory DJ

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

    GAME 3 below makes that contrast explicit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## GAME 3 · Needle in a Haystack

    A long prompt with one key fact buried inside, plus a question at the end.
    Pick where to hide the needle and how tight your cache budget is — then
    watch which policies actually preserve it.

    SnapKV's observation window picks up the question's intent, so it tends to
    pull the needle in. Recent-only and random can't.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    needle_pos = mo.ui.radio(
        options=["early", "middle", "late"],
        value="middle",
        label="Where is the needle hidden?",
        inline=True,
    )
    needle_budget = mo.ui.slider(
        start=0.10, stop=0.60, step=0.05, value=0.25,
        label="Cache budget"
    )
    haystack_n = mo.ui.slider(
        start=30, stop=100, step=10, value=60,
        label="Haystack size (filler tokens)"
    )
    mo.vstack([needle_pos, needle_budget, haystack_n])
    return haystack_n, needle_budget, needle_pos


@app.cell
def _(haystack_n, needle_budget, needle_pos, run_needle_demo):
    run_needle_demo(
        needle_position=needle_pos.value,
        budget=needle_budget.value,
        haystack_size=haystack_n.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Move the needle to "early" with a tight budget — Recent Only will fail
    immediately. SnapKV usually keeps it, because the question keywords at
    the end light up the needle's position via window attention.
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
    ### Reading the chart

    **Heads 0–1 (positional)** show a smooth exponential decay from right to left —
    they heavily weight the most recent tokens regardless of content. These heads
    track things like sentence position and local syntactic dependencies.

    **Heads 2–3 (semantic)** show sparse spikes on a handful of positions. These
    heads identify a few high-value tokens — often named entities, key verbs, or
    the topic of the sentence — and concentrate nearly all their attention there.
    They are less interested in position and more interested in *meaning*.

    **Heads 4–7** are mixed: some recency bias, some semantic spikes, general-purpose.

    ### Why this matters for compression

    Head specialisation is not just an interesting observation — it directly shapes
    how SnapKV should compress the cache.

    If you were to run a single global top-k selection across all heads, the
    "winners" would inevitably be the tokens every head agrees on: the very first
    token (a universal attention sink), the most recent tokens, and perhaps a
    few prominent nouns. Semantic heads would have to accept whatever the positional
    heads voted for, and vice versa.

    By selecting top-k **per head**, each head gets its own private budget. The
    positional head keeps recent tokens. The semantic head keeps its sparse set of
    high-value tokens. Neither cannibalises the other. The result is a compressed
    cache that preserves the full *diversity* of the model's attention behaviour,
    not just the globally popular tokens.

    This is also why naive baselines like "keep the most-attended tokens overall"
    underperform — they collapse that diversity into a single shared ranking, and
    heads that specialise in rare but critical information lose out.
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
    ## GAME 4 · Live Demo — Build Your Own Memory Policy

    You've now seen recency, frequency, and the SnapKV-style window-attention
    signal. Combine them yourself and see how close you can get to SnapKV
    with a hand-tuned formula:

    $$\text{score}(t) \;=\; w_r \cdot \text{recency}(t) \;+\; w_f \cdot \text{frequency}(t) \;+\; w_a \cdot \text{attention}(t)$$

    Then keep the top-budget tokens. Compare your selection to SnapKV's on
    the same prompt.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    custom_recency = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.30,
                                  label="Recency weight  (wᵣ)")
    custom_frequency = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.20,
                                    label="Frequency weight (w_f)")
    custom_attention = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.50,
                                    label="Attention weight (wₐ)")
    custom_budget = mo.ui.slider(start=0.10, stop=0.80, step=0.05, value=0.30,
                                 label="Cache budget")
    mo.vstack([custom_recency, custom_frequency, custom_attention, custom_budget])
    return custom_attention, custom_budget, custom_frequency, custom_recency


@app.cell
def _(
    custom_attention,
    custom_budget,
    custom_frequency,
    custom_recency,
    prompt_input,
    run_custom_policy,
):
    run_custom_policy(
        prompt=prompt_input.value,
        budget=custom_budget.value,
        recency_w=custom_recency.value,
        frequency_w=custom_frequency.value,
        attention_w=custom_attention.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two ways to play:

    - **Reverse-engineer SnapKV.** Set attention = 1, others = 0. Watch the
      overlap with SnapKV jump to ~100%.
    - **Reverse-engineer StreamingLLM.** Set recency = 1, others = 0. Watch
      it collapse onto the tail of the prompt and miss any earlier needle.

    Mixed weights interpolate between these extremes — that's the whole
    eviction-policy design space, in three sliders.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 12 · Competitors

    SnapKV isn't alone — it sits in a fast-moving family of KV-cache
    compression methods. Here are the nine most directly comparable papers,
    with the **2025 wave** marked: D2O, CAKE, and SCOPE all push into
    per-layer dynamic budgets and recoverability — territory that was mostly
    empty when SnapKV first landed.
    """)
    return


@app.cell
def _(plot_competitors_table):
    plot_competitors_table()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What each one actually does differently

    Mechanisms read similarly in prose, so here they are as a capability
    matrix. Five axes — per-head budget, per-layer budget, adaptive sizing,
    decode-aware behaviour, and the ability to recover evicted entries.
    """)
    return


@app.cell
def _(plot_capability_matrix):
    plot_capability_matrix()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A few patterns worth pausing on:

    - **SnapKV** owns the per-head column from 2024 — that's its central
      contribution. It doesn't push per-layer or recovery, which is
      exactly where the 2025 work has gone.
    - **PyramidKV** was the first to seriously vary budget per layer; D2O
      and CAKE refined the idea with dynamic and cascading variants.
    - **Recovery** (resurrecting evicted entries from a summary or merge)
      barely existed before D2O. It's the cleanest systems-style answer
      to "what if the obs-window vote was wrong?"
    - **SCOPE** is the odd one out — it doesn't push per-layer or per-head,
      it splits compression by *phase* (prefill vs decode), which matters
      most for long-output reasoning chains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GAME 5 · Method Picker

    Pick a scenario and the picker scores all nine methods against it,
    using the fitness profiles encoded in [`src/visualizations.py`](src/visualizations.py).
    No ML — just transparent ranking on top of the capability matrix above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    pick_context = mo.ui.dropdown(
        options=["Short (< 8K)", "Medium (8–32K)", "Long (32–128K)", "Very long (> 128K)"],
        value="Long (32–128K)",
        label="Context length"
    )
    pick_workload = mo.ui.radio(
        options=["Single-shot Q&A", "Streaming chat", "Long generation / reasoning"],
        value="Single-shot Q&A",
        label="Workload type"
    )
    pick_dropin = mo.ui.checkbox(value=True, label="Drop-in required (no model surgery)")
    pick_recovery = mo.ui.checkbox(value=False, label="Need to recover evicted info")
    mo.vstack([pick_context, pick_workload, pick_dropin, pick_recovery])
    return pick_context, pick_dropin, pick_recovery, pick_workload


@app.cell
def _(
    pick_context,
    pick_dropin,
    pick_recovery,
    pick_workload,
    run_method_picker,
):
    run_method_picker(
        context_length=pick_context.value,
        workload=pick_workload.value,
        drop_in_required=pick_dropin.value,
        recovery_needed=pick_recovery.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try a few scenarios:

    - **Drop-in required + Single-shot Q&A on 32–128K context** → SnapKV
      tends to win. It's the cleanest plug-and-play option in this regime.
    - **Long generation + recovery needed** → D2O or SCOPE jump to the top —
      they're built for long decode and the 2025 wave's recovery mechanisms.
    - **Streaming chat** → StreamingLLM is hard to beat for that exact
      workload; it was designed for it.

    The picker has no ground truth — it's a transparent scoring function
    over the capability matrix above. Useful as an orientation tool, not
    a benchmark.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 13 · The Bigger Picture: Two Axes

    SnapKV is one move on a board with two axes. People tackling long-context
    inference usually pick one — or, increasingly, both at once.
    """)
    return


@app.cell
def _(plot_two_axes):
    plot_two_axes()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Where does this go next? Two natural directions, both in active research:

    - **Token-level memory beyond eviction.** Instead of throwing entries
      away, *compress* them — merge similar K/V pairs, summarise blocks,
      or learn a small recurrent state that absorbs evicted tokens.
    - **Layer- and head-aware budgets.** Not every layer needs the same
      cache size; not every head deserves an equal share. Methods like
      Ada-KV and PyramidKV start to vary the budget across the model;
      our adaptive observation window above is one small step in that
      direction.

    The unifying question stays the same: *what should the model remember,
    and at what granularity?*

    Which is exactly the question agent designers have been asking too —
    just at a different scale. That's the next section.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 14 · From KV Cache to Agentic Memory

    Zoom out one level. An LLM agent doesn't run for one prompt — it runs for
    *many turns*. Each turn appends user input, tool outputs, and assistant
    responses. The conversation grows. The same memory crisis we just fought
    at the **token level** shows up at the **turn level**: you can't keep
    everything, you can't drop everything, and which past turns matter
    depends on what the agent is trying to do *now*.

    Modern agent systems answer this with a memory hierarchy:
    """)
    return


@app.cell
def _(plot_memory_hierarchy):
    plot_memory_hierarchy()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Types of memory — and where KV cache fits

    The hierarchy above is *where* memory lives. The picture below is
    *what kind* of memory it is. Cognitive science gives us five canonical
    types; map each onto an LLM agent and the role of KV cache becomes
    sharp:
    """)
    return


@app.cell
def _(plot_memory_types):
    plot_memory_types()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two takeaways from the table:

    - **KV cache management can't fix what isn't there.** It won't add
      semantic knowledge the model didn't learn, and it won't recover
      episodes the agent never wrote down. Those are different problems
      with different solutions (training, RAG, memory stores).
    - **But it gates everything else.** A retrieved episode, a tool definition,
      a reasoning chain — all of them have to ride the cache to influence
      the next token. So how cleverly you spend cache budget compounds with
      every other memory layer your agent uses.

    That's why SnapKV-style ideas matter beyond the paper's benchmarks:
    they make the gateway wider for the same hardware budget.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The same knobs, one level up

    Every design decision SnapKV makes at the token level has a direct
    counterpart in agent memory:
    """)
    return


@app.cell
def _(plot_agent_mapping):
    plot_agent_mapping()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The interesting thing isn't that the analogy exists — it's that the
    *same shape of solution wins at both scales*. "Recent + currently
    relevant, with local context preserved" is a good policy for tokens in
    a KV cache and for turns in an agent loop. Get that wrong and either
    layer breaks the same way: lose old context, lose the thread.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## GAME 6 · Agent Memory Over Turns

    Run the same kinds of memory policies we used for tokens — but now over
    a multi-turn agent conversation. Each turn adds tokens; the strategy
    decides what stays in the **hot tier** (live context), what gets pushed
    to the **cold tier** (summarised / external store), and what gets lost.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    agent_strategy = mo.ui.radio(
        options=[
            "Full Cache",
            "Streaming (recent only)",
            "SnapKV-style (intent-aware)",
            "Agent + Summarise",
        ],
        value="Agent + Summarise",
        label="Memory strategy",
    )
    agent_turns = mo.ui.slider(start=4, stop=30, step=1, value=14,
                               label="Number of conversation turns")
    agent_limit = mo.ui.slider(start=200, stop=4000, step=100, value=800,
                               label="Hot-tier limit (tokens)")
    mo.vstack([agent_strategy, agent_turns, agent_limit])
    return agent_limit, agent_strategy, agent_turns


@app.cell
def _(agent_limit, agent_strategy, agent_turns, simulate_agent_loop):
    simulate_agent_loop(
        n_turns=agent_turns.value,
        strategy=agent_strategy.value,
        kv_limit=agent_limit.value,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to try:

    - **Full Cache** with a long conversation — accessible tokens stay at
      100%, but the hot bar grows unbounded. This is what current agents
      *can't* do indefinitely.
    - **Streaming (recent only)** — once you exceed the hot-tier limit,
      tokens fall straight into the gray "evicted" band. Old context is
      gone forever.
    - **SnapKV-style** — same eviction shape, but the surviving tokens are
      the *informative* ones. Same hot footprint, better recall in practice.
    - **Agent + Summarise** — overflow gets compressed into the cold tier
      instead of being lost. Recall rate stays close to 100%, hot tier
      stays bounded — at the cost of running a summariser between turns.

    The recall-rate stat in the cards is the bottom line: *of all the
    tokens this agent has ever seen, how many can it still get to?*
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

    ### Why this matters beyond one paper

    The same idea generalises. At the **token level** it's SnapKV picking which
    KV entries to keep. At the **turn level** it's an agent picking which past
    messages to keep verbatim, which to summarise, and which to drop. Same
    question — *what should we remember, given what we're trying to do now?* —
    same answer-shape: recent + currently relevant, with local context preserved.

    SnapKV is appealing not because it's the final word on memory, but because
    it's a clean, intuitive instance of a much bigger pattern.

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
