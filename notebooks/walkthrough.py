import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium", title="SnapKV: LLM Knows What You Are Looking For Before Generation")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # SnapKV: LLM Knows What You Are Looking For Before Generation
        
        > **NeurIPS 2024** · Li et al. · UIUC + Cohere + Princeton
        
        Large language models store all past token representations in a **KV cache** during
        generation. This grows linearly with context length — a serious bottleneck for long
        inputs. SnapKV solves this with a single elegant observation:
        
        > *The model already knows which tokens matter before it starts generating.*
        
        In this notebook you'll build intuition for **why** that's true, **how** SnapKV
        exploits it, and interactively explore what happens when you compress the cache.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · The Problem: KV Cache Grows With Context")
    return


@app.cell
def __(mo, plot_kv_growth):
    mo.vstack([
        mo.md("Slide to see how cache memory scales with sequence length across different methods:"),
        plot_kv_growth,
    ])
    return


@app.cell(hide_code=True)
def __(mo):
    seq_len_slider = mo.ui.slider(
        start=1024, stop=32768, step=1024, value=8192,
        label="Sequence length (tokens)"
    )
    return seq_len_slider,


@app.cell
def __(seq_len_slider, plot_cache_comparison):
    plot_cache_comparison(seq_len_slider.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 2 · The Key Insight: Attention Is Consistent
        
        SnapKV is built on one empirical observation: **attention heads are consistent**.
        
        When generating token $t$, the attention pattern over the prompt looks almost
        identical to the pattern at token $t+1$, $t+2$, ... The model "decides" which
        prompt tokens matter during the **prefill stage** and sticks to that decision.
        
        This means we can look at the last few tokens of the prompt (the **observation
        window**) and use their attention patterns to predict which KV entries will be
        needed during generation — *before generation even starts*.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Attention consistency across generation steps")
    return


@app.cell(hide_code=True)
def __(mo):
    obs_window_slider = mo.ui.slider(
        start=4, stop=64, step=4, value=16,
        label="Observation window size"
    )
    head_selector = mo.ui.dropdown(
        options={"Head 0": 0, "Head 1": 1, "Head 2": 2, "Head 3": 3},
        value="Head 0",
        label="Attention head"
    )
    return obs_window_slider, head_selector


@app.cell
def __(obs_window_slider, head_selector, plot_attention_consistency):
    plot_attention_consistency(obs_window_slider.value, head_selector.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 3 · How SnapKV Works
        
        SnapKV compresses the KV cache in two stages after prefill:
        
        **Stage 1 — Vote:** For each attention head, use the observation window
        (last $w$ prompt tokens) to score every earlier token by how much attention
        it received. Select the top-$k$ tokens per head.
        
        **Stage 2 — Cluster:** Don't just keep the winners. Keep their neighbors too,
        using a pooling kernel of size $k_{pool}$. This preserves local context around
        important tokens, reducing fragmentation artifacts.
        
        The result: a compressed KV cache of fixed size $k$ that gets *concatenated*
        with the observation window, then used for all subsequent generation.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("### Interactive SnapKV walkthrough — step through the algorithm")
    return


@app.cell(hide_code=True)
def __(mo):
    algo_step = mo.ui.slider(
        start=1, stop=4, step=1, value=1,
        label="Algorithm step"
    )
    return algo_step,


@app.cell
def __(algo_step, render_algo_step):
    render_algo_step(algo_step.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 4 · Live Demo: SnapKV vs Full Cache vs H2O
        
        Enter a prompt and watch how each method selects (or evicts) tokens.
        Use the budget slider to control how aggressively we compress.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    prompt_input = mo.ui.text_area(
        value="The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930.",
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
        label="Eviction method"
    )
    return prompt_input, budget_slider, method_select


@app.cell
def __(prompt_input, budget_slider, method_select, run_demo):
    run_demo(prompt_input.value, budget_slider.value, method_select.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 5 · Head-Level Analysis
        
        One of SnapKV's strengths is that it operates **per attention head** — each head
        gets to keep the tokens *it* cares about, not a global average. This section
        shows how different heads specialize in different token types.
        """
    )
    return


@app.cell
def __(prompt_input, plot_per_head_heatmap):
    plot_per_head_heatmap(prompt_input.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 6 · Extension: Adaptive Observation Window
        
        The original paper uses a **fixed** observation window size $w$. But is that optimal?
        
        Our extension: measure **attention entropy** across heads to adaptively size the
        observation window. Heads with high entropy (diffuse attention) benefit from a
        larger window; heads with low entropy (focused attention) need fewer tokens.
        
        This mirrors Ada-KV's per-head budget idea but applies it to the *observation
        window* rather than the cache budget.
        """
    )
    return


@app.cell
def __(prompt_input, plot_adaptive_window):
    plot_adaptive_window(prompt_input.value)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## 7 · Budget vs Quality Tradeoff
        
        How much does quality degrade as we compress more aggressively?
        Compare SnapKV against baselines across different cache budgets.
        """
    )
    return


@app.cell
def __(plot_budget_quality):
    plot_budget_quality()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Summary
        
        | Method | Key idea | Budget type | Quality |
        |---|---|---|---|
        | Full cache | Keep everything | 100% | Baseline |
        | StreamingLLM | Sink tokens + recent window | Fixed | Degrades on long context |
        | H2O | Greedy top-k by cumulative attention | Fixed | Good |
        | **SnapKV** | **Obs. window voting + clustering** | **Fixed per head** | **Best** |
        | Adaptive SnapKV *(our extension)* | Entropy-guided window | Dynamic | *TBD* |
        
        SnapKV's insight — that attention patterns are set before generation begins —
        is both elegant and practical. It requires no fine-tuning, minimal code changes
        to a standard HuggingFace model, and delivers significant memory savings with
        negligible quality loss.
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()