"""
Visualization helpers for the SnapKV marimo notebook.
All functions return marimo-compatible objects (plots, html, md).
"""

import math
import torch
import numpy as np
import marimo as mo


# ── Colour palette (matches the paper's figures) ──────────────────────────────
KEPT_COLOR    = "#1D9E75"   # teal  — tokens SnapKV keeps
EVICTED_COLOR = "#D85A30"   # coral — tokens evicted
WINDOW_COLOR  = "#534AB7"   # purple — observation window
NEUTRAL_COLOR = "#888780"   # gray

# ── 1. KV cache memory growth ─────────────────────────────────────────────────

def plot_kv_growth(seq_lengths=None, n_heads=32, head_dim=128, n_layers=32, dtype_bytes=2):
    """Bar chart: memory used by full KV cache vs compressed variants."""
    import json
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    def mem_mb(seq_len, ratio=1.0):
        # 2 (K+V) * layers * heads * seq * head_dim * bytes / 1e6
        return 2 * n_layers * n_heads * int(seq_len * ratio) * head_dim * dtype_bytes / 1e6

    labels  = [f"{s//1024}K" for s in seq_lengths]
    full    = [mem_mb(s, 1.0)  for s in seq_lengths]
    snapkv  = [mem_mb(s, 0.25) for s in seq_lengths]
    h2o     = [mem_mb(s, 0.20) for s in seq_lengths]
    stream  = [mem_mb(s, 0.15) for s in seq_lengths]

    html = f"""
    <div style="position:relative;width:100%;height:320px;">
    <canvas id="kvgrowth" role="img" aria-label="KV cache memory vs sequence length for different methods">
      Full cache grows from {full[0]:.0f}MB at 1K tokens to {full[-1]:.0f}MB at 32K tokens.
    </canvas></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('kvgrowth'),{{
      type:'bar',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{label:'Full cache', data:{json.dumps(full)}, backgroundColor:'#B4B2A9', borderWidth:0}},
          {{label:'H2O (20%)',  data:{json.dumps(h2o)},  backgroundColor:'#BA7517', borderWidth:0}},
          {{label:'StreamingLLM', data:{json.dumps(stream)}, backgroundColor:'#534AB7', borderWidth:0}},
          {{label:'SnapKV (25%)', data:{json.dumps(snapkv)}, backgroundColor:'#1D9E75', borderWidth:0}},
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{legend:{{position:'top', labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Sequence length'}}}},
          y:{{title:{{display:true,text:'Memory (MB)'}},beginAtZero:true}}
        }}
      }}
    }});
    </script>
    """
    return mo.Html(html)


# ── 2. Attention consistency heatmap ─────────────────────────────────────────

def plot_attention_consistency(window_size: int = 16, head_idx: int = 0, seq_len: int = 64):
    """
    Show that attention patterns in the obs. window predict full-sequence attention.
    Uses synthetic data that mirrors Figure 3 from the paper.
    """
    import json
    torch.manual_seed(42 + head_idx)
    # Simulate a "true" attention pattern with a few heavy hitters
    true_importance = torch.zeros(seq_len)
    heavy_positions = torch.randint(0, seq_len - window_size, (5,))
    true_importance[heavy_positions] = torch.rand(5) * 0.5 + 0.5
    true_importance = true_importance + torch.rand(seq_len) * 0.1
    true_importance = (true_importance / true_importance.sum()).tolist()

    # Obs. window prediction (noisier but correlated)
    obs_prediction = [(v + np.random.uniform(0, 0.02)) for v in true_importance]
    obs_sum = sum(obs_prediction)
    obs_prediction = [v / obs_sum for v in obs_prediction]

    labels = [str(i) for i in range(seq_len)]
    corr = np.corrcoef(true_importance, obs_prediction)[0, 1]

    html = f"""
    <div style="margin-bottom:8px;font-size:13px;color:var(--color-text-secondary)">
      Correlation between obs. window prediction and full-sequence attention:
      <strong style="color:#1D9E75">{corr:.3f}</strong>
      &nbsp;(window size = {window_size}, head {head_idx})
    </div>
    <div style="position:relative;width:100%;height:260px;">
    <canvas id="consistency" role="img" aria-label="Attention score comparison: observation window vs full sequence">
    </canvas></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('consistency'),{{
      type:'line',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{
            label:'Full-sequence attention',
            data:{json.dumps([round(v,4) for v in true_importance])},
            borderColor:'#534AB7', borderWidth:1.5,
            pointRadius:0, fill:false, tension:0.3
          }},
          {{
            label:'Obs. window prediction (w={window_size})',
            data:{json.dumps([round(v,4) for v in obs_prediction])},
            borderColor:'#1D9E75', borderWidth:1.5, borderDash:[4,2],
            pointRadius:0, fill:false, tension:0.3
          }}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Token position'}},ticks:{{maxTicksLimit:16}}}},
          y:{{title:{{display:true,text:'Attention score'}},beginAtZero:true}}
        }}
      }}
    }});
    </script>
    """
    return mo.Html(html)


# ── 3. Per-head heatmap ───────────────────────────────────────────────────────

def plot_per_head_heatmap(prompt: str, n_heads: int = 8, n_show: int = 32):
    """
    Heatmap grid showing which tokens each head attends to.
    Simulates head specialization (syntactic, semantic, positional heads).
    """
    import json
    tokens = prompt.split()[:n_show]
    T = len(tokens)
    if T == 0:
        return mo.md("*Enter a prompt above to see per-head attention.*")

    torch.manual_seed(7)
    head_patterns = []
    for h in range(n_heads):
        if h < 2:   # positional heads — prefer recent tokens
            w = torch.exp(-torch.arange(T, dtype=torch.float) * 0.15).flip(0)
        elif h < 4: # semantic heads — sparse high-value tokens
            w = torch.zeros(T)
            w[torch.randperm(T)[:max(1, T//5)]] = torch.rand(max(1, T//5))
        else:       # mixed heads
            w = torch.rand(T)
        w = (w / (w.sum() + 1e-9)).tolist()
        head_patterns.append([round(v, 4) for v in w])

    tok_labels = [t[:8] for t in tokens]  # truncate for display

    datasets = []
    colors = ["#534AB7","#1D9E75","#D85A30","#BA7517","#3266ad","#639922","#D4537E","#888780"]
    for h in range(n_heads):
        datasets.append({
            "label": f"Head {h}",
            "data": head_patterns[h],
            "borderColor": colors[h % len(colors)],
            "backgroundColor": colors[h % len(colors)] + "22",
            "borderWidth": 1.2,
            "pointRadius": 0,
            "fill": False,
            "tension": 0.2,
        })

    html = f"""
    <div style="position:relative;width:100%;height:280px;">
    <canvas id="perhead" role="img" aria-label="Per-head attention patterns over prompt tokens"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('perhead'),{{
      type:'line',
      data:{{labels:{json.dumps(tok_labels)},datasets:{json.dumps(datasets)}}},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:11}}}}}}}},
        scales:{{
          x:{{ticks:{{maxRotation:45,font:{{size:10}}}}}},
          y:{{beginAtZero:true,title:{{display:true,text:'Attention weight'}}}}
        }}
      }}
    }});
    </script>
    """
    return mo.Html(html)


# ── 4. Budget vs quality tradeoff ─────────────────────────────────────────────

def plot_budget_quality():
    """
    Reproduces the qualitative shape of Figure 5 from SnapKV paper:
    LongBench score vs KV cache budget for different methods.
    """
    import json
    budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    def sigmoid_quality(budgets, steepness, midpoint, ceiling):
        return [ceiling / (1 + math.exp(-steepness * (b - midpoint))) for b in budgets]

    full_cache = [62.0] * len(budgets)
    snapkv  = sigmoid_quality(budgets, 15, 0.18, 61.5)
    h2o     = sigmoid_quality(budgets, 12, 0.25, 59.0)
    stream  = sigmoid_quality(budgets, 10, 0.30, 56.0)

    html = f"""
    <div style="position:relative;width:100%;height:300px;">
    <canvas id="bqtradeoff" role="img" aria-label="LongBench score vs KV cache budget for SnapKV, H2O, StreamingLLM and full cache"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('bqtradeoff'),{{
      type:'line',
      data:{{
        labels:{json.dumps([f"{int(b*100)}%" for b in budgets])},
        datasets:[
          {{label:'Full cache',   data:{json.dumps([round(v,1) for v in full_cache])}, borderColor:'#B4B2A9', borderDash:[6,3], borderWidth:1.5, pointRadius:0, fill:false}},
          {{label:'StreamingLLM', data:{json.dumps([round(v,1) for v in stream])},    borderColor:'#534AB7', borderWidth:1.5, pointRadius:3, fill:false}},
          {{label:'H2O',          data:{json.dumps([round(v,1) for v in h2o])},       borderColor:'#BA7517', borderWidth:1.5, pointRadius:3, fill:false}},
          {{label:'SnapKV',       data:{json.dumps([round(v,1) for v in snapkv])},    borderColor:'#1D9E75', borderWidth:2.5, pointRadius:4, fill:false}},
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}},
          tooltip:{{callbacks:{{label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1)}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'KV cache budget (fraction of full)'}}}},
          y:{{title:{{display:true,text:'LongBench score'}},min:30,max:65}}
        }}
      }}
    }});
    </script>
    """
    return mo.Html(html)


# ── 5. Algorithm step-through ─────────────────────────────────────────────────

def render_algo_step(step: int):
    steps = {
        1: ("Prefill the full sequence",
            "The model processes the entire prompt and builds the full KV cache. "
            "Every token has a Key and Value vector stored in GPU memory.",
            "#534AB7"),
        2: ("Identify the observation window",
            "Take the last <strong>w</strong> tokens of the prompt as the observation window. "
            "These tokens will always be kept — they represent the most recent context the "
            "model needs to answer the query.",
            "#1D9E75"),
        3: ("Vote: score prefix tokens using window attention",
            "For each attention head, compute how much each prefix token was attended to "
            "by the observation window. Average across window positions to get a per-token "
            "importance score. Select the top-<strong>k</strong> tokens per head.",
            "#BA7517"),
        4: ("Cluster: pool neighbors of selected tokens",
            "Apply max-pooling with kernel size k_pool around each selected token. "
            "This keeps local context around important tokens, avoiding fragmented "
            "retrieval artifacts. Concatenate with the observation window → compressed cache.",
            "#D85A30"),
    }
    title, desc, color = steps[step]
    return mo.Html(f"""
    <div style="border-left:3px solid {color};padding:12px 16px;border-radius:0 8px 8px 0;
         background:var(--color-background-secondary);margin:8px 0;">
      <div style="font-size:11px;font-weight:500;color:{color};letter-spacing:.06em;
           text-transform:uppercase;margin-bottom:4px;">Step {step} / 4</div>
      <div style="font-size:15px;font-weight:500;color:var(--color-text-primary);margin-bottom:6px;">
        {title}
      </div>
      <div style="font-size:13px;color:var(--color-text-secondary);line-height:1.6;">
        {desc}
      </div>
    </div>
    """)


# ── 6. Extension: adaptive obs. window ───────────────────────────────────────

def plot_adaptive_window(prompt: str, n_heads: int = 8, seq_len: int = 48):
    """
    Show how attention entropy varies per head, motivating adaptive window sizes.
    """
    import json
    torch.manual_seed(3)
    entropies, fixed_w, adaptive_w = [], [], []
    base_window = 16

    for h in range(n_heads):
        # Simulate a random attention distribution
        logits = torch.randn(seq_len)
        attn = torch.softmax(logits, dim=0)
        entropy = -(attn * (attn + 1e-9).log()).sum().item()
        max_entropy = math.log(seq_len)
        norm_entropy = entropy / max_entropy

        entropies.append(round(norm_entropy, 3))
        fixed_w.append(base_window)
        # Adaptive: scale window with entropy (more diffuse → bigger window)
        adaptive_w.append(round(base_window * (0.5 + norm_entropy), 1))

    html = f"""
    <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:12px;line-height:1.6">
      <strong style="color:var(--color-text-primary)">Adaptive obs. window (our extension):</strong>
      heads with high attention entropy (diffuse, uncertain) get a larger observation window;
      focused heads need fewer tokens to identify what matters.
    </div>
    <div style="position:relative;width:100%;height:260px;">
    <canvas id="adaptive" role="img" aria-label="Normalized attention entropy and window size per head"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('adaptive'),{{
      type:'bar',
      data:{{
        labels:{json.dumps([f"H{h}" for h in range(n_heads)])},
        datasets:[
          {{
            type:'bar', label:'Fixed window (baseline)',
            data:{json.dumps(fixed_w)},
            backgroundColor:'#B4B2A980', yAxisID:'y1'
          }},
          {{
            type:'bar', label:'Adaptive window (ours)',
            data:{json.dumps(adaptive_w)},
            backgroundColor:'#1D9E7599', yAxisID:'y1'
          }},
          {{
            type:'line', label:'Attention entropy',
            data:{json.dumps(entropies)},
            borderColor:'#D85A30', borderWidth:2,
            pointRadius:4, fill:false, yAxisID:'y2'
          }}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          y1:{{position:'left', title:{{display:true,text:'Window size (tokens)'}},beginAtZero:true}},
          y2:{{position:'right', title:{{display:true,text:'Entropy (normalised)'}},
               min:0,max:1,grid:{{drawOnChartArea:false}}}}
        }}
      }}
    }});
    </script>
    """
    return mo.Html(html)


# ── 7. Human memory triage ───────────────────────────────────────────────────

def plot_human_memory(
    recent_weight: float = 0.35,
    repeat_weight: float = 0.25,
    goal_weight: float = 0.40,
    top_k: int = 3,
):
    """
    Bar chart: simulate human memory triage using recency, repetition, and goal relevance.
    Maps to SnapKV's observation-window voting concept.
    """
    import json

    events = [
        "saw a shoe ad",
        "friend mentioned sneakers",
        "walked past a sports store",
        "read laptop review",
        "checked running routes",
        "opened shoe size chart",
        "searched: best running shoes",
    ]
    n = len(events)

    repeat_score = [0.3, 0.7, 0.6, 0.2, 0.5, 0.8, 1.0]
    goal_score   = [0.4, 0.7, 0.8, 0.1, 0.9, 0.95, 1.0]
    recent_score = [round(0.2 + 0.8 * i / (n - 1), 3) for i in range(n)]

    total = recent_weight + repeat_weight + goal_weight + 1e-9
    rw = recent_weight / total
    pw = repeat_weight / total
    gw = goal_weight  / total

    final_score = [
        round(rw * recent_score[i] + pw * repeat_score[i] + gw * goal_score[i], 4)
        for i in range(n)
    ]

    indexed = sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)
    keep_idx = set(i for i, _ in indexed[:top_k])

    colors = [KEPT_COLOR if i in keep_idx else NEUTRAL_COLOR for i in range(n)]

    html = f"""
    <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:12px;line-height:1.7">
      Current goal: <strong style="color:var(--color-text-primary)">"Which running shoes fit my trip under budget?"</strong>
      &nbsp;— which past events should you remember?
      &nbsp;<span style="color:{KEPT_COLOR}">■ kept</span>
      &nbsp;<span style="color:{NEUTRAL_COLOR}">■ evicted</span>
    </div>
    <div style="position:relative;width:100%;height:280px;">
    <canvas id="humanmem" role="img" aria-label="Human memory triage bar chart"></canvas>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    new Chart(document.getElementById('humanmem'),{{
      type:'bar',
      data:{{
        labels:{json.dumps(events)},
        datasets:[{{
          label:'Memory importance score',
          data:{json.dumps(final_score)},
          backgroundColor:{json.dumps(colors)},
          borderWidth:0,
          borderRadius:4
        }}]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false,
        plugins:{{
          legend:{{display:false}},
          tooltip:{{callbacks:{{
            label: ctx => 'score: ' + ctx.parsed.y.toFixed(3)
          }}}}
        }},
        scales:{{
          x:{{ticks:{{maxRotation:30, font:{{size:11}}}}}},
          y:{{beginAtZero:true, max:1.05,
              title:{{display:true,text:'Importance score'}}}}
        }}
      }}
    }});
    </script>
    <div style="margin-top:10px;font-size:12px;color:var(--color-text-secondary)">
      Weights: recency={recent_weight:.0%} · repetition={repeat_weight:.0%} · goal relevance={goal_weight:.0%}
      &nbsp;·&nbsp; keeping top {top_k} memories
    </div>
    """
    return mo.Html(html)


def plot_naive_strategy(strategy: str, n_tokens: int = 36, budget_pct: float = 0.35):
    """
    Colour-coded token strip: show which tokens a naive eviction policy keeps.
    """
    import json

    n_keep = max(2, int(n_tokens * budget_pct))
    labels  = [f"t{i}" for i in range(n_tokens)]

    if strategy == "Keep Everything":
        kept = list(range(n_tokens))
    elif strategy == "Recent Only":
        kept = list(range(n_tokens - n_keep, n_tokens))
    elif strategy == "Random Drop":
        import random; random.seed(42)
        kept = sorted(random.sample(range(n_tokens), n_keep))
    elif strategy == "Uniform Stride":
        step = max(1, n_tokens // n_keep)
        kept = list(range(0, n_tokens, step))[:n_keep]
    else:
        kept = list(range(n_tokens))

    kept_set = set(kept)
    colors  = [KEPT_COLOR if i in kept_set else EVICTED_COLOR for i in range(n_tokens)]
    opacity = ["ff"       if i in kept_set else "55"          for i in range(n_tokens)]

    token_spans = "".join(
        f'<span title="{"kept" if i in kept_set else "evicted"}" '
        f'style="display:inline-block;margin:2px 1px;padding:4px 7px;'
        f'background:{colors[i]}33;border:1.5px solid {colors[i]};'
        f'border-radius:4px;font-size:11px;font-family:monospace">{labels[i]}</span>'
        for i in range(n_tokens)
    )

    compression = len(kept_set) / n_tokens
    problems = {
        "Keep Everything": "Problem: memory grows without bound. At 32K tokens you need &gt;100 GB just for KV cache.",
        "Recent Only": "Problem: loses all earlier context. If the answer is in the first paragraph, it's gone.",
        "Random Drop": "Problem: random eviction may discard critical tokens. Results are unpredictable.",
        "Uniform Stride": "Problem: no notion of importance. Evenly spaced tokens often miss key content.",
    }
    problem_text = problems.get(strategy, "")

    html = f"""
    <div style="display:flex;gap:12px;margin-bottom:14px;align-items:center">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 16px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Strategy</div>
        <div style="font-size:16px;font-weight:500;color:var(--color-text-primary)">{strategy}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 16px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Tokens kept</div>
        <div style="font-size:16px;font-weight:500;color:{KEPT_COLOR}">{len(kept_set)} / {n_tokens}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px 16px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Cache size</div>
        <div style="font-size:16px;font-weight:500;color:#BA7517">{compression:.0%}</div>
      </div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:12px;line-height:2;margin-bottom:10px">
      {token_spans}
    </div>
    <div style="padding:10px 14px;background:#D85A3015;border-left:3px solid {EVICTED_COLOR};
         border-radius:0 6px 6px 0;font-size:13px;color:var(--color-text-secondary)">
      ⚠ {problem_text}
    </div>
    """
    return mo.Html(html)


# ── 8. Live demo token highlighting ──────────────────────────────────────────

def run_demo(prompt: str, budget: float, method: str):
    """
    Tokenise the prompt (word-level) and colour-code which tokens each method keeps.
    Returns an HTML token display + stats card.
    """
    tokens = prompt.split()
    if not tokens:
        return mo.md("*Enter a prompt above.*")

    T = len(tokens)
    torch.manual_seed(99)

    # Simulate importance scores
    importance = torch.rand(T)
    importance[0:3] += 0.4   # sink tokens are naturally important
    importance[-4:] += 0.3   # recent tokens too
    importance = importance / importance.sum()

    n_keep = max(1, int(T * budget))

    if method == "Full Cache":
        kept = set(range(T))
        obs_set = set()
    elif method == "StreamingLLM":
        n_sink = max(1, min(4, T // 10))
        n_recent = max(1, n_keep - n_sink)
        kept = set(range(n_sink)) | set(range(T - n_recent, T))
        obs_set = set(range(T - n_recent, T))
    elif method == "H2O":
        n_recent = max(1, n_keep // 4)
        scores = importance.clone()
        scores[-n_recent:] = 2.0
        _, idx = scores.topk(n_keep)
        kept = set(idx.tolist())
        obs_set = set(range(T - n_recent, T))
    else:  # SnapKV
        obs_w = max(1, min(int(T * 0.25), 16))
        obs_set = set(range(T - obs_w, T))
        prefix_scores = importance[:-obs_w]
        n_prefix = max(1, n_keep - obs_w)
        _, idx = prefix_scores.topk(min(n_prefix, len(prefix_scores)))
        kept = set(idx.tolist()) | obs_set

    # Build token spans
    spans = []
    for i, tok in enumerate(tokens):
        if i in obs_set:
            bg, border, label = "#AFA9EC", "#534AB7", "obs. window"
        elif i in kept:
            bg, border, label = "#9FE1CB", "#1D9E75", "kept"
        else:
            bg, border, label = "#F5C4B3", "#D85A30", "evicted"
        spans.append(
            f'<span title="{label}" style="display:inline-block;margin:3px 2px;padding:3px 6px;'
            f'background:{bg};border:1px solid {border};border-radius:4px;font-size:13px;'
            f'font-family:var(--font-mono)">{tok}</span>'
        )

    compression = len(kept) / T
    html = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em">Method</div>
        <div style="font-size:18px;font-weight:500;color:var(--color-text-primary)">{method}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em">Tokens kept</div>
        <div style="font-size:18px;font-weight:500;color:#1D9E75">{len(kept)} / {T}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:12px;text-align:center">
        <div style="font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em">Cache size</div>
        <div style="font-size:18px;font-weight:500;color:#BA7517">{compression:.0%}</div>
      </div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:14px;line-height:1.9">
      <div style="font-size:11px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">
        Token selection
        &nbsp;·&nbsp;<span style="background:#9FE1CB;border:1px solid #1D9E75;border-radius:3px;padding:1px 6px;font-size:11px">kept</span>
        &nbsp;<span style="background:#AFA9EC;border:1px solid #534AB7;border-radius:3px;padding:1px 6px;font-size:11px">obs. window</span>
        &nbsp;<span style="background:#F5C4B3;border:1px solid #D85A30;border-radius:3px;padding:1px 6px;font-size:11px">evicted</span>
      </div>
      {''.join(spans)}
    </div>
    """
    return mo.Html(html)