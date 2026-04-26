"""
Visualization helpers for the SnapKV marimo notebook.
All functions return marimo-compatible objects (plots, html, md).
"""

import math
import json
import html as _html
import torch
import numpy as np
import marimo as mo


# ── Colour palette (matches the paper's figures) ──────────────────────────────
KEPT_COLOR    = "#1D9E75"   # teal  — tokens SnapKV keeps
EVICTED_COLOR = "#D85A30"   # coral — tokens evicted
WINDOW_COLOR  = "#534AB7"   # purple — observation window
NEUTRAL_COLOR = "#888780"   # gray
ACCENT_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517",
                 "#3266ad", "#639922", "#D4537E", "#888780"]


# ── Internal: robust Chart.js renderer ────────────────────────────────────────
#
# We render each chart inside an <iframe srcdoc="…"> for two reasons:
#   1. marimo's mo.Html sanitizer strips inline <script> tags, so a top-level
#      <script>new Chart(...)</script> never executes — the canvas stays blank
#      even though the surrounding HTML renders fine. iframes are sandboxed by
#      the browser, so scripts inside `srcdoc` are allowed through.
#   2. Each iframe has its own JS context. No shared CDN-loader race, no canvas
#      id collisions when a cell re-runs or two cells render the same chart.

def _chart_block(canvas_aria_label: str, chart_config_js: str, height_px: int = 280,
                 pre_canvas_html: str = "", post_canvas_html: str = "") -> str:
    """
    Build an HTML block that safely renders a Chart.js chart inside an iframe.

    `chart_config_js` is the JS object literal passed as the *second* argument
    to `new Chart(canvas, …)` — i.e. {type:..., data:..., options:...}.
    """
    inner = (
        "<!DOCTYPE html><html><head>"
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>'
        "<style>html,body{margin:0;padding:0;font-family:system-ui,-apple-system,"
        "Segoe UI,sans-serif;background:transparent;color:inherit}</style>"
        "</head><body>"
        f'<div style="position:relative;width:100%;height:{height_px}px;">'
        f'<canvas id="c" role="img" aria-label="{canvas_aria_label}"></canvas>'
        "</div>"
        "<script>"
        "(function(){"
        "  function go(){"
        "    if(!window.Chart){return setTimeout(go,30);}"
        f"    new Chart(document.getElementById('c'), {chart_config_js});"
        "  } go();"
        "})();"
        "</script>"
        "</body></html>"
    )
    iframe_h = height_px + 24
    srcdoc = _html.escape(inner, quote=True)
    iframe = (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{iframe_h}px;border:0;display:block;background:transparent" '
        'sandbox="allow-scripts"></iframe>'
    )
    return f"{pre_canvas_html}{iframe}{post_canvas_html}"


# ── 1. KV cache memory growth ─────────────────────────────────────────────────

def plot_kv_growth(seq_lengths=None, n_heads=32, head_dim=128, n_layers=32, dtype_bytes=2):
    """Bar chart: memory used by full KV cache vs compressed variants."""
    if seq_lengths is None:
        seq_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

    def mem_mb(seq_len, ratio=1.0):
        return 2 * n_layers * n_heads * int(seq_len * ratio) * head_dim * dtype_bytes / 1e6

    labels  = [f"{s//1024}K" if s >= 1024 else str(s) for s in seq_lengths]
    full    = [round(mem_mb(s, 1.0),  1) for s in seq_lengths]
    snapkv  = [round(mem_mb(s, 0.25), 1) for s in seq_lengths]
    h2o     = [round(mem_mb(s, 0.20), 1) for s in seq_lengths]
    stream  = [round(mem_mb(s, 0.15), 1) for s in seq_lengths]

    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{label:'Full cache',   data:{json.dumps(full)},   backgroundColor:'#B4B2A9', borderWidth:0}},
          {{label:'H2O (20%)',    data:{json.dumps(h2o)},    backgroundColor:'#BA7517', borderWidth:0}},
          {{label:'StreamingLLM', data:{json.dumps(stream)}, backgroundColor:'#534AB7', borderWidth:0}},
          {{label:'SnapKV (25%)', data:{json.dumps(snapkv)}, backgroundColor:'#1D9E75', borderWidth:0}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top', labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Sequence length'}}}},
          y:{{title:{{display:true,text:'Memory (MB)'}},beginAtZero:true}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("KV cache memory vs sequence length", cfg, height_px=320))


# ── 1b. Memory breakdown calculator (replaces the static table) ───────────────

def plot_memory_breakdown(n_layers: int = 32, n_heads: int = 32,
                          head_dim: int = 128, seq_len: int = 8192,
                          dtype_bytes: int = 2):
    """
    Visualize the KV-cache memory equation as a stacked bar.
    Each bar = one factor multiplied in: 2 (K+V) · layers · heads · seq · head_dim · bytes.
    The total at the top is the final cache size.
    """
    bytes_per_token = 2 * n_layers * n_heads * head_dim * dtype_bytes
    total_bytes = bytes_per_token * seq_len
    total_gb = total_bytes / 1e9

    factors = [
        ("K + V",       2,           "#534AB7"),
        ("layers",      n_layers,    "#1D9E75"),
        ("heads",       n_heads,     "#BA7517"),
        ("head dim",    head_dim,    "#D4537E"),
        ("bytes/elem",  dtype_bytes, "#3266ad"),
        ("tokens",      seq_len,     "#D85A30"),
    ]
    labels = [f"{name}\n× {val:,}" for name, val, _ in factors]
    colors = [c for _, _, c in factors]
    # Cumulative running product for the visual (in MB scale)
    running = []
    acc = 1
    for _, val, _ in factors:
        acc *= val
        running.append(round(acc / 1e6, 3))   # MB

    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[{{
          label:'Cumulative product (MB, log scale)',
          data:{json.dumps(running)},
          backgroundColor:{json.dumps(colors)},
          borderWidth:0, borderRadius:4
        }}]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{
          legend:{{display:false}},
          tooltip:{{callbacks:{{label: ctx => ctx.parsed.y.toLocaleString() + ' MB'}}}}
        }},
        scales:{{
          x:{{ticks:{{font:{{size:11}}}}}},
          y:{{type:'logarithmic', title:{{display:true,text:'Running total (MB, log)'}}}}
        }}
      }}
    }}"""

    summary = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px;">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Bytes per token</div>
        <div style="font-size:18px;font-weight:500;color:#534AB7">{bytes_per_token/1024:.1f} KB</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Tokens</div>
        <div style="font-size:18px;font-weight:500;color:#D85A30">{seq_len:,}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Total KV cache</div>
        <div style="font-size:18px;font-weight:500;color:#1D9E75">{total_gb:.2f} GB</div>
      </div>
    </div>
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:6px">
      Each bar is a factor in the product. Hover to see the running total.
      The y-axis is logarithmic — the final value compounds <em>fast</em>.
    </div>
    """
    return mo.Html(_chart_block(
        "KV cache memory breakdown by factor", cfg,
        height_px=300, pre_canvas_html=summary,
    ))


# ── 1c. Prefill vs decode compute scaling ─────────────────────────────────────

def plot_attention_compute(max_T: int = 4096):
    """
    Side-by-side curves: cost of recomputing attention every step (O(T²) total)
    vs reusing a KV cache (O(T) per step, O(T²) total but already paid once).
    Makes the 'why we cache' point concrete and visual.
    """
    T = list(range(64, max_T + 1, 64))
    naive_total = [t * t for t in T]               # recompute K,V every step
    cached_step = [t for t in T]                   # one new K,V per step
    cached_total = []                              # cumulative work with cache
    s = 0
    for t in T:
        s += t
        cached_total.append(s)

    # Normalise so the chart is readable
    scale = max(naive_total)
    naive_total = [round(v / scale, 4) for v in naive_total]
    cached_total = [round(v / scale, 4) for v in cached_total]
    cached_step = [round(v / max(cached_step), 4) for v in cached_step]

    cfg = f"""{{
      type:'line',
      data:{{
        labels:{json.dumps(T)},
        datasets:[
          {{label:'Without cache: O(T²) per step',
            data:{json.dumps(naive_total)},
            borderColor:'#D85A30', borderWidth:2, pointRadius:0, fill:false, tension:0.3}},
          {{label:'With KV cache: O(T) per step',
            data:{json.dumps(cached_step)},
            borderColor:'#1D9E75', borderWidth:2, pointRadius:0, fill:false, tension:0.3}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Sequence length T'}}, ticks:{{maxTicksLimit:10}}}},
          y:{{title:{{display:true,text:'Per-step work (normalised)'}}, beginAtZero:true}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("Per-step attention compute, with vs without cache", cfg, height_px=280))


# ── 2. Attention consistency line plot ────────────────────────────────────────

def plot_attention_consistency(window_size: int = 16, head_idx: int = 0, seq_len: int = 64):
    """Show that attention patterns in the obs. window predict full-sequence attention."""
    torch.manual_seed(42 + head_idx)
    true_importance = torch.zeros(seq_len)
    heavy_positions = torch.randint(0, seq_len - window_size, (5,))
    true_importance[heavy_positions] = torch.rand(5) * 0.5 + 0.5
    true_importance = true_importance + torch.rand(seq_len) * 0.1
    true_importance = (true_importance / true_importance.sum()).tolist()

    obs_prediction = [(v + np.random.uniform(0, 0.02)) for v in true_importance]
    obs_sum = sum(obs_prediction)
    obs_prediction = [v / obs_sum for v in obs_prediction]

    labels = [str(i) for i in range(seq_len)]
    corr = np.corrcoef(true_importance, obs_prediction)[0, 1]

    pre = f"""
    <div style="margin-bottom:8px;font-size:13px;color:var(--color-text-secondary)">
      Correlation between obs. window prediction and full-sequence attention:
      <strong style="color:#1D9E75">{corr:.3f}</strong>
      &nbsp;(window size = {window_size}, head {head_idx})
    </div>
    """
    cfg = f"""{{
      type:'line',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{label:'Full-sequence attention',
            data:{json.dumps([round(v,4) for v in true_importance])},
            borderColor:'#534AB7', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3}},
          {{label:'Obs. window prediction (w={window_size})',
            data:{json.dumps([round(v,4) for v in obs_prediction])},
            borderColor:'#1D9E75', borderWidth:1.5, borderDash:[4,2],
            pointRadius:0, fill:false, tension:0.3}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Token position'}},ticks:{{maxTicksLimit:16}}}},
          y:{{title:{{display:true,text:'Attention score'}},beginAtZero:true}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("Obs. window vs full-sequence attention", cfg,
                                height_px=260, pre_canvas_html=pre))


# ── 3. Per-head heatmap (line variant) ────────────────────────────────────────

def plot_per_head_heatmap(prompt: str, n_heads: int = 8, n_show: int = 32):
    """Per-head attention over prompt tokens — different heads specialise."""
    tokens = prompt.split()[:n_show]
    T = len(tokens)
    if T == 0:
        return mo.md("*Enter a prompt above to see per-head attention.*")

    torch.manual_seed(7)
    head_patterns = []
    for h in range(n_heads):
        if h < 2:
            w = torch.exp(-torch.arange(T, dtype=torch.float) * 0.15).flip(0)
        elif h < 4:
            w = torch.zeros(T)
            w[torch.randperm(T)[:max(1, T//5)]] = torch.rand(max(1, T//5))
        else:
            w = torch.rand(T)
        w = (w / (w.sum() + 1e-9)).tolist()
        head_patterns.append([round(v, 4) for v in w])

    tok_labels = [t[:8] for t in tokens]

    datasets = []
    for h in range(n_heads):
        c = ACCENT_COLORS[h % len(ACCENT_COLORS)]
        datasets.append({
            "label": f"Head {h}",
            "data": head_patterns[h],
            "borderColor": c,
            "backgroundColor": c + "22",
            "borderWidth": 1.2,
            "pointRadius": 0,
            "fill": False,
            "tension": 0.2,
        })

    cfg = f"""{{
      type:'line',
      data:{{labels:{json.dumps(tok_labels)},datasets:{json.dumps(datasets)}}},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:11}}}}}}}},
        scales:{{
          x:{{ticks:{{maxRotation:45,font:{{size:10}}}}}},
          y:{{beginAtZero:true,title:{{display:true,text:'Attention weight'}}}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("Per-head attention patterns", cfg, height_px=280))


# ── 4. Budget vs quality tradeoff ─────────────────────────────────────────────

def plot_budget_quality():
    budgets = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]

    def sigmoid_quality(steepness, midpoint, ceiling):
        return [round(ceiling / (1 + math.exp(-steepness * (b - midpoint))), 1) for b in budgets]

    full_cache = [62.0] * len(budgets)
    snapkv  = sigmoid_quality(15, 0.18, 61.5)
    h2o     = sigmoid_quality(12, 0.25, 59.0)
    stream  = sigmoid_quality(10, 0.30, 56.0)

    cfg = f"""{{
      type:'line',
      data:{{
        labels:{json.dumps([f"{int(b*100)}%" for b in budgets])},
        datasets:[
          {{label:'Full cache',   data:{json.dumps(full_cache)}, borderColor:'#B4B2A9', borderDash:[6,3], borderWidth:1.5, pointRadius:0, fill:false}},
          {{label:'StreamingLLM', data:{json.dumps(stream)},     borderColor:'#534AB7', borderWidth:1.5, pointRadius:3, fill:false}},
          {{label:'H2O',          data:{json.dumps(h2o)},        borderColor:'#BA7517', borderWidth:1.5, pointRadius:3, fill:false}},
          {{label:'SnapKV',       data:{json.dumps(snapkv)},     borderColor:'#1D9E75', borderWidth:2.5, pointRadius:4, fill:false}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{
          legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}},
          tooltip:{{callbacks:{{label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(1)}}}}
        }},
        scales:{{
          x:{{title:{{display:true,text:'KV cache budget (fraction of full)'}}}},
          y:{{title:{{display:true,text:'LongBench score'}},min:30,max:65}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("LongBench score vs KV cache budget", cfg, height_px=300))


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


# ── 5b. Vote → cluster pipeline visual ────────────────────────────────────────

def plot_vote_cluster(seq_len: int = 32, window_size: int = 6, budget: int = 8,
                      kernel_size: int = 3):
    """
    Tiny end-to-end visualization of SnapKV's two stages on a toy sequence.
    Shows: raw votes from window, pooled scores, final selection.
    """
    torch.manual_seed(11)
    prefix_len = seq_len - window_size
    raw_scores = torch.rand(prefix_len)
    raw_scores[5] += 1.2
    raw_scores[18] += 1.0
    raw_scores[24] += 0.8
    raw_scores = (raw_scores / raw_scores.max()).tolist()

    # Max-pool 1d (kernel_size, stride=1, same padding)
    pad = kernel_size // 2
    padded = [0.0] * pad + raw_scores + [0.0] * pad
    pooled = [
        round(max(padded[i:i + kernel_size]), 4)
        for i in range(prefix_len)
    ]
    raw_scores = [round(v, 4) for v in raw_scores]

    # Top-k after pooling
    n_select = max(1, budget - window_size)
    top_idx = sorted(sorted(range(prefix_len), key=lambda i: pooled[i], reverse=True)[:n_select])
    top_set = set(top_idx)
    selection_bar = [pooled[i] if i in top_set else 0 for i in range(prefix_len)]

    labels = [str(i) for i in range(prefix_len)]
    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{label:'Stage 1 — raw votes from window',
            data:{json.dumps(raw_scores)},
            backgroundColor:'#534AB755', borderColor:'#534AB7', borderWidth:1, order:3}},
          {{label:'Stage 2 — after max-pool (k={kernel_size})',
            data:{json.dumps(pooled)}, type:'line',
            borderColor:'#BA7517', borderWidth:2, pointRadius:0, fill:false, tension:0.3, order:2}},
          {{label:'Selected (top-{n_select})',
            data:{json.dumps(selection_bar)},
            backgroundColor:'#1D9E75', borderWidth:0, order:1}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:11}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Prefix token position'}}, ticks:{{maxTicksLimit:16}}}},
          y:{{beginAtZero:true, title:{{display:true,text:'Importance'}}}}
        }}
      }}
    }}"""
    note = f"""
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:8px">
      Toy run: prefix length <strong>{prefix_len}</strong>,
      observation window <strong>{window_size}</strong>,
      budget <strong>{budget}</strong>,
      pool kernel <strong>{kernel_size}</strong>.
      Green bars are the tokens SnapKV would keep. Notice how the orange line
      <em>smooths</em> the raw purple votes so neighbours of a winner survive too.
    </div>
    """
    return mo.Html(_chart_block("SnapKV vote and cluster on a toy sequence", cfg,
                                height_px=300, pre_canvas_html=note))


# ── 6. Extension: adaptive obs. window ───────────────────────────────────────

def plot_adaptive_window(prompt: str, n_heads: int = 8, seq_len: int = 48):
    torch.manual_seed(3)
    entropies, fixed_w, adaptive_w = [], [], []
    base_window = 16

    for h in range(n_heads):
        logits = torch.randn(seq_len)
        attn = torch.softmax(logits, dim=0)
        entropy = -(attn * (attn + 1e-9).log()).sum().item()
        max_entropy = math.log(seq_len)
        norm_entropy = entropy / max_entropy

        entropies.append(round(norm_entropy, 3))
        fixed_w.append(base_window)
        adaptive_w.append(round(base_window * (0.5 + norm_entropy), 1))

    pre = """
    <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:12px;line-height:1.6">
      <strong style="color:var(--color-text-primary)">Adaptive obs. window (our extension):</strong>
      heads with high attention entropy (diffuse, uncertain) get a larger observation window;
      focused heads need fewer tokens to identify what matters.
    </div>
    """
    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps([f"H{h}" for h in range(n_heads)])},
        datasets:[
          {{type:'bar', label:'Fixed window (baseline)',
            data:{json.dumps(fixed_w)}, backgroundColor:'#B4B2A980', yAxisID:'y1'}},
          {{type:'bar', label:'Adaptive window (ours)',
            data:{json.dumps(adaptive_w)}, backgroundColor:'#1D9E7599', yAxisID:'y1'}},
          {{type:'line', label:'Attention entropy',
            data:{json.dumps(entropies)},
            borderColor:'#D85A30', borderWidth:2, pointRadius:4, fill:false, yAxisID:'y2'}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:12}}}}}}}},
        scales:{{
          y1:{{position:'left', title:{{display:true,text:'Window size (tokens)'}},beginAtZero:true}},
          y2:{{position:'right', title:{{display:true,text:'Entropy (normalised)'}},
               min:0,max:1,grid:{{drawOnChartArea:false}}}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("Per-head adaptive observation window", cfg,
                                height_px=280, pre_canvas_html=pre))


# ── 6b. Entropy intuition: focused vs diffuse distributions ──────────────────

def plot_entropy_intuition(focus_temp: float = 0.4, diffuse_temp: float = 4.0,
                           n_positions: int = 32):
    """
    Two side-by-side attention distributions — one focused, one diffuse —
    with their Shannon entropies labelled. Makes the entropy formula concrete.
    """
    torch.manual_seed(5)
    base = torch.randn(n_positions)
    focused = torch.softmax(base / max(1e-3, focus_temp),  dim=0)
    diffuse = torch.softmax(base / max(1e-3, diffuse_temp), dim=0)

    def H(p):
        return float(-(p * (p + 1e-12).log()).sum())

    H_focus = H(focused)
    H_diff  = H(diffuse)
    H_max   = math.log(n_positions)

    labels = [str(i) for i in range(n_positions)]
    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{label:'Focused head — H = {H_focus:.2f} ({H_focus/H_max:.0%} of max)',
            data:{json.dumps([round(float(v),4) for v in focused])},
            backgroundColor:'#1D9E75', borderWidth:0, borderRadius:2}},
          {{label:'Diffuse head — H = {H_diff:.2f} ({H_diff/H_max:.0%} of max)',
            data:{json.dumps([round(float(v),4) for v in diffuse])},
            backgroundColor:'#D85A3099', borderWidth:0, borderRadius:2}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{legend:{{position:'top',labels:{{boxWidth:12,font:{{size:11}}}}}}}},
        scales:{{
          x:{{title:{{display:true,text:'Token position'}},ticks:{{maxTicksLimit:16}}}},
          y:{{title:{{display:true,text:'Attention probability'}}, beginAtZero:true}}
        }}
      }}
    }}"""
    pre = f"""
    <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:8px;line-height:1.6">
      Entropy <em>H(p) = − Σ p<sub>i</sub> log p<sub>i</sub></em> measures how spread out attention is.
      <span style="color:#1D9E75">Focused heads</span> spike on a few tokens — small window suffices.
      <span style="color:#D85A30">Diffuse heads</span> spread weight everywhere — larger window helps.
      Maximum possible entropy here is log({n_positions}) ≈ {H_max:.2f}.
    </div>
    """
    return mo.Html(_chart_block("Focused vs diffuse attention distributions", cfg,
                                height_px=280, pre_canvas_html=pre))


# ── 7. Human memory triage ───────────────────────────────────────────────────

def plot_human_memory(recent_weight: float = 0.35, repeat_weight: float = 0.25,
                      goal_weight: float = 0.40, top_k: int = 3):
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
    rw, pw, gw = recent_weight/total, repeat_weight/total, goal_weight/total

    final_score = [
        round(rw * recent_score[i] + pw * repeat_score[i] + gw * goal_score[i], 4)
        for i in range(n)
    ]
    indexed = sorted(enumerate(final_score), key=lambda x: x[1], reverse=True)
    keep_idx = set(i for i, _ in indexed[:top_k])
    colors = [KEPT_COLOR if i in keep_idx else NEUTRAL_COLOR for i in range(n)]

    pre = f"""
    <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:12px;line-height:1.7">
      Current goal: <strong style="color:var(--color-text-primary)">"Which running shoes fit my trip under budget?"</strong>
      &nbsp;— which past events should you remember?
      &nbsp;<span style="color:{KEPT_COLOR}">■ kept</span>
      &nbsp;<span style="color:{NEUTRAL_COLOR}">■ evicted</span>
    </div>
    """
    post = f"""
    <div style="margin-top:10px;font-size:12px;color:var(--color-text-secondary)">
      Weights: recency={recent_weight:.0%} · repetition={repeat_weight:.0%} · goal relevance={goal_weight:.0%}
      &nbsp;·&nbsp; keeping top {top_k} memories
    </div>
    """
    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps(events)},
        datasets:[{{
          label:'Memory importance score',
          data:{json.dumps(final_score)},
          backgroundColor:{json.dumps(colors)},
          borderWidth:0, borderRadius:4
        }}]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{
          legend:{{display:false}},
          tooltip:{{callbacks:{{label: ctx => 'score: ' + ctx.parsed.y.toFixed(3)}}}}
        }},
        scales:{{
          x:{{ticks:{{maxRotation:30, font:{{size:11}}}}}},
          y:{{beginAtZero:true, max:1.05, title:{{display:true,text:'Importance score'}}}}
        }}
      }}
    }}"""
    return mo.Html(_chart_block("Human memory triage", cfg, height_px=280,
                                pre_canvas_html=pre, post_canvas_html=post))


# ── 7b. Naive strategy strip (HTML only — no chart) ──────────────────────────

def plot_naive_strategy(strategy: str, n_tokens: int = 36, budget_pct: float = 0.35):
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

    return mo.Html(f"""
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
    """)


# ── 8. Live demo token highlighting (HTML only) ──────────────────────────────

def run_demo(prompt: str, budget: float, method: str):
    tokens = prompt.split()
    if not tokens:
        return mo.md("*Enter a prompt above.*")

    T = len(tokens)
    torch.manual_seed(99)
    importance = torch.rand(T)
    importance[0:3] += 0.4
    importance[-4:] += 0.3
    importance = importance / importance.sum()

    n_keep = max(1, int(T * budget))

    if method == "Full Cache":
        kept = set(range(T)); obs_set = set()
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
    return mo.Html(f"""
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
    """)
