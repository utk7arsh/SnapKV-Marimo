"""
Visualization helpers for the SnapKV marimo notebook.
All functions return marimo-compatible objects (plots, html, md).
"""

import math
import json
import html as _html
import numpy as np
import marimo as mo


# ── Colour palette (matches the paper's figures) ──────────────────────────────
KEPT_COLOR    = "#1D9E75"   # teal  — tokens SnapKV keeps
EVICTED_COLOR = "#D85A30"   # coral — tokens evicted
WINDOW_COLOR  = "#534AB7"   # purple — observation window
NEUTRAL_COLOR = "#888780"   # gray
ACCENT_COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517",
                 "#3266ad", "#639922", "#D4537E", "#888780"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _softmax(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    shifted = arr - np.max(arr)
    exps = np.exp(shifted)
    return exps / (exps.sum() + 1e-12)


def _topk_indices(x, k: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or k <= 0:
        return np.array([], dtype=int)
    k = min(k, arr.size)
    idx = np.argpartition(arr, -k)[-k:]
    return idx[np.argsort(arr[idx])[::-1]]


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
    # Cumulative running product for the visual (in MB scale).
    # Keep full precision here: rounding early can turn the first factors into
    # literal zeros, which breaks Chart.js on a logarithmic axis.
    running = []
    acc = 1
    for _, val, _ in factors:
        acc *= val
        running.append(acc / 1e6)

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
          tooltip:{{callbacks:{{label: ctx => {{
            const y = ctx.parsed.y;
            return (y >= 0.01 ? y.toLocaleString(undefined, {{maximumFractionDigits: 3}})
                              : y.toExponential(2)) + ' MB';
          }}}}}}
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
    rng = _rng(42 + head_idx)
    true_importance = np.zeros(seq_len, dtype=float)
    heavy_positions = rng.integers(0, max(1, seq_len - window_size), size=5)
    true_importance[heavy_positions] = rng.random(5) * 0.5 + 0.5
    true_importance = true_importance + rng.random(seq_len) * 0.1
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

    rng = _rng(7)
    head_patterns = []
    for h in range(n_heads):
        if h < 2:
            w = np.exp(-np.arange(T, dtype=float) * 0.15)[::-1]
        elif h < 4:
            w = np.zeros(T, dtype=float)
            idx = rng.permutation(T)[:max(1, T // 5)]
            w[idx] = rng.random(max(1, T // 5))
        else:
            w = rng.random(T)
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
    rng = _rng(11)
    prefix_len = seq_len - window_size
    raw_scores = rng.random(prefix_len)
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
    rng = _rng(3)
    entropies, fixed_w, adaptive_w = [], [], []
    base_window = 16

    for h in range(n_heads):
        logits = rng.standard_normal(seq_len)
        attn = _softmax(logits)
        entropy = float(-(attn * np.log(attn + 1e-9)).sum())
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
    rng = _rng(5)
    base = rng.standard_normal(n_positions)
    focused = _softmax(base / max(1e-3, focus_temp))
    diffuse = _softmax(base / max(1e-3, diffuse_temp))

    def H(p):
        return float(-(p * np.log(p + 1e-12)).sum())

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
    rng = _rng(99)
    importance = rng.random(T)
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
        scores = importance.copy()
        scores[-n_recent:] = 2.0
        idx = _topk_indices(scores, n_keep)
        kept = set(idx.tolist())
        obs_set = set(range(T - n_recent, T))
    else:  # SnapKV
        obs_w = max(1, min(int(T * 0.25), 16))
        obs_set = set(range(T - obs_w, T))
        prefix_scores = importance[:-obs_w]
        n_prefix = max(1, n_keep - obs_w)
        idx = _topk_indices(prefix_scores, min(n_prefix, len(prefix_scores)))
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


# ── Internal: shared scoring helper used by GAME 3 and GAME 4 ────────────────

def _score_methods(T: int, importance: np.ndarray, n_keep: int, method: str):
    """
    Returns (kept_set, obs_set) for a method on a sequence of length T.
    Same logic as run_demo, factored out so other games can reuse it.
    """
    if method == "Full Cache":
        return set(range(T)), set()
    if method == "Recent Only":
        return set(range(T - n_keep, T)), set()
    if method == "Random":
        import random
        random.seed(42)
        return set(random.sample(range(T), min(n_keep, T))), set()
    if method == "StreamingLLM":
        n_sink = max(1, min(4, T // 10))
        n_recent = max(1, n_keep - n_sink)
        return set(range(n_sink)) | set(range(T - n_recent, T)), set(range(T - n_recent, T))
    if method == "H2O":
        n_recent = max(1, n_keep // 4)
        scores = importance.copy()
        scores[-n_recent:] = 2.0
        idx = _topk_indices(scores, min(n_keep, T))
        return set(idx.tolist()), set(range(T - n_recent, T))
    # SnapKV
    obs_w = max(1, min(int(T * 0.25), 16))
    obs_set = set(range(T - obs_w, T))
    prefix_scores = importance[:-obs_w] if obs_w < T else importance.copy()
    n_prefix = max(1, n_keep - obs_w)
    idx = _topk_indices(prefix_scores, min(n_prefix, len(prefix_scores)))
    return set(idx.tolist()) | obs_set, obs_set


# ── GAME 3. Needle in a haystack ─────────────────────────────────────────────

def run_needle_demo(needle_position: str = "middle", budget: float = 0.30,
                    haystack_size: int = 60):
    """
    Hide one important fact in a sea of filler tokens, then run every policy
    and show which ones preserved the needle.
    """
    filler = ["the", "and", "of", "in", "to", "a", "is", "that", "for", "it",
              "on", "as", "with", "at", "by", "an", "or", "this", "from", "but"]
    tokens = [filler[i % len(filler)] for i in range(haystack_size)]

    if needle_position == "early":
        needle_idx = max(2, haystack_size // 8)
    elif needle_position == "late":
        needle_idx = haystack_size - max(2, haystack_size // 8)
    else:
        needle_idx = haystack_size // 2

    needle_text = "★capital=Lima★"
    tokens[needle_idx] = needle_text
    question = ["What", "is", "the", "capital?"]
    tokens.extend(question)

    T = len(tokens)
    n_keep = max(2, int(T * budget))

    # Synthetic importance: needle gets a moderate boost (model usually learns
    # the keyword pattern), recent tokens get a small one. SnapKV's window
    # attention will *also* boost the needle if the question keywords correlate.
    rng = _rng(123 + needle_idx)
    importance = rng.random(T) * 0.3
    importance[needle_idx] = 0.95
    importance[-4:] += 0.4
    importance = importance / importance.sum()

    methods = ["SnapKV", "H2O", "StreamingLLM", "Recent Only", "Random"]
    rows_html = []
    for m in methods:
        kept, _ = _score_methods(T, importance, n_keep, m)
        survived = needle_idx in kept
        badge_color = KEPT_COLOR if survived else EVICTED_COLOR
        badge_text = "✓ needle kept" if survived else "✗ needle lost"

        spans = []
        for i, tok in enumerate(tokens):
            is_needle = i == needle_idx
            in_kept = i in kept
            if is_needle and in_kept:
                bg, border = "#FFD66B", "#BA7517"
            elif is_needle and not in_kept:
                bg, border = "#F5C4B3", "#D85A30"
            elif in_kept:
                bg, border = "#9FE1CB", "#1D9E75"
            else:
                bg, border = "#EEECE5", "#B4B2A9"
            weight = "600" if is_needle else "400"
            spans.append(
                f'<span style="display:inline-block;margin:1px;padding:2px 5px;'
                f'background:{bg};border:1px solid {border};border-radius:3px;'
                f'font-size:10px;font-family:var(--font-mono);font-weight:{weight}">{tok}</span>'
            )

        rows_html.append(f"""
        <div style="margin-bottom:14px">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <div style="font-weight:600;font-size:13px;min-width:90px">{m}</div>
            <div style="background:{badge_color}22;border:1px solid {badge_color};
                        color:{badge_color};border-radius:12px;padding:2px 10px;
                        font-size:11px;font-weight:500">{badge_text}</div>
            <div style="color:var(--color-text-secondary);font-size:11px">
              {len(kept)} / {T} tokens kept
            </div>
          </div>
          <div style="background:var(--color-background-secondary);border-radius:6px;
                      padding:8px;line-height:1.8">{''.join(spans)}</div>
        </div>
        """)

    legend = """
    <div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:10px">
      <span style="background:#FFD66B;border:1px solid #BA7517;border-radius:3px;padding:1px 6px;font-weight:600">★ needle kept</span>
      &nbsp;<span style="background:#F5C4B3;border:1px solid #D85A30;border-radius:3px;padding:1px 6px;font-weight:600">★ needle lost</span>
      &nbsp;<span style="background:#9FE1CB;border:1px solid #1D9E75;border-radius:3px;padding:1px 6px">other kept</span>
      &nbsp;<span style="background:#EEECE5;border:1px solid #B4B2A9;border-radius:3px;padding:1px 6px">evicted</span>
    </div>
    """
    intro = f"""
    <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:8px;line-height:1.6">
      Hidden fact at position <strong>{needle_idx}</strong>
      ({needle_position}) in a {haystack_size}-token haystack.
      Question is appended at the end. Budget = <strong>{int(budget*100)}%</strong>
      ({n_keep} of {T} tokens).
    </div>
    """
    return mo.Html(intro + legend + "".join(rows_html))


# ── GAME 4. Build your own memory policy ─────────────────────────────────────

def run_custom_policy(prompt: str, budget: float, recency_w: float,
                      frequency_w: float, attention_w: float):
    """
    Score every token by a user-weighted combination of three signals:
      - recency:  how close to the end the token is
      - frequency: how often the token (or stem) repeats
      - attention: simulated obs.-window attention (the SnapKV signal)
    Compare the user's selection to SnapKV's selection on the same prompt.
    """
    tokens = prompt.split()
    if not tokens:
        return mo.md("*Enter a prompt above.*")

    T = len(tokens)
    n_keep = max(1, int(T * budget))

    # Signal 1: recency — linearly increasing toward the end
    recency = np.linspace(0.0, 1.0, T)

    # Signal 2: frequency — count occurrences of each token (case-insensitive)
    counts = {}
    for t in tokens:
        k = t.lower().strip(".,!?:;")
        counts[k] = counts.get(k, 0) + 1
    max_count = max(counts.values())
    frequency = np.array(
        [counts[t.lower().strip(".,!?:;")] / max_count for t in tokens]
    )

    # Signal 3: simulated obs-window attention (same source as run_demo)
    rng = _rng(99)
    attention = rng.random(T)
    attention[0:3] += 0.4
    attention[-4:] += 0.3
    attention = attention / attention.max()

    total_w = recency_w + frequency_w + attention_w + 1e-9
    rw, fw, aw = recency_w / total_w, frequency_w / total_w, attention_w / total_w
    custom_score = rw * recency + fw * frequency + aw * attention

    custom_idx = _topk_indices(custom_score, n_keep)
    custom_kept = set(custom_idx.tolist())

    # SnapKV reference selection on the same prompt and budget
    snapkv_kept, _ = _score_methods(T, attention, n_keep, "SnapKV")

    overlap = len(custom_kept & snapkv_kept)
    overlap_pct = overlap / max(1, len(snapkv_kept))

    def _strip(kept_set):
        spans = []
        for i, tok in enumerate(tokens):
            if i in kept_set:
                bg, border = "#9FE1CB", "#1D9E75"
            else:
                bg, border = "#F5C4B3", "#D85A30"
            spans.append(
                f'<span style="display:inline-block;margin:2px 1px;padding:3px 6px;'
                f'background:{bg};border:1px solid {border};border-radius:4px;'
                f'font-size:12px;font-family:var(--font-mono)">{tok}</span>'
            )
        return "".join(spans)

    return mo.Html(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Tokens kept</div>
        <div style="font-size:16px;font-weight:500;color:{KEPT_COLOR}">{n_keep} / {T}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Weights (norm.)</div>
        <div style="font-size:13px;font-weight:500">R {rw:.2f} · F {fw:.2f} · A {aw:.2f}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Overlap with SnapKV</div>
        <div style="font-size:16px;font-weight:500;color:{WINDOW_COLOR}">{overlap_pct:.0%} ({overlap}/{len(snapkv_kept)})</div>
      </div>
    </div>

    <div style="margin-bottom:6px;font-size:12px;font-weight:600;color:var(--color-text-primary)">
      Your custom policy
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;line-height:1.9;margin-bottom:14px">
      {_strip(custom_kept)}
    </div>

    <div style="margin-bottom:6px;font-size:12px;font-weight:600;color:var(--color-text-primary)">
      SnapKV (same prompt, same budget)
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;line-height:1.9">
      {_strip(snapkv_kept)}
    </div>

    <div style="margin-top:10px;padding:10px 14px;background:#534AB715;border-left:3px solid {WINDOW_COLOR};
         border-radius:0 6px 6px 0;font-size:12px;color:var(--color-text-secondary);line-height:1.6">
      Crank <strong>attention</strong> to 1 and the others to 0 → your policy converges to SnapKV.
      Crank <strong>recency</strong> alone → you reproduce StreamingLLM's recent-window behaviour.
      Crank <strong>frequency</strong> alone → you get a "common-words" baseline (rarely useful on its own).
    </div>
    """)


# ── Two axes: sparse attention vs cache eviction ─────────────────────────────

def plot_two_axes():
    """
    Side-by-side comparison panel: the two complementary axes for taming
    long-context costs. HTML only — no chart needed, the contrast IS the visual.
    """
    return mo.Html(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
      <div style="background:var(--color-background-secondary);border-left:3px solid {WINDOW_COLOR};
                  border-radius:0 8px 8px 0;padding:14px 16px">
        <div style="font-size:11px;color:{WINDOW_COLOR};font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:6px">Axis 1 · Sparse attention</div>
        <div style="font-size:14px;font-weight:500;margin-bottom:8px">Reduce <em>compute</em></div>
        <div style="font-size:12px;color:var(--color-text-secondary);line-height:1.6">
          Change the attention math itself so each token only attends to a
          subset of others — sliding windows, global tokens, blocks, random
          patterns. The full KV cache is still built; you just don't visit
          all of it.
          <br><br>
          <strong style="color:var(--color-text-primary)">Examples:</strong>
          Longformer, BigBird, sliding-window attention, block-sparse attention.
          <br><br>
          <strong style="color:var(--color-text-primary)">Cost cut:</strong>
          attention compute drops from O(T²) toward O(T·w).
        </div>
      </div>

      <div style="background:var(--color-background-secondary);border-left:3px solid {KEPT_COLOR};
                  border-radius:0 8px 8px 0;padding:14px 16px">
        <div style="font-size:11px;color:{KEPT_COLOR};font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:6px">Axis 2 · KV cache eviction</div>
        <div style="font-size:14px;font-weight:500;margin-bottom:8px">Reduce <em>memory</em></div>
        <div style="font-size:12px;color:var(--color-text-secondary);line-height:1.6">
          Keep the attention math intact. After computing K and V for every
          token, throw away the entries that aren't worth holding onto. The
          model still sees full attention during prefill — it just decodes
          against a slimmer cache.
          <br><br>
          <strong style="color:var(--color-text-primary)">Examples:</strong>
          SnapKV, H2O, StreamingLLM, Scissorhands, Ada-KV.
          <br><br>
          <strong style="color:var(--color-text-primary)">Cost cut:</strong>
          KV memory drops from O(T) toward O(k) per layer per head.
        </div>
      </div>
    </div>

    <div style="margin-top:14px;padding:12px 16px;background:#BA751715;border-left:3px solid #BA7517;
         border-radius:0 6px 6px 0;font-size:13px;color:var(--color-text-secondary);line-height:1.7">
      <strong style="color:var(--color-text-primary)">They're complementary, not competing.</strong>
      Sparse attention and KV eviction can stack: a sliding-window model can
      <em>also</em> compress its sliding cache with SnapKV-style voting. SnapKV
      itself is appealing precisely because it requires <em>no</em> changes to
      the underlying attention — it's a drop-in for any standard transformer.
    </div>
    """)


# ── Competitors: comparison table, capability matrix, method picker ──────────
#
# Nine KV-cache compression methods that compete with or extend SnapKV.
# Single source of truth — used by all three competitor visualizations below.
#
# Fields per method:
#   year     publication year
#   venue    venue label (short)
#   mech     one-line core mechanism
#   budget   budget granularity
#   latest   True for 2025 papers (D2O, CAKE, SCOPE)
#   anchor   True for SnapKV (the notebook's anchor paper)
#   caps     dict of capabilities — values: 1 (yes), 0.5 (partial), 0 (no)
#            keys: per_head, per_layer, adaptive, decode_aware, recovery
#   fit      dict of fitness scores per scenario axis (1.0 best, 0 worst)

_COMPETITORS = [
    {
        "name": "H2O", "year": 2023, "venue": "NeurIPS '23",
        "mech": "Recent tokens + attention heavy hitters",
        "budget": "Fixed",
        "latest": False, "anchor": False,
        "caps": {"per_head": 0,   "per_layer": 0,   "adaptive": 0.5, "decode_aware": 1, "recovery": 0},
        "fit":  {"long_context": 0.6, "streaming": 0.7, "long_gen": 0.7, "drop_in": 0.9, "recovery": 0},
    },
    {
        "name": "Scissorhands", "year": 2023, "venue": "NeurIPS '23",
        "mech": "Persistent pivotal tokens — no finetuning",
        "budget": "Fixed",
        "latest": False, "anchor": False,
        "caps": {"per_head": 0,   "per_layer": 0,   "adaptive": 0.5, "decode_aware": 1, "recovery": 0},
        "fit":  {"long_context": 0.6, "streaming": 0.6, "long_gen": 0.7, "drop_in": 0.8, "recovery": 0},
    },
    {
        "name": "StreamingLLM", "year": 2024, "venue": "ICLR '24",
        "mech": "Attention sinks + sliding recent window",
        "budget": "Fixed",
        "latest": False, "anchor": False,
        "caps": {"per_head": 0,   "per_layer": 0,   "adaptive": 0,   "decode_aware": 1, "recovery": 0},
        "fit":  {"long_context": 0.7, "streaming": 1.0, "long_gen": 0.6, "drop_in": 1.0, "recovery": 0},
    },
    {
        "name": "FastGen", "year": 2024, "venue": "ICLR '24",
        "mech": "Per-module attention profiling",
        "budget": "Adaptive",
        "latest": False, "anchor": False,
        "caps": {"per_head": 1,   "per_layer": 1,   "adaptive": 1,   "decode_aware": 0.5, "recovery": 0},
        "fit":  {"long_context": 0.7, "streaming": 0.6, "long_gen": 0.7, "drop_in": 0.6, "recovery": 0},
    },
    {
        "name": "SnapKV", "year": 2024, "venue": "NeurIPS '24",
        "mech": "Obs-window voting + max-pool clustering",
        "budget": "Per-head fixed",
        "latest": False, "anchor": True,
        "caps": {"per_head": 1,   "per_layer": 0,   "adaptive": 0.5, "decode_aware": 0.5, "recovery": 0},
        "fit":  {"long_context": 0.85, "streaming": 0.7, "long_gen": 0.8, "drop_in": 1.0, "recovery": 0},
    },
    {
        "name": "PyramidKV", "year": 2024, "venue": "—",
        "mech": "Layer-wise budget allocation",
        "budget": "Per-layer",
        "latest": False, "anchor": False,
        "caps": {"per_head": 0.5, "per_layer": 1,   "adaptive": 1,   "decode_aware": 0.5, "recovery": 0},
        "fit":  {"long_context": 0.9,  "streaming": 0.5, "long_gen": 0.7, "drop_in": 0.6, "recovery": 0},
    },
    {
        "name": "NACL", "year": 2024, "venue": "ACL '24",
        "mech": "One-shot prefill eviction (proxy + random)",
        "budget": "Fixed",
        "latest": False, "anchor": False,
        "caps": {"per_head": 0,   "per_layer": 0,   "adaptive": 0,   "decode_aware": 0,   "recovery": 0},
        "fit":  {"long_context": 0.7, "streaming": 0.5, "long_gen": 0.5, "drop_in": 0.9, "recovery": 0},
    },
    {
        "name": "D2O", "year": 2025, "venue": "ICLR '25",
        "mech": "Layer-aware budget + recall/merge of evicted",
        "budget": "Per-layer dynamic",
        "latest": True, "anchor": False,
        "caps": {"per_head": 0.5, "per_layer": 1,   "adaptive": 1,   "decode_aware": 1,   "recovery": 1},
        "fit":  {"long_context": 1.0, "streaming": 0.6, "long_gen": 0.9, "drop_in": 0.5, "recovery": 1.0},
    },
    {
        "name": "CAKE", "year": 2025, "venue": "ICLR '25",
        "mech": "Cascading adaptive eviction + layer prefs",
        "budget": "Per-layer",
        "latest": True, "anchor": False,
        "caps": {"per_head": 0.5, "per_layer": 1,   "adaptive": 1,   "decode_aware": 1,   "recovery": 0.5},
        "fit":  {"long_context": 1.0, "streaming": 0.6, "long_gen": 0.9, "drop_in": 0.5, "recovery": 0.6},
    },
    {
        "name": "SCOPE", "year": 2025, "venue": "ACL '25",
        "mech": "Separate prefill vs decoding compression",
        "budget": "Phase-aware",
        "latest": True, "anchor": False,
        "caps": {"per_head": 0.5, "per_layer": 0.5, "adaptive": 1,   "decode_aware": 1,   "recovery": 0.5},
        "fit":  {"long_context": 0.8, "streaming": 0.7, "long_gen": 1.0, "drop_in": 0.6, "recovery": 0.5},
    },
]


def plot_competitors_table():
    """Comparison table of the 9 competitor methods (SnapKV row highlighted)."""
    LATEST_BG, LATEST_BORDER = "#1D9E7522", "#1D9E75"
    ANCHOR_BG = "#534AB715"

    rows = ""
    for m in _COMPETITORS:
        row_bg = ANCHOR_BG if m["anchor"] else "transparent"
        anchor_label = (
            f' <span style="background:#534AB7;color:white;border-radius:3px;'
            f'padding:1px 6px;font-size:9px;letter-spacing:.05em;margin-left:6px;'
            f'vertical-align:middle">ANCHOR</span>' if m["anchor"] else ""
        )
        latest_label = (
            f' <span style="background:{LATEST_BG};color:{LATEST_BORDER};'
            f'border:1px solid {LATEST_BORDER};border-radius:10px;padding:1px 8px;'
            f'font-size:10px;font-weight:600;letter-spacing:.04em;margin-left:6px;'
            f'vertical-align:middle">LATEST · 2025</span>' if m["latest"] else ""
        )
        rows += f"""
        <tr style="background:{row_bg}">
          <td style="padding:9px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:middle;font-size:13px;font-weight:600;
                     color:var(--color-text-primary)">{m['name']}{anchor_label}{latest_label}</td>
          <td style="padding:9px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:middle;font-size:11px;color:var(--color-text-secondary);
                     font-family:var(--font-mono)">{m['venue']}</td>
          <td style="padding:9px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:middle;font-size:12px;color:var(--color-text-secondary);
                     line-height:1.5">{m['mech']}</td>
          <td style="padding:9px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:middle">
            <span style="background:#BA751722;color:#BA7517;border:1px solid #BA7517;
                         border-radius:10px;padding:2px 8px;font-size:11px;font-weight:500">
              {m['budget']}
            </span>
          </td>
        </tr>
        """

    return mo.Html(f"""
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:10px;line-height:1.6">
      Nine methods in the same family as SnapKV. The
      <span style="color:#534AB7;font-weight:600">anchor row</span> is SnapKV;
      the <span style="color:{LATEST_BORDER};font-weight:600">LATEST · 2025</span>
      badges mark the freshest work (D2O, CAKE, SCOPE) — all push toward
      per-layer dynamic budgets and recoverability.
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;
                padding:4px 10px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Method</th>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Venue</th>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Core mechanism</th>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Budget</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """)


def plot_capability_matrix():
    """Methods × capabilities matrix with color-coded cells."""
    cap_labels = {
        "per_head":     ("Per-head budget",   "Each attention head gets its own kept set"),
        "per_layer":    ("Per-layer budget",  "Different layers get different cache sizes"),
        "adaptive":     ("Adaptive sizing",   "Budget responds to attention statistics"),
        "decode_aware": ("Decode-aware",      "Maintains quality through long generation"),
        "recovery":     ("Recovery",          "Can resurrect / merge evicted entries"),
    }

    def cell(value: float) -> str:
        if value >= 0.99:
            color, dot, label = "#1D9E75", "●", "yes"
        elif value >= 0.5:
            color, dot, label = "#BA7517", "◐", "partial"
        else:
            color, dot, label = "#88878055", "○", "no"
        return (
            f'<td title="{label}" style="text-align:center;padding:8px 6px;'
            f'border-bottom:1px solid var(--color-border, #e5e5e5);'
            f'font-size:18px;color:{color};line-height:1">{dot}</td>'
        )

    head_cells = "".join(
        f'<th title="{tooltip}" style="text-align:center;padding:8px 6px;font-size:10px;'
        f'text-transform:uppercase;letter-spacing:.05em;color:var(--color-text-secondary);'
        f'font-weight:600;line-height:1.3">{label}</th>'
        for _, (label, tooltip) in cap_labels.items()
    )

    body = ""
    for m in _COMPETITORS:
        anchor_style = "background:#534AB715;font-weight:600" if m["anchor"] else ""
        latest = (
            ' <span style="background:#1D9E7522;color:#1D9E75;border:1px solid #1D9E75;'
            'border-radius:8px;padding:0 6px;font-size:9px;font-weight:600">2025</span>'
            if m["latest"] else ""
        )
        cells = "".join(cell(m["caps"][k]) for k in cap_labels)
        body += f"""
        <tr style="{anchor_style}">
          <td style="padding:8px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     font-size:13px;color:var(--color-text-primary);font-weight:500">
            {m['name']}{latest}
          </td>
          {cells}
        </tr>
        """

    return mo.Html(f"""
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:10px;line-height:1.6">
      Five capabilities that differentiate these methods.
      <span style="color:#1D9E75;font-weight:600">●</span> supported,
      <span style="color:#BA7517;font-weight:600">◐</span> partial,
      <span style="color:#88878088;font-weight:600">○</span> not supported.
      Hover any cell for a label.
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;
                padding:4px 10px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Method</th>
            {head_cells}
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    <div style="margin-top:12px;padding:10px 14px;background:#534AB715;border-left:3px solid #534AB7;
         border-radius:0 6px 6px 0;font-size:12px;color:var(--color-text-secondary);line-height:1.6">
      The 2025 wave (D2O, CAKE, SCOPE) is filling the right-hand columns —
      <strong style="color:var(--color-text-primary)">decode-aware</strong> and
      <strong style="color:var(--color-text-primary)">recovery</strong> were
      mostly empty even a year ago.
    </div>
    """)


def run_method_picker(context_length: str, workload: str,
                      drop_in_required: bool, recovery_needed: bool):
    """
    Score every method against the user's scenario and recommend the top fits.
    Pure ranking — no ML, just fitness lookups defined per method above.
    """
    ctx_to_axis = {
        "Short (< 8K)":       0.5,
        "Medium (8–32K)":     0.7,
        "Long (32–128K)":     0.95,
        "Very long (> 128K)": 1.0,
    }
    workload_to_axis = {
        "Single-shot Q&A":     "long_context",
        "Streaming chat":      "streaming",
        "Long generation / reasoning": "long_gen",
    }

    ctx_weight = ctx_to_axis.get(context_length, 0.7)
    workload_axis = workload_to_axis.get(workload, "long_context")
    # Keep long-context fit relevant even for short prompts, but don't hand
    # every method the same large baseline that washes out differences.
    context_floor = 0.50

    scored = []
    for m in _COMPETITORS:
        score = 0.0
        score += 0.40 * (m["fit"][workload_axis])
        score += 0.25 * (
            m["fit"]["long_context"] * ctx_weight
            + (1 - ctx_weight) * context_floor
        )
        if drop_in_required:
            # When "drop-in" is a hard requirement, it should outweigh the
            # small freshness bonus and meaningfully counterbalance methods
            # that are stronger overall but require more integration work.
            score += 0.30 * m["fit"]["drop_in"]
        else:
            score += 0.10 * m["fit"]["drop_in"]  # mild penalty for clunky setup either way
        if recovery_needed:
            score += 0.30 * m["fit"]["recovery"]
        # Small boost for being recent (signals active maintenance)
        if m["latest"]:
            score += 0.02
        scored.append((round(score, 3), m))

    scored.sort(key=lambda t: t[0], reverse=True)
    top3 = scored[:3]
    max_score = top3[0][0] or 1.0

    cards = ""
    medals = ["#1D9E75", "#BA7517", "#534AB7"]
    medal_labels = ["BEST FIT", "RUNNER-UP", "ALSO WORTH TRYING"]
    for i, (score, m) in enumerate(top3):
        bar_pct = int(100 * score / max_score)
        latest = (
            ' <span style="background:#1D9E7522;color:#1D9E75;border:1px solid #1D9E75;'
            'border-radius:8px;padding:0 6px;font-size:10px;font-weight:600">LATEST</span>'
            if m["latest"] else ""
        )
        cards += f"""
        <div style="border-left:4px solid {medals[i]};padding:12px 14px;
                    background:var(--color-background-secondary);border-radius:0 8px 8px 0;
                    margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
            <div>
              <span style="font-size:10px;font-weight:600;color:{medals[i]};
                           letter-spacing:.06em">{medal_labels[i]}</span>
              <span style="font-size:15px;font-weight:600;margin-left:8px;
                           color:var(--color-text-primary)">{m['name']}</span>
              {latest}
              <span style="font-size:11px;color:var(--color-text-secondary);
                           font-family:var(--font-mono);margin-left:6px">{m['venue']}</span>
            </div>
            <div style="font-size:11px;color:var(--color-text-secondary)">score {score:.2f}</div>
          </div>
          <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:6px">
            {m['mech']} &nbsp;·&nbsp;
            <span style="color:#BA7517;font-weight:500">{m['budget']}</span>
          </div>
          <div style="background:var(--color-background-primary, #fff);height:6px;border-radius:3px;
                      overflow:hidden">
            <div style="width:{bar_pct}%;height:100%;background:{medals[i]};
                        border-radius:3px"></div>
          </div>
        </div>
        """

    full_table = ""
    for score, m in scored:
        bar_pct = int(100 * score / max_score)
        full_table += f"""
        <tr>
          <td style="padding:6px 10px;font-size:12px;color:var(--color-text-primary);
                     border-bottom:1px solid var(--color-border, #e5e5e5)">{m['name']}</td>
          <td style="padding:6px 10px;font-size:11px;color:var(--color-text-secondary);
                     border-bottom:1px solid var(--color-border, #e5e5e5);width:60%">
            <div style="background:var(--color-background-primary, #fff);height:5px;
                        border-radius:3px;overflow:hidden">
              <div style="width:{bar_pct}%;height:100%;background:#888780;border-radius:3px"></div>
            </div>
          </td>
          <td style="padding:6px 10px;font-size:11px;color:var(--color-text-secondary);
                     border-bottom:1px solid var(--color-border, #e5e5e5);text-align:right;
                     font-family:var(--font-mono)">{score:.2f}</td>
        </tr>
        """

    return mo.Html(f"""
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:12px">
      Scenario: <strong>{context_length}</strong> context ·
      <strong>{workload}</strong> ·
      drop-in {"required" if drop_in_required else "optional"} ·
      recovery {"needed" if recovery_needed else "not needed"}
    </div>
    {cards}
    <details style="margin-top:8px">
      <summary style="cursor:pointer;font-size:12px;color:var(--color-text-secondary)">
        Full ranking (all {len(scored)} methods)
      </summary>
      <div style="background:var(--color-background-secondary);border-radius:8px;
                  padding:4px 10px;margin-top:8px">
        <table style="width:100%;border-collapse:collapse">{full_table}</table>
      </div>
    </details>
    """)


# ── Agentic memory: 3-tier hierarchy panel ───────────────────────────────────

def plot_memory_hierarchy():
    """
    Three-column panel: hot / warm / cold memory tiers, with parallels
    between LLM-internal memory and agent-system memory.
    """
    HOT, WARM, COLD = "#D85A30", "#BA7517", "#534AB7"
    return mo.Html(f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
      <div style="background:var(--color-background-secondary);border-top:4px solid {HOT};
                  border-radius:6px;padding:14px 16px">
        <div style="font-size:10px;color:{HOT};font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px">Hot tier</div>
        <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:var(--color-text-primary)">
          fast · expensive · small
        </div>
        <div style="font-size:12px;color:var(--color-text-secondary);line-height:1.6">
          <strong style="color:var(--color-text-primary)">In an LLM:</strong>
          the KV cache. Every generation step touches it. Bytes per token are
          enormous; size is bounded by GPU memory.<br><br>
          <strong style="color:var(--color-text-primary)">In an agent:</strong>
          the live context window. Whatever the model can see <em>right now</em>
          while planning the next action.
        </div>
      </div>

      <div style="background:var(--color-background-secondary);border-top:4px solid {WARM};
                  border-radius:6px;padding:14px 16px">
        <div style="font-size:10px;color:{WARM};font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px">Warm tier</div>
        <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:var(--color-text-primary)">
          medium · cheaper · larger
        </div>
        <div style="font-size:12px;color:var(--color-text-secondary);line-height:1.6">
          <strong style="color:var(--color-text-primary)">In an LLM:</strong>
          the prompt text itself, sitting in CPU/RAM, ready to be re-prefilled
          if needed.<br><br>
          <strong style="color:var(--color-text-primary)">In an agent:</strong>
          a scratchpad, working notes, recent tool outputs — text the agent
          can pull back into context cheaply on the next turn.
        </div>
      </div>

      <div style="background:var(--color-background-secondary);border-top:4px solid {COLD};
                  border-radius:6px;padding:14px 16px">
        <div style="font-size:10px;color:{COLD};font-weight:600;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px">Cold tier</div>
        <div style="font-size:14px;font-weight:600;margin-bottom:8px;color:var(--color-text-primary)">
          slow · cheap · ~unbounded
        </div>
        <div style="font-size:12px;color:var(--color-text-secondary);line-height:1.6">
          <strong style="color:var(--color-text-primary)">In an LLM:</strong>
          retrieval (RAG). Not really "the model's memory" — it's a database
          the model queries on demand.<br><br>
          <strong style="color:var(--color-text-primary)">In an agent:</strong>
          long-term memory store: vector DB, episodic logs, learned skills,
          summarised past sessions (MemGPT, Letta, Claude memory).
        </div>
      </div>
    </div>

    <div style="margin-top:14px;padding:12px 16px;background:#1D9E7515;border-left:3px solid {KEPT_COLOR};
         border-radius:0 6px 6px 0;font-size:13px;color:var(--color-text-secondary);line-height:1.7">
      <strong style="color:var(--color-text-primary)">The same question, three scales.</strong>
      SnapKV asks "what should I keep in the hot tier?" using the model's own
      attention as the signal. Agents face the identical question one level up:
      "what should I keep in context, what should I summarise, what should I
      drop?" The answer-shape is the same — <em>recent + currently relevant</em> —
      and the design knobs (budget, observation window, importance signal) all
      have direct agent counterparts.
    </div>
    """)


def plot_memory_types():
    """
    Five canonical memory types and how KV cache management does (or doesn't)
    affect each of them. The point: KV cache is the working-memory layer —
    every other type only matters when it's surfaced through it.
    """
    # impact levels: "direct" (green), "gateway" (orange), "none" (gray)
    rows = [
        {
            "name": "Working",
            "icon": "",
            "what": "What the model is actively manipulating right now to produce the next token.",
            "where": "KV cache (in GPU)",
            "where_color": "#D85A30",
            "impact": "direct",
            "impact_text": "★ This <em>is</em> the KV cache. SnapKV optimises here.",
        },
        {
            "name": "Short-term",
            "icon": "",
            "what": "Recent turns of the conversation — still in the prompt the agent sees.",
            "where": "Context window → KV cache",
            "where_color": "#BA7517",
            "impact": "direct",
            "impact_text": "Cache size sets a hard ceiling on how much of this stays addressable.",
        },
        {
            "name": "Episodic",
            "icon": "",
            "what": "Specific past events: previous sessions, what the user said yesterday, prior tool calls.",
            "where": "External store · vector DB · summary log",
            "where_color": "#534AB7",
            "impact": "gateway",
            "impact_text": "Indirect. Only matters once retrieved into context — then it's working memory again.",
        },
        {
            "name": "Semantic",
            "icon": "",
            "what": "World knowledge and facts the model already learned during training.",
            "where": "Model weights (parametric)",
            "where_color": "#888780",
            "impact": "none",
            "impact_text": "Separate substrate. KV cache compression cannot help or hurt it.",
        },
        {
            "name": "Procedural",
            "icon": "",
            "what": "Skills and how-to: tool use, output formats, reasoning patterns.",
            "where": "Model weights + tool definitions in prompt",
            "where_color": "#888780",
            "impact": "gateway",
            "impact_text": "Skills live in weights; tool definitions live in context — those <em>do</em> share the cache.",
        },
    ]

    impact_styles = {
        "direct":  ("#1D9E75", "#1D9E7515"),
        "gateway": ("#BA7517", "#BA751715"),
        "none":    ("#888780", "#88878015"),
    }

    body = ""
    for r in rows:
        border, bg = impact_styles[r["impact"]]
        body += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:top;width:120px">
            <div style="font-size:18px;line-height:1">{r['icon']}</div>
            <div style="font-size:13px;font-weight:600;margin-top:2px;color:var(--color-text-primary)">
              {r['name']}
            </div>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     font-size:12px;color:var(--color-text-secondary);vertical-align:top;line-height:1.55">
            {r['what']}
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:top;width:200px">
            <span style="display:inline-block;background:{r['where_color']}22;
                         color:{r['where_color']};border:1px solid {r['where_color']};
                         border-radius:10px;padding:2px 8px;font-size:11px;font-weight:500">
              {r['where']}
            </span>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     vertical-align:top;background:{bg};border-left:3px solid {border};
                     font-size:12px;color:var(--color-text-secondary);line-height:1.55">
            {r['impact_text']}
          </td>
        </tr>
        """

    return mo.Html(f"""
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:10px;line-height:1.6">
      Cognitive science distinguishes several memory types. Map them onto an
      LLM agent and a clear picture appears: <strong style="color:var(--color-text-primary)">KV
      cache is the gateway</strong> — anything from any memory type, to shape
      the next token, must pass through it.
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;
                padding:4px 10px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Type</th>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">In an agent</th>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Where it lives</th>
            <th style="text-align:left;padding:8px 12px;font-size:10px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">KV cache role</th>
          </tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    <div style="margin-top:12px;padding:10px 14px;background:#1D9E7515;border-left:3px solid #1D9E75;
         border-radius:0 6px 6px 0;font-size:12px;color:var(--color-text-secondary);line-height:1.6">
      <strong style="color:var(--color-text-primary)">The bottleneck principle:</strong>
      a great long-term memory store doesn't help if the agent can't fit the
      retrieved chunk into context. A clever KV-cache policy is what lets the
      gateway carry richer episodic and procedural content per turn.
      That's why working-memory optimisation (SnapKV and friends) compounds
      with every other memory layer you add.
    </div>
    """)


def plot_agent_mapping():
    """Side-by-side mapping table: SnapKV concept ↔ agent memory concept."""
    pairs = [
        ("KV cache",                      "Conversation context window"),
        ("Observation window (last w tokens)", "Current user message · current goal"),
        ("Per-token attention voting",    "Per-turn relevance scoring"),
        ("Top-k token selection",         "Which past turns to keep verbatim"),
        ("Max-pool clustering",           "Keeping local context around an important turn"),
        ("Per-head budget",               "Per-tool / per-skill memory budget"),
        ("Adaptive observation window",   "Adaptive recall depth based on task uncertainty"),
        ("Eviction (drop entry)",         "Drop or summarise an old turn"),
    ]
    rows = "".join(
        f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     font-size:13px;font-family:var(--font-mono);color:#1D9E75">{a}</td>
          <td style="padding:8px 12px;border-bottom:1px solid var(--color-border, #e5e5e5);
                     font-size:13px;color:var(--color-text-primary)">→ &nbsp;{b}</td>
        </tr>
        """
        for a, b in pairs
    )
    return mo.Html(f"""
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:8px">
      Every SnapKV design knob has a direct counterpart in an agent's memory loop.
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;
                padding:6px 10px;overflow-x:auto">
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr>
            <th style="text-align:left;padding:8px 12px;font-size:11px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">SnapKV (tokens)</th>
            <th style="text-align:left;padding:8px 12px;font-size:11px;
                       text-transform:uppercase;letter-spacing:.06em;
                       color:var(--color-text-secondary);font-weight:600">Agent memory (turns)</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """)


# ── GAME 5. Agent memory over turns ──────────────────────────────────────────

def simulate_agent_loop(n_turns: int = 12, strategy: str = "Agent + Summarise",
                        kv_limit: int = 800):
    """
    Stacked-bar simulator: tokens accumulating across turns of an agent
    conversation, partitioned into hot (in KV) / cold (summarised, recoverable) /
    evicted (gone) tiers, under different memory strategies.
    """
    rng = _rng(7)
    per_turn = [int(60 + float(rng.random()) * 80) for _ in range(n_turns)]

    hot_series, cold_series, evicted_series = [], [], []
    hot, cold, evicted = 0, 0, 0
    SUMMARY_RATIO = 0.15  # summary = ~15% of summarised tokens

    for added in per_turn:
        hot += added
        if strategy == "Full Cache":
            pass
        elif strategy == "Streaming (recent only)":
            if hot > kv_limit:
                evicted += hot - kv_limit
                hot = kv_limit
        elif strategy == "SnapKV-style (intent-aware)":
            # SnapKV evicts the SAME number of tokens as Streaming, but it
            # picks them via observation-window voting. Some of the "evicted"
            # tokens remain effectively recoverable through the kept tokens'
            # attention to neighbours — modelled as a small fraction surviving
            # in the cold tier. The rest is fully lost. This is what makes the
            # bar visibly different from plain Streaming.
            if hot > kv_limit:
                overflow = hot - kv_limit
                hot = kv_limit
                kept_through_attention = int(overflow * 0.30)
                cold += kept_through_attention
                evicted += overflow - kept_through_attention
        elif strategy == "Agent + Summarise":
            if hot > kv_limit:
                overflow = hot - kv_limit
                hot = kv_limit
                cold += int(overflow * SUMMARY_RATIO)
                # nothing fully evicted — recoverable from cold tier
        hot_series.append(hot)
        cold_series.append(cold)
        evicted_series.append(evicted)

    total_seen = sum(per_turn)
    accessible = hot_series[-1] + cold_series[-1]
    compression = accessible / max(1, total_seen)

    labels = [f"T{i+1}" for i in range(n_turns)]
    cfg = f"""{{
      type:'bar',
      data:{{
        labels:{json.dumps(labels)},
        datasets:[
          {{label:'Hot — in KV cache',
            data:{json.dumps(hot_series)},
            backgroundColor:'#D85A30', borderWidth:0, stack:'mem'}},
          {{label:'Cold — summarised / external',
            data:{json.dumps(cold_series)},
            backgroundColor:'#534AB7', borderWidth:0, stack:'mem'}},
          {{label:'Evicted — lost',
            data:{json.dumps(evicted_series)},
            backgroundColor:'#B4B2A9', borderWidth:0, stack:'mem'}}
        ]
      }},
      options:{{
        responsive:true, maintainAspectRatio:false, animation:{{duration:400}},
        plugins:{{
          legend:{{position:'top',labels:{{boxWidth:12,font:{{size:11}}}}}},
          tooltip:{{callbacks:{{label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toLocaleString() + ' tokens'}}}}
        }},
        scales:{{
          x:{{stacked:true, title:{{display:true,text:'Conversation turn'}}}},
          y:{{stacked:true, title:{{display:true,text:'Tokens'}}, beginAtZero:true}}
        }}
      }}
    }}"""

    summary = f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Strategy</div>
        <div style="font-size:13px;font-weight:500;color:var(--color-text-primary)">{strategy}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Total seen</div>
        <div style="font-size:16px;font-weight:500">{total_seen:,}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Still accessible</div>
        <div style="font-size:16px;font-weight:500;color:#1D9E75">{accessible:,}</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:10px;text-align:center">
        <div style="font-size:10px;color:var(--color-text-secondary);text-transform:uppercase;letter-spacing:.06em">Recall rate</div>
        <div style="font-size:16px;font-weight:500;color:#BA7517">{compression:.0%}</div>
      </div>
    </div>
    <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:6px;line-height:1.6">
      KV limit = {kv_limit:,} tokens · {n_turns} turns · summarised tokens compressed to
      ~{int(SUMMARY_RATIO*100)}% of original.
    </div>
    """
    return mo.Html(_chart_block("Agent memory over turns by strategy", cfg,
                                height_px=320, pre_canvas_html=summary))
