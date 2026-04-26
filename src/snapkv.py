"""
SnapKV: core algorithm implementation.
Paper: https://arxiv.org/abs/2404.14469
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class SnapKVConfig:
    window_size: int = 16       # observation window (w in paper)
    max_capacity: int = 256     # budget: max KV pairs to keep per head (k)
    kernel_size: int = 5        # pooling kernel for clustering (k_pool)
    pooling: str = "maxpool"    # "maxpool" or "avgpool"


class SnapKVCache:
    """
    Minimal SnapKV implementation following Algorithm 1 in the paper.

    Operates on a single layer's KV cache of shape:
        keys:   (batch, heads, seq_len, head_dim)
        values: (batch, heads, seq_len, head_dim)
    """

    def __init__(self, config: SnapKVConfig):
        self.cfg = config

    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ):
        """
        Compress KV cache using SnapKV.

        Args:
            keys:             (B, H, S, D)
            values:           (B, H, S, D)
            attention_scores: (B, H, S, S) — full attention matrix from prefill.
                              If None, we compute a proxy using Q=K (self-similarity).

        Returns:
            compressed_keys:   (B, H, K+W, D)
            compressed_values: (B, H, K+W, D)
            selected_indices:  (B, H, K+W) — which original positions were kept
        """
        B, H, S, D = keys.shape
        w = min(self.cfg.window_size, S)
        budget = min(self.cfg.max_capacity, S - w)

        # Split into prefix (to compress) and observation window (always kept)
        prefix_keys   = keys[:, :, :-w, :]    # (B, H, S-w, D)
        prefix_values = values[:, :, :-w, :]
        window_keys   = keys[:, :, -w:, :]    # (B, H, w, D)
        window_values = values[:, :, -w:, :]

        if prefix_keys.shape[2] == 0 or budget == 0:
            return keys, values, torch.arange(S).unsqueeze(0).unsqueeze(0).expand(B, H, -1)

        # Stage 1: Vote — score prefix tokens using obs. window attention
        vote_scores = self._compute_vote_scores(prefix_keys, window_keys, attention_scores, w)
        # vote_scores: (B, H, S-w)

        # Stage 2: Cluster — pool scores to keep local neighborhoods
        pooled_scores = self._pool_scores(vote_scores)

        # Select top-k indices
        k = min(budget, pooled_scores.shape[2])
        _, top_indices = pooled_scores.topk(k, dim=-1, sorted=True)
        top_indices, _ = top_indices.sort(dim=-1)  # restore temporal order

        # Gather selected KV pairs
        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        sel_keys   = torch.gather(prefix_keys,   2, idx_expanded)
        sel_values = torch.gather(prefix_values, 2, idx_expanded)

        # Concatenate selected prefix + full observation window
        comp_keys   = torch.cat([sel_keys,   window_keys],   dim=2)
        comp_values = torch.cat([sel_values, window_values], dim=2)

        # Track which original positions were kept
        window_indices = torch.arange(S - w, S, device=keys.device)
        window_indices = window_indices.unsqueeze(0).unsqueeze(0).expand(B, H, -1)
        selected_indices = torch.cat([top_indices, window_indices], dim=-1)

        return comp_keys, comp_values, selected_indices

    def _compute_vote_scores(self, prefix_keys, window_keys, attention_scores, w):
        """
        Compute importance score for each prefix token using the obs. window.
        Uses average attention score from obs. window queries over prefix keys.
        """
        B, H, Sp, D = prefix_keys.shape
        # Use window keys as query proxy (Q ≈ K, reasonable approximation)
        # window_keys: (B, H, w, D)
        scale = D ** -0.5
        # (B, H, w, Sp) = (B, H, w, D) @ (B, H, D, Sp)
        attn = torch.matmul(window_keys, prefix_keys.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        # Average over the obs. window dimension → (B, H, Sp)
        scores = attn.mean(dim=2)
        return scores

    def _pool_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling kernel to smooth scores (clustering step).
        scores: (B, H, S)
        """
        B, H, S = scores.shape
        k = self.cfg.kernel_size
        padding = k // 2
        # Reshape for F.max_pool1d: (B*H, 1, S)
        x = scores.view(B * H, 1, S)
        if self.cfg.pooling == "maxpool":
            pooled = F.max_pool1d(x, kernel_size=k, stride=1, padding=padding)
        else:
            pooled = F.avg_pool1d(x, kernel_size=k, stride=1, padding=padding)
        return pooled.view(B, H, S)

    def get_compression_stats(self, original_seq_len: int) -> dict:
        w = min(self.cfg.window_size, original_seq_len)
        k = min(self.cfg.max_capacity, original_seq_len - w)
        kept = k + w
        return {
            "original":        original_seq_len,
            "kept":            kept,
            "compression":     kept / original_seq_len,
            "reduction":       1 - kept / original_seq_len,
            "obs_window":      w,
            "selected_prefix": k,
        }


class H2OCache:
    """
    H2O baseline: Heavy-Hitter Oracle.
    Retains top-k tokens by cumulative attention score + recent window.
    Paper: https://arxiv.org/abs/2306.14048
    """

    def __init__(self, heavy_ratio: float = 0.1, recent_ratio: float = 0.1):
        self.heavy_ratio = heavy_ratio
        self.recent_ratio = recent_ratio

    def compress(self, keys: torch.Tensor, values: torch.Tensor):
        B, H, S, D = keys.shape
        n_recent = max(1, int(S * self.recent_ratio))
        n_heavy  = max(1, int(S * self.heavy_ratio))

        # Compute cumulative attention scores as proxy
        scale = D ** -0.5
        attn = torch.matmul(keys, keys.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        cum_scores = attn.sum(dim=2)  # (B, H, S) — accumulated attention received

        # Mask out recent tokens from heavy-hitter selection
        cum_scores[:, :, -n_recent:] = -1e9
        _, heavy_idx = cum_scores.topk(n_heavy, dim=-1)
        heavy_idx, _ = heavy_idx.sort(dim=-1)

        recent_idx = torch.arange(S - n_recent, S, device=keys.device)
        recent_idx = recent_idx.unsqueeze(0).unsqueeze(0).expand(B, H, -1)

        selected_idx = torch.cat([heavy_idx, recent_idx], dim=-1)
        idx_exp = selected_idx.unsqueeze(-1).expand(-1, -1, -1, D)

        return (
            torch.gather(keys,   2, idx_exp),
            torch.gather(values, 2, idx_exp),
            selected_idx,
        )


class StreamingLLMCache:
    """
    StreamingLLM baseline: keep attention sink tokens + recent window.
    Paper: https://arxiv.org/abs/2309.17453
    """

    def __init__(self, n_sink: int = 4, n_recent: int = 256):
        self.n_sink   = n_sink
        self.n_recent = n_recent

    def compress(self, keys: torch.Tensor, values: torch.Tensor):
        B, H, S, D = keys.shape
        n_sink   = min(self.n_sink,   S)
        n_recent = min(self.n_recent, S - n_sink)

        sink_idx   = torch.arange(0, n_sink, device=keys.device)
        recent_idx = torch.arange(S - n_recent, S, device=keys.device)
        selected_idx = torch.cat([sink_idx, recent_idx])
        selected_idx = selected_idx.unsqueeze(0).unsqueeze(0).expand(B, H, -1)

        idx_exp = selected_idx.unsqueeze(-1).expand(-1, -1, -1, D)
        return (
            torch.gather(keys,   2, idx_exp),
            torch.gather(values, 2, idx_exp),
            selected_idx,
        )