"""Standalone, batched, Markovian GRU recall predictor (roadmap step 4).

This is a dependency-free re-implementation of srs-benchmark's ``models/gru.py`` (the
``GRU`` class, ``--short --secs`` variant) for use *inside* the FSRS-7 simulator as the
pseudo-ground-truth p(recall) model. It carries **per-user weights** stacked over a
``parallel`` (= user) axis and runs in float64 to match the simulator.

Why this can be Markovian (the whole point of step 4)
------------------------------------------------------
In the reference ``GRU`` the only recurrent layer is the ``nn.GRU``; every other layer
(the ``Linear -> SiLU -> LayerNorm`` *before* it, the ``LayerNorm -> Linear -> SiLU ->
LayerNorm`` *after* it, and the ``w_fc/s_fc/d_fc`` heads) is applied **per timestep**
(``LayerNorm`` normalizes the hidden dim, not time). So the GRU hidden vector ``h`` (size
``N_HIDDEN=7``) is a *sufficient statistic* for the entire review history -- exactly like
FSRS's ``(s_long, s_short, d)``. We carry ``h`` per card and step it one review at a time;
this reproduces the reference's full-sequence ``forward`` bit-for-bit (verified by
``tests/test_gru_step_parity.py``), with no history buffers.

Per-step pipeline (mirrors ``models/gru.py`` exactly)
-----------------------------------------------------
Input feature vector per review:  ``x = [ (log(1e-5 + dt) - input_mean) / input_std ,
one_hot(rating - 1, 4) ]``  (5 dims; ``dt`` is the elapsed interval in **fractional days**,
matching ``delta_t_secs = elapsed_seconds / 86400``).

  1. pre:  ``c = LayerNorm2( SiLU( Linear0(x) ) )``           (per timestep)
  2. GRU:  ``h' = GRUCell(c, h)``                              (the only recurrent step)
  3. post: ``g = LayerNorm7( SiLU( Linear5( LayerNorm4(h') ) ) )``   (per timestep)
  4. heads: ``w = softmax(w_fc(g)); s = exp(clamp(s_fc(g))); d = exp(clamp(d_fc(g)))``
  5. forgetting curve (2-curve mixture):
        ``p(t) = (1 - 1e-7) * sum_k w_k * (1 + t / (1e-7 + s_k)) ** (-d_k)``

GRUCell equations (PyTorch convention; gate order **reset, update, new** -- see the
``bias_ih`` init comment in ``models/gru.py``). ``weight_ih`` rows ``[0:H]=ir, [H:2H]=iz,
[2H:3H]=in``; ``weight_hh`` likewise for the hidden side::

    r = sigmoid(W_ir c + b_ir + W_hr h + b_hr)
    z = sigmoid(W_iz c + b_iz + W_hz h + b_hz)
    n = tanh  (W_in c + b_in + r * (W_hn h + b_hn))
    h' = (1 - z) * n + z * h

These equations are written out explicitly (rather than calling ``nn.GRU``) so the Rust /
candle port -- candle reads the same ``.pth`` files -- has an unambiguous reference.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

N_INPUT = 5  # log-delay (1) + rating one-hot (4)
N_HIDDEN = 7
N_CURVES = 2
LN_EPS = 1e-5  # torch.nn.LayerNorm default eps
S_CLAMP = 25.0  # reference clamps the s_fc / d_fc logits to [-25, 25] before exp


def _silu(x):
    return x * torch.sigmoid(x)


def _layernorm(x, gamma):
    """LayerNorm over the last dim with no bias (matches nn.LayerNorm(H, bias=False)).

    x: (P, D, H); gamma: (P, H). Uses the biased (population) variance, like torch.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    xn = (x - mean) / torch.sqrt(var + LN_EPS)
    return xn * gamma.unsqueeze(1)


def _linear(x, weight, bias):
    """Batched (over P) affine map. x: (P, D, in); weight: (P, out, in); bias: (P, out)."""
    return torch.einsum("pdi,poi->pdo", x, weight) + bias.unsqueeze(1)


class BatchedGRU:
    """A per-user GRU recall predictor, batched over the ``parallel`` (user) axis.

    All weight tensors have a leading ``P`` (= ``parallel``) dimension. Call ``init_hidden``
    once per simulation to get the per-card hidden state, then ``step`` it forward at every
    review and read p(recall) with ``p_recall`` / ``curve_params``.
    """

    def __init__(self, weights: dict[str, torch.Tensor]):
        self.w = weights
        self.parallel = weights["pre_w"].shape[0]
        self.device = weights["pre_w"].device
        self.dtype = weights["pre_w"].dtype

    # Canonical flat order for exporting per-user weights to the Rust simulator (the Rust
    # GruW::from_row reader unpacks the same order). Row-major within each tensor.
    FLAT_ORDER = (
        "pre_w",
        "pre_b",
        "ln_pre",
        "weight_ih",
        "weight_hh",
        "bias_ih",
        "bias_hh",
        "ln_post1",
        "post_w",
        "post_b",
        "ln_post2",
        "w_fc_w",
        "w_fc_b",
        "s_fc_w",
        "s_fc_b",
        "d_fc_w",
        "d_fc_b",
        "input_mean",
        "input_std",
    )
    FLAT_LEN = 505  # 35+7+7+147+147+21+21+7+49+7+7+14+2+14+2+14+2+1+1

    def flat_weights(self) -> torch.Tensor:
        """Per-user weights as a single contiguous ``(parallel, FLAT_LEN)`` f64 tensor,
        concatenated in ``FLAT_ORDER`` (row-major). Fed to the Rust simulator, which reads
        the identical layout. Reuses the same numbers loaded from the ``.pth`` files."""
        parts = [self.w[k].reshape(self.parallel, -1) for k in self.FLAT_ORDER]
        flat = torch.cat(parts, dim=1).to(torch.float64).contiguous()
        assert flat.shape == (self.parallel, self.FLAT_LEN), flat.shape
        return flat

    # -- construction ----------------------------------------------------------------

    @classmethod
    def from_state_dicts(cls, state_dicts, device="cpu", dtype=torch.float64):
        """Stack a list of per-user reference ``GRU`` state_dicts over the P axis.

        Each state_dict uses the reference key names (``process.0.weight``,
        ``process.3.module.weight_ih_l0``, ``w_fc.weight``, ``input_mean``, ...).
        """

        def stack(key):
            return torch.stack(
                [
                    torch.as_tensor(sd[key], dtype=dtype, device=device)
                    for sd in state_dicts
                ]
            )

        def stack_scalar(key):
            # input_mean/input_std are a single value per user, but the saved buffer may be
            # 0-d (scalar) or shape (1,) depending on how it was registered/loaded. Normalize
            # to (P, 1) so it broadcasts against (P, deck) regardless.
            return torch.stack(
                [
                    torch.as_tensor(sd[key], dtype=dtype, device=device).reshape(1)
                    for sd in state_dicts
                ]
            )

        weights = {
            # input normalization (delay only) -> (P, 1)
            "input_mean": stack_scalar("input_mean"),
            "input_std": stack_scalar("input_std"),
            # pre block: Linear(5 -> 7) then LayerNorm(7)
            "pre_w": stack("process.0.weight"),  # (P, 7, 5)
            "pre_b": stack("process.0.bias"),  # (P, 7)
            "ln_pre": stack("process.2.weight"),  # (P, 7)
            # GRU cell (input_size = hidden_size = 7)
            "weight_ih": stack("process.3.module.weight_ih_l0"),  # (P, 21, 7)
            "weight_hh": stack("process.3.module.weight_hh_l0"),  # (P, 21, 7)
            "bias_ih": stack("process.3.module.bias_ih_l0"),  # (P, 21)
            "bias_hh": stack("process.3.module.bias_hh_l0"),  # (P, 21)
            # post block: LayerNorm(7) -> Linear(7 -> 7) -> LayerNorm(7)
            "ln_post1": stack("process.4.weight"),  # (P, 7)
            "post_w": stack("process.5.weight"),  # (P, 7, 7)
            "post_b": stack("process.5.bias"),  # (P, 7)
            "ln_post2": stack("process.7.weight"),  # (P, 7)
            # heads: Linear(7 -> 2) each
            "w_fc_w": stack("w_fc.weight"),  # (P, 2, 7)
            "w_fc_b": stack("w_fc.bias"),  # (P, 2)
            "s_fc_w": stack("s_fc.weight"),
            "s_fc_b": stack("s_fc.bias"),
            "d_fc_w": stack("d_fc.weight"),
            "d_fc_b": stack("d_fc.bias"),
        }
        return cls(weights)

    @classmethod
    def from_pth_paths(cls, paths, device="cpu", dtype=torch.float64):
        sds = [
            torch.load(Path(p), weights_only=True, map_location="cpu") for p in paths
        ]
        return cls.from_state_dicts(sds, device=device, dtype=dtype)

    # -- inference -------------------------------------------------------------------

    def init_hidden(self, deck_size: int) -> torch.Tensor:
        """Per-card GRU hidden state, all zeros (nn.GRU's default h_0). (P, deck, H)."""
        return torch.zeros(
            (self.parallel, deck_size, N_HIDDEN), dtype=self.dtype, device=self.device
        )

    def _features(self, dt: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        """Build the (P, deck, 5) input: normalized log-delay + rating one-hot."""
        w = self.w
        x_delay = torch.log(1e-5 + dt)  # (P, deck)
        x_main = (x_delay - w["input_mean"]) / w[
            "input_std"
        ]  # (P, deck) via (P,1) bcast
        x_main = x_main.unsqueeze(-1)  # (P, deck, 1)
        idx = torch.clamp(rating, min=1).long() - 1  # 0..3
        onehot = F.one_hot(idx, num_classes=4).to(self.dtype)  # (P, deck, 4)
        return torch.cat([x_main, onehot], dim=-1)  # (P, deck, 5)

    def step(
        self, h: torch.Tensor, dt: torch.Tensor, rating: torch.Tensor
    ) -> torch.Tensor:
        """Advance the hidden state by one review ``(dt, rating)``. Returns new h (P,deck,H).

        ``dt`` is the elapsed interval (fractional days), ``rating`` an int in 1..4. Both
        are (P, deck). ``h`` is (P, deck, H).
        """
        w = self.w
        x = self._features(dt, rating)  # (P, deck, 5)
        # pre block
        c = _layernorm(
            _silu(_linear(x, w["pre_w"], w["pre_b"])), w["ln_pre"]
        )  # (P,deck,7)
        # GRU cell
        gi = _linear(c, w["weight_ih"], w["bias_ih"])  # (P, deck, 21)
        gh = _linear(h, w["weight_hh"], w["bias_hh"])  # (P, deck, 21)
        H = N_HIDDEN
        i_r, i_z, i_n = gi[..., :H], gi[..., H : 2 * H], gi[..., 2 * H :]
        h_r, h_z, h_n = gh[..., :H], gh[..., H : 2 * H], gh[..., 2 * H :]
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h

    def curve_params(self, h: torch.Tensor):
        """Map the hidden state to the 2-curve forgetting-curve params (w, s, d).

        Each output is (P, deck, N_CURVES). ``w`` sums to 1 over the curve axis.
        """
        wts = self.w
        g = _layernorm(h, wts["ln_post1"])
        g = _linear(g, wts["post_w"], wts["post_b"])
        g = _layernorm(_silu(g), wts["ln_post2"])  # (P, deck, 7)
        w_curve = torch.softmax(_linear(g, wts["w_fc_w"], wts["w_fc_b"]), dim=-1)
        s_curve = torch.exp(
            torch.clamp(_linear(g, wts["s_fc_w"], wts["s_fc_b"]), -S_CLAMP, S_CLAMP)
        )
        d_curve = torch.exp(
            torch.clamp(_linear(g, wts["d_fc_w"], wts["d_fc_b"]), -S_CLAMP, S_CLAMP)
        )
        return w_curve, s_curve, d_curve

    @staticmethod
    def forgetting_curve(t, w_curve, s_curve, d_curve):
        """2-curve power forgetting curve (matches GRU.forgetting_curve). t: (P, deck)."""
        t3 = t.unsqueeze(-1)  # (P, deck, 1) broadcast over curves
        return (1 - 1e-7) * torch.sum(
            w_curve * (1 + t3 / (1e-7 + s_curve)) ** (-d_curve), dim=-1
        )

    def p_recall(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """p(recall) after elapsed ``dt`` (fractional days), given hidden state ``h``."""
        w_curve, s_curve, d_curve = self.curve_params(h)
        return self.forgetting_curve(dt, w_curve, s_curve, d_curve)
