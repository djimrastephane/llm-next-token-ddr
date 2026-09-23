"""Next-token decoding math, kept independent of the model so it can be tested exactly.

All functions operate on 1-D float64 NumPy arrays over the full vocabulary.
The order of operations follows Hugging Face's sampling pipeline:
    logits -> / temperature -> softmax -> top-p nucleus -> renormalize -> sample
"""

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)  # numerical stability; does not change the result
    e = np.exp(z)
    return e / e.sum()


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Divide logits by T. T must be > 0; T = 0 would be a division by zero and is handled by the caller
    (greedy decoding is the T -> 0 limit)."""
    if not temperature > 0:
        raise ValueError(
            f"temperature must be > 0 for sampling (got {temperature}). "
            "Temperature 0 is the deterministic limit: use --mode greedy instead."
        )
    return np.asarray(logits, dtype=np.float64) / temperature


def rank_order(probs: np.ndarray) -> np.ndarray:
    """Token ids sorted by descending probability. Stable: ties keep ascending token-id order."""
    return np.argsort(-probs, kind="stable")


def top_p_mask(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Boolean mask of the nucleus: the smallest highest-probability set whose cumulative probability >= top_p.

    A token is kept if the cumulative probability of all tokens ranked *above* it is still < top_p.
    The top-ranked token is therefore always kept.
    """
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    order = rank_order(probs)
    cum = np.cumsum(probs[order])
    cum_before = np.concatenate([[0.0], cum[:-1]])
    keep_sorted = cum_before < top_p
    mask = np.zeros_like(probs, dtype=bool)
    mask[order[keep_sorted]] = True
    return mask


def renormalize(probs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.where(mask, probs, 0.0)
    return out / out.sum()


@dataclass
class Decision:
    """Everything about one decoding step, over the full vocabulary."""

    raw_logits: np.ndarray
    scaled_logits: np.ndarray  # raw / T (sampling) or raw (greedy)
    model_probs: np.ndarray  # softmax(raw logits): the model's own distribution (T = 1)
    temperature_probs: np.ndarray  # softmax(raw / T): the distribution top-p is applied to
    order: np.ndarray  # token ids by rank
    nucleus: np.ndarray | None  # bool mask (sampling only)
    sampling_probs: np.ndarray | None  # renormalized nucleus (sampling only)
    selected_id: int


def decide(
    raw_logits: np.ndarray,
    mode: str,
    temperature: float | None = None,
    top_p: float | None = None,
    rng: np.random.Generator | None = None,
) -> Decision:
    raw = np.asarray(raw_logits, dtype=np.float64)
    model_probs = softmax(raw)
    if mode == "greedy":
        # Deterministic: argmax. No temperature, no top-p, no randomness.
        order = rank_order(model_probs)
        return Decision(raw, raw, model_probs, model_probs, order, None, None, int(order[0]))
    if mode == "sampling":
        if temperature is None or top_p is None or rng is None:
            raise ValueError("sampling needs temperature, top_p and rng")
        scaled = apply_temperature(raw, temperature)
        t_probs = softmax(scaled)
        order = rank_order(t_probs)
        nucleus = top_p_mask(t_probs, top_p)
        s_probs = renormalize(t_probs, nucleus)
        selected = int(rng.choice(len(s_probs), p=s_probs))
        return Decision(raw, scaled, model_probs, t_probs, order, nucleus, s_probs, selected)
    raise ValueError(f"unknown mode {mode!r}")
