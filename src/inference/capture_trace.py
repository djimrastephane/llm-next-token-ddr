"""Capture a real next-token inference trace from a local Hugging Face causal LM.

Example:
    python -m src.inference.capture_trace --mode sampling --temperature 0.7 --top-p 0.90 --seed 42 --steps 5
    python -m src.inference.capture_trace --mode greedy --steps 5 --output data/generated/inference_trace_greedy.json
    python -m src.inference.capture_trace --context "Drilled 8½ in. hole section..."
"""

import argparse
import json
import platform
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .device import select_device
from .sampling import decide
from .schemas import SCHEMA_VERSION, validate_trace
from .tokenizer_utils import describe_token

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTEXTS = ROOT / "data" / "input" / "ddr_contexts.json"
DEFAULT_OUTPUT = ROOT / "data" / "generated" / "inference_trace.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Used only in chat-template mode. The DDR text is placed at the start of the assistant turn
# so the model *continues* it rather than answering a question about it.
CHAT_SYSTEM = "You are a drilling engineer writing concise Daily Drilling Report (DDR) operations log entries."
CHAT_USER = "Write the next DDR operations log entry."

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face causal LM id or local path")
    p.add_argument("--mode", choices=["greedy", "sampling"], default="sampling")
    p.add_argument("--prompt-mode", choices=["raw-text", "chat-template"], default="raw-text")
    p.add_argument("--temperature", type=float, default=0.7, help="sampling only; must be > 0")
    p.add_argument("--top-p", type=float, default=0.90, help="sampling only; in (0, 1]")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=5, help="number of tokens to generate (the cap with --until-sentence-end)")
    p.add_argument(
        "--until-sentence-end",
        action="store_true",
        help="stop early once the model emits a token ending the sentence ('.', '!', '?', newline) or end-of-text",
    )
    p.add_argument("--context", help="custom DDR text (overrides --context-id)")
    p.add_argument("--context-id", help="id from data/input/ddr_contexts.json")
    p.add_argument("--contexts-file", default=str(DEFAULT_CONTEXTS))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--display-top-k", type=int, default=10, help="candidates exported per step (plus the full nucleus)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--dtype", choices=list(DTYPES), default="float32")
    p.add_argument("--no-save-logits", action="store_true", help="skip the full-vocabulary .npz sidecar")
    args = p.parse_args(argv)

    if args.steps < 1:
        p.error("--steps must be >= 1")
    if args.mode == "sampling":
        if args.temperature == 0:
            p.error("--temperature 0 would divide by zero. Temperature 0 means deterministic decoding: use --mode greedy.")
        if args.temperature < 0:
            p.error("--temperature must be > 0")
        if not 0 < args.top_p <= 1:
            p.error("--top-p must be in (0, 1]")
    return args


def load_context(args) -> dict:
    if args.context:
        return {
            "id": "custom",
            "category": "custom",
            "title": "Custom context",
            "text": args.context,
            "facts": {},
            "synthetic": None,
        }
    data = json.loads(Path(args.contexts_file).read_text(encoding="utf-8"))
    cid = args.context_id or data["default_context_id"]
    for c in data["contexts"]:
        if c["id"] == cid:
            return {**c, "synthetic": True}
    sys.exit(f"context id {cid!r} not found in {args.contexts_file}. Available: {[c['id'] for c in data['contexts']]}")


def build_input_ids(tokenizer, text: str, prompt_mode: str) -> tuple[list[int], int]:
    """Return (input_ids, index where the DDR text tokens start).

    The prefix (chat template) and the DDR text are tokenized separately and concatenated,
    so the DDR tokens are identical in both modes and the boundary is exact.
    """
    if prompt_mode == "raw-text":
        prefix_ids: list[int] = []
    else:
        messages = [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": CHAT_USER}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    text_ids = tokenizer(text, add_special_tokens=False).input_ids
    return prefix_ids + text_ids, len(prefix_ids)


def ends_sentence(decoded: str) -> bool:
    """True if a generated token closes a sentence: it ends with . ! ? (ignoring trailing spaces) or holds a newline.

    Decided before looking at any output; note a decimal point ("3.") would also count.
    """
    return "\n" in decoded or decoded.rstrip(" ").endswith((".", "!", "?"))


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def candidate_record(tokenizer, d, token_id: int, rank: int, cum: float) -> dict:
    sampling = d.nucleus is not None
    return {
        "rank": rank,
        **describe_token(tokenizer, token_id),
        "raw_logit": float(d.raw_logits[token_id]),
        "scaled_logit": float(d.scaled_logits[token_id]),
        "model_probability": float(d.model_probs[token_id]),
        "temperature_probability": float(d.temperature_probs[token_id]),
        "cumulative_probability": cum,
        "inside_top_p": bool(d.nucleus[token_id]) if sampling else None,
        "sampling_probability": float(d.sampling_probs[token_id]) if sampling else None,
        "selected": token_id == d.selected_id,
    }


def main(argv=None):
    args = parse_args(argv)
    ctx = load_context(args)
    device = select_device(args.device)
    rng = seed_everything(args.seed)

    print(f"[load] {args.model} on {device} ({args.dtype})", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype]).to(device)
    model.eval()

    input_ids, text_start = build_input_ids(tokenizer, ctx["text"], args.prompt_mode)
    eos_ids = set(np.atleast_1d(model.generation_config.eos_token_id or []).tolist())
    if tokenizer.eos_token_id is not None:
        eos_ids.add(tokenizer.eos_token_id)

    input_tokens = [
        {"position": i, **describe_token(tokenizer, t), "in_display_context": i >= text_start} for i, t in enumerate(input_ids)
    ]

    ids = list(input_ids)
    generated: list[int] = []
    steps, all_logits = [], []
    stop_reason = "max_steps"

    for step in range(1, args.steps + 1):
        context_before = ctx["text"] + tokenizer.decode(generated)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(torch.tensor([ids], device=device))
        raw = out.logits[0, -1].float().cpu().numpy()  # next-token logits over the full vocabulary
        forward_ms = (time.perf_counter() - t0) * 1000
        all_logits.append(raw.astype(np.float32))

        d = decide(raw, args.mode, args.temperature, args.top_p, rng)
        probs_for_rank = d.temperature_probs
        n_nucleus = int(d.nucleus.sum()) if d.nucleus is not None else None
        n_export = max(args.display_top_k, n_nucleus or 0)
        selected_rank = int(np.where(d.order == d.selected_id)[0][0]) + 1
        n_export = max(n_export, selected_rank)  # always include the selected token

        cum = np.cumsum(probs_for_rank[d.order[:n_export]])
        candidates = [candidate_record(tokenizer, d, int(tid), r + 1, float(cum[r])) for r, tid in enumerate(d.order[:n_export])]
        listed = d.order[:n_export]
        omitted_model = float(1.0 - d.model_probs[listed].sum())
        omitted_temp = float(1.0 - d.temperature_probs[listed].sum())

        sel = next(c for c in candidates if c["selected"])
        ids.append(d.selected_id)
        generated.append(d.selected_id)
        context_after = ctx["text"] + tokenizer.decode(generated)

        steps.append(
            {
                "step": step,
                "context_before": context_before,
                "vocab_size": int(raw.shape[0]),
                "forward_ms": round(forward_ms, 1),
                "nucleus_size": n_nucleus,
                "nucleus_temperature_mass": float(d.temperature_probs[d.nucleus].sum()) if d.nucleus is not None else None,
                "omitted_model_probability_mass": max(omitted_model, 0.0),
                "omitted_temperature_probability_mass": max(omitted_temp, 0.0),
                "top_candidates": candidates,
                "selected_token": {
                    k: sel[k]
                    for k in (
                        "rank",
                        "token_id",
                        "raw_token",
                        "decoded_token",
                        "display_token",
                        "is_special",
                        "raw_logit",
                        "model_probability",
                        "temperature_probability",
                        "sampling_probability",
                    )
                },
                "context_after": context_after,
            }
        )
        print(
            f"[step {step}] selected {sel['display_token']!r} (rank {sel['rank']}, "
            f"p_model={sel['model_probability']:.4f}"
            + (f", p_sample={sel['sampling_probability']:.4f}, nucleus={n_nucleus}" if n_nucleus else "")
            + ")",
            file=sys.stderr,
        )
        if d.selected_id in eos_ids:
            stop_reason = "eos"
            break
        if args.until_sentence_end and ends_sentence(sel["decoded_token"]):
            stop_reason = "sentence_end"
            break

    sampling = args.mode == "sampling"
    trace = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "model": args.model,
            "model_display_name": args.model.rstrip("/").split("/")[-1],
            "model_revision": getattr(model.config, "_commit_hash", None),
            "device": device.type,
            "dtype": args.dtype,
            "generation_mode": args.mode,
            "prompt_mode": args.prompt_mode,
            "temperature": args.temperature if sampling else None,
            "top_p": args.top_p if sampling else None,
            "seed": args.seed,
            "steps_requested": args.steps,
            "steps_generated": len(steps),
            "stop_reason": stop_reason,
            "until_sentence_end": args.until_sentence_end,
            "display_top_k": args.display_top_k,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
            },
            "method": {
                "model_probability": "softmax(raw_logits): the model's own distribution (temperature 1)",
                "temperature_probability": "softmax(raw_logits / temperature): the distribution top-p is applied to"
                if sampling
                else "same as model_probability (greedy uses no temperature)",
                "cumulative_probability": "running sum of temperature_probability in rank order",
                "sampling_probability": "temperature_probability renormalized over the top-p nucleus; 0 outside it"
                if sampling
                else None,
                "selection": "numpy.random.default_rng(seed).choice over sampling_probability"
                if sampling
                else "argmax (rank 1); deterministic, no sampling",
                "ranking_ties": "stable sort: equal probabilities keep ascending token-id order",
                "not_applied": (
                    "No repetition penalty, top-k or other logits processors are applied, even if the model's "
                    "generation_config defines them (Qwen2.5-Instruct sets repetition_penalty=1.1, top_k=20, top_p=0.8). "
                    "model.generate() would apply those defaults and can therefore select different tokens."
                ),
            },
            "reproducibility_note": (
                "Seeds are fixed (python random, numpy, torch) and the sampling draw uses a seeded NumPy generator. "
                "Logits can still differ in the last floating-point digits across devices, dtypes and library "
                "versions, so bit-for-bit identical traces are only expected on the same setup."
            ),
        },
        "source_context": {k: ctx[k] for k in ("id", "category", "title", "facts", "synthetic")},
        "display_context": ctx["text"],
        "actual_model_input": tokenizer.decode(input_ids, skip_special_tokens=False),
        "display_context_token_start": text_start,
        "input_tokens": input_tokens,
        "steps": steps,
        "generated_text": tokenizer.decode(generated),
        "final_display_context": ctx["text"] + tokenizer.decode(generated),
    }

    validate_trace(trace)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[write] {out_path} ({out_path.stat().st_size / 1024:.0f} KB)", file=sys.stderr)

    if not args.no_save_logits:
        npz = out_path.with_suffix(".logits.npz")
        np.savez_compressed(
            npz,
            input_ids=np.array(input_ids),
            generated_ids=np.array(generated),
            raw_logits=np.stack(all_logits),
        )
        print(f"[write] {npz} (full-vocabulary raw logits per step)", file=sys.stderr)
    return trace


if __name__ == "__main__":
    main()
