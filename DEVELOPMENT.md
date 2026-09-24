# Development guide

How to run, modify and verify the project. For what the project shows and why, start with the [README](README.md).

- [Install](#install)
- [Capture the traces](#capture-the-traces)
- [Preview and render](#preview-and-render)
- [Tests and checks](#tests-and-checks)
- [How the numbers are computed](#how-the-numbers-are-computed)
- [Trace format](#trace-format-schema-11)
- [Repository layout](#repository-layout)

---

## Install

```bash
# Python 3.11+ (exact tested versions are pinned in requirements.txt)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Node 22.12+ (required by the locked Vitest; 22.18 used here)
npm install
```

**Model download.** On first run, Hugging Face downloads `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) into `~/.cache/huggingface`, at the pinned commit `989aa798…` so a later upstream update can't silently change the weights (override with `--revision`; other models default to their latest revision). Weights are loaded from safetensors files only, never from pickled `.bin` files. After that everything runs offline (`HF_HUB_OFFLINE=1`). Float32 needs ~7 GB of RAM; use `--dtype bfloat16` on smaller machines (this changes the logits slightly). The device is chosen automatically in the order CUDA → MPS → CPU (override with `--device`).

**Fonts.** Rendering downloads the Inter and JetBrains Mono fonts from Google Fonts; model inference itself needs no network once the model is cached.

## Capture the traces

```bash
scripts/capture_trace.sh          # sampling trace (drives the video) + greedy trace, same context
OUT_DIR=/tmp/run scripts/capture_trace.sh   # write both traces elsewhere (--output is refused: it would make one overwrite the other)
```

or directly:

```bash
python -m src.inference.capture_trace --mode sampling --temperature 0.7 --top-p 0.90 --seed 42 --steps 40 --until-sentence-end
python -m src.inference.capture_trace --mode greedy --steps 40 --until-sentence-end --output data/generated/inference_trace_greedy.json
```

| Change… | Flag |
|---|---|
| DDR example | `--context-id completions_packer` (see `data/input/ddr_contexts.json`) |
| Custom DDR text | `--context "Drilled 8½ in. hole section…"` (labelled "custom input, provenance not recorded" in the video) |
| Model | `--model Qwen/Qwen2.5-0.5B-Instruct` (any HF causal LM) |
| Model revision | `--revision <commit>` (the default model is pinned) |
| Temperature / top-p / seed | `--temperature 0.9 --top-p 0.95 --seed 7` |
| Greedy vs sampling | `--mode greedy` / `--mode sampling` |
| Number of tokens | `--steps 8`, or `--until-sentence-end` to stop when the model ends the sentence (`--steps` is then the cap). The video's loop scene grows automatically. |
| Prompt format | `--prompt-mode chat-template` |
| Very large nucleus | `--max-export 1000` (default): cap on exported nucleus candidates per step |

The video renders `data/generated/inference_trace.json` (must be a sampling trace, because the video explains temperature and top-p) and compares it with `data/generated/inference_trace_greedy.json` (must come from the same context, model, revision and prompt mode). After capturing new traces, re-render; there is nothing to edit by hand.

## Preview and render

```bash
scripts/preview_video.sh          # Remotion Studio
scripts/render_video.sh           # → out/next_token_ddr.mp4
node scripts/render_stills.mjs 1200 2800   # QA stills → out/stills/
```

## Tests and checks

```bash
.venv/bin/python -m pytest        # 90 tests: maths, top-p vs Hugging Face, tokenization, schema and tamper detection, recompute from logits,
                                  #   review regressions (incl. one top-p = 1 capture with the cached 0.5B model; deselect with -m 'not slow')
npm test                          # 11 tests: loader serves the traces unchanged, rejects old/tampered/mismatched traces, keeps low-ranked selections visible
npm run typecheck && npm run lint && .venv/bin/ruff check src tests
```

---

## How the numbers are computed

For raw logits $z$ over vocabulary $V$ and temperature $T > 0$:

$$p_\text{model}(i) = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad p_T(i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Sort by $p_T$ descending. The nucleus $N$ is the smallest prefix whose cumulative $p_T$ reaches $p$: a token is kept if the cumulative probability of the tokens ranked above it is still $< p$, so the top token is always kept. Then:

$$p_\text{sample}(i) = \frac{p_T(i)}{\sum_{k\in N} p_T(k)} \text{ for } i \in N,\quad 0 \text{ otherwise}$$

and the token is drawn with `numpy.random.default_rng(seed).choice(V, p=p_sample)`.

Implementation notes (verified, not assumed):

- All maths runs in float64 NumPy on the model's float32 logits (`src/inference/sampling.py`). Ranking ties use a stable sort (lower token ID first).
- The nucleus matches Hugging Face's own `TemperatureLogitsWarper` + `TopPLogitsWarper` exactly at every step; a test checks this.
- **Temperature 0**: dividing by zero is never attempted. `--mode sampling --temperature 0` is rejected with a message pointing to `--mode greedy` (the T → 0 limit). Non-finite temperature or top-p values are rejected too. Greedy mode ignores temperature and top-p and records them as `null`.
- **No other logits processors are applied.** Qwen's `generation_config` sets `repetition_penalty=1.1, top_k=20, top_p=0.8`, and `model.generate()` applies these even with `do_sample=False`. We verified that `generate()` with default settings writes " constant at 37 psi. No gas was…", which differs from our greedy trace from step 4 on (the repetition penalty lowers tokens already in the context, such as the digits in "12,450"). With `repetition_penalty=1.0`, `generate()` reproduces our greedy trace token for token. This project deliberately shows the plain pipeline.
- Each step runs a full forward pass over the whole sequence (no KV cache) under `torch.inference_mode()`, which matches `logits = model(context)` literally.
- **Reproducibility.** Python `random`, NumPy, and PyTorch are all seeded, and re-running on the same machine reproduced the trace exactly. Logits can differ in the last digits across devices, dtypes, and library versions, so bit-identical traces are only expected on the same setup.
- **Stopping.** `--until-sentence-end` stops at a generated newline, `!` or `?`, or a `.` that doesn't follow a digit (possible decimal point) or a DDR abbreviation such as `in.`, `ft.` or `approx.`. The word before the period may come from the prompt.

### Prompt formatting matters

`--prompt-mode raw-text` (default) feeds the DDR text as-is. `--prompt-mode chat-template` wraps it with the tokenizer's official chat template (system + user instruction), placing the DDR text at the start of the assistant turn so the model continues it. **Different prompt formatting gives a different probability distribution.** With the chat template, step 1 picks ` stable` (54.7%) instead of ` constant` (31.0% in raw-text mode). The trace always stores both `display_context` (what viewers see) and `actual_model_input` (the exact decoded model input, including any special tokens), and flags which input tokens belong to the DDR text.

## Trace format (schema 1.1)

`data/generated/inference_trace.json`:

- `metadata`: model, requested and loaded revision, `tokenizer_vocab_size`, device, dtype, modes, temperature, top_p, seed, versions, and the exact method used for each quantity
- `source_context.synthetic`: `true` only when the contexts file declares it; `null` for `--context` text or files that don't say. The video labels the text accordingly and never assumes "synthetic".
- `display_context`, `actual_model_input`, `input_tokens[]` (id, raw tokenizer string, exact decoded text, UI-safe display text, `in_tokenizer`)
- `steps[]`, each with `context_before`/`context_after`, `appended_text` (the text this step added; a character split across tokens, such as an emoji, is credited to the token that completes it), `nucleus_size`, `nucleus_complete`, omitted probability mass, and `top_candidates[]`: the top 10 **plus the nucleus** (up to `--max-export`, default 1,000) **plus the selected token**, each with `raw_logit`, `scaled_logit`, `model_probability`, `temperature_probability`, `cumulative_probability`, `inside_top_p`, `sampling_probability`, `selected`
- `selected_token` and `generated_text`

A very large nucleus (for example top-p = 1.0, where every one of the 151,936 output rows is eligible) is exported only up to `--max-export` and marked `nucleus_complete: false`; the maths still runs over the full vocabulary. Output rows beyond the tokenizer (padding) are exported with `in_tokenizer: false` and empty text rather than an invented token string. Schema 1.0 traces are rejected with a message to re-capture.

The full-vocabulary logits for every step are saved next to the trace as `*.logits.npz` (git-ignored), and the tests recompute every JSON probability from them. The trace is validated on write (`src/inference/schemas.py`) and again on load in the video (`src/video/data/loadInferenceTrace.ts`). Validation checks structure and consistency: probabilities must follow from the recorded logits, the selected token must match its candidate record, all numbers must be finite, and the per-step text must reproduce the generated text.

## Repository layout

```
data/input/ddr_contexts.json          synthetic DDR examples
data/generated/inference_trace*.json  captured traces (source of truth)
src/inference/                        device.py · tokenizer_utils.py · sampling.py · schemas.py · capture_trace.py
src/video/data/loadInferenceTrace.ts  the ONLY place the video reads model output
src/video/components/                 ProbabilityChart/Bar, TokenChip, TopPNucleus, SelectionClaw, HUD, CodeEditor, …
src/video/scenes/                     S1Question … S10Final (S8bCompare: greedy vs sampling)
tests/                                pytest suite (incl. a check that README quotes match the traces)
scripts/                              capture / preview / render helpers
docs/media/                           README images, taken from the rendered video
```
