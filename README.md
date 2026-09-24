# How an LLM Picks the Next Word in a Drilling Report

**When an LLM writes the next part of a drilling report, how does it decide what comes next?**

This project answers that with a real experiment, not an illustration. A language model runs locally on a laptop. We give it the start of a Daily Drilling Report (DDR) entry, record exactly what it computes at each step, and turn that recording into a short vertical video (1080 × 1920, 60 fps). Its length follows the number of generated tokens: about 50 s for the current 2-token run, up to about 60 s for 10 tokens.

> **All token candidates, probabilities, rankings, and selections shown in this project are captured from an actual local LLM inference run. They are not manually authored for the animation.**

---

## Provenance of the rendered video

| | |
|---|---|
| **Model** | `Qwen/Qwen2.5-1.5B-Instruct` (revision `989aa798…`), float32 |
| **Device** | `mps` (Apple M4 Pro), local, no cloud API |
| **DDR context** (synthetic) | *Drilled 8½ in. hole section to 12,450 ft MD. Performed flow check, well static. Circulated bottoms up. Pump pressure remained* |
| **Prompt mode** | `raw-text` (the model sees exactly the DDR text, nothing else) |
| **Generation mode** | top-p sampling |
| **Temperature** | 0.70 |
| **Top-p** | 0.90 |
| **Seed** | 42 |
| **Steps** | until the model ends the sentence (`--until-sentence-end`, cap 40): 2 tokens |
| **Result (sampling)** | `… Pump pressure remained` **` constant.`** |
| **Result (greedy, same context)** | `… Pump pressure remained` **` constant at 1,000 psi.`** |
| **Software** | Python 3.11.9 · torch 2.14.0 · transformers 5.17.0 · numpy 2.4.6 |

> **Any number in the generated text is predicted by the model, not a hydraulics calculation or a measurement.** Greedy decoding on this context writes "1,000 psi", but the DDR input contains no flow rate, mud weight, rheology or string geometry, so that value is physically unconstrained. An earlier context wording (release v0.2.1) led the sampled run to "3,800 psi" instead. When the generated text contains a number, the video shows this warning on screen.

The full record is in `data/generated/inference_trace.json`. Sampling did not always take the most likely token: at step 2 it selected the **rank-2** token `.` (41.0% sampling probability) over ` at` (43.7%), ending the sentence where greedy decoding continues with "at 1,000 psi.". The video keeps that result as captured. The stopping rule (stop at the first token that ends a sentence) was fixed before looking at any output.

The DDR examples in `data/input/ddr_contexts.json` are **synthetic, written for education**. They don't describe any real well or operator. They were written before any model output was seen, and none has been edited to steer the model. The drilling context was reworded once, for operational correctness (QA/QC finding F3). A flow check is done with the pumps off, so the original wording ("…Circulated bottoms up and performed flow check. Pump pressure remained") put a pump-pressure observation after it. The flow check now comes first. The new wording was fixed before re-running the model, and the earlier run is preserved in release v0.2.1.

---

## The ideas, in plain language

**Token.** LLMs don't read words. They read *tokens*: chunks of text from a fixed dictionary. A token can be a whole word (` pressure`), part of a word (`Dr` + `illed`), a single digit (`1`, `2`, `4`, `5`, `0`), punctuation (`,`), a space, or a special control marker. In our DDR text, 20 words became **36 tokens**. Many tokens begin with a space; the video shows that space as `␠` (e.g. `␠pressure`) so it isn't hidden.

**Tokenizer.** The tool that splits text into tokens. It is fixed and deterministic, and it belongs to the model.

**Token ID.** Each token's number in the model's dictionary. ` pressure` is 7262. The model only ever sees these numbers.

**Logit (score).** For the next position, the model produces one raw score for **every** entry in its vocabulary. For Qwen2.5 that is 151,936 scores: 151,665 real tokens plus 271 unused padding slots, whose total probability is below 0.00002%. A higher score means the model considers that token more fitting. Scores aren't probabilities: they can be any number.

**Softmax → probability.** Softmax turns the scores into probabilities between 0 and 100% that add up to 100%. **Softmax does not choose anything**; it only converts.

**Temperature.** A dial applied before softmax. Below 1 (we use 0.70), likely tokens become even more likely (the distribution *sharpens*). Above 1, it flattens. Temperature adds **no randomness by itself**; it only reshapes the probabilities used by the next steps. In our run, ` constant` went from 31.0% (the model's own distribution) to 52.0% at T = 0.70.

**Top-p (nucleus).** Rank the tokens from most to least likely and keep adding them until their probabilities add up to at least *p* (we use 90%). Only these tokens are *eligible*; everything else is *excluded*. The eligible probabilities are then rescaled to add up to 100% again. That's why the video shows two different numbers:

- **model probability**: the model's own distribution (temperature 1);
- **sampling probability**: after temperature, top-p filtering, and rescaling. This is what the random draw actually uses.

**Greedy decoding.** Always take the #1 token. Deterministic, with no randomness, no temperature, and no top-p.

**Stochastic sampling.** Draw one eligible token at random, weighted by sampling probability. The top token is the most likely pick but **not guaranteed**. We fix the random seed so the run is repeatable.

**Autoregressive generation.** The selected token is appended to the text, and the whole process repeats for the next token. Text grows one token at a time.

---

## Technical detail

For raw logits $z$ over vocabulary $V$ and temperature $T > 0$:

$$p_\text{model}(i) = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad p_T(i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Sort by $p_T$ descending. The nucleus $N$ is the smallest prefix whose cumulative $p_T$ reaches $p$: a token is kept if the cumulative probability of the tokens ranked above it is still $< p$, so the top token is always kept. Then:

$$p_\text{sample}(i) = \frac{p_T(i)}{\sum_{k\in N} p_T(k)} \text{ for } i \in N,\quad 0 \text{ otherwise}$$

and the token is drawn with `numpy.random.default_rng(seed).choice(V, p=p_sample)`.

Implementation notes (verified, not assumed):

- All maths runs in float64 NumPy on the model's float32 logits (`src/inference/sampling.py`). Ranking ties use a stable sort (lower token ID first).
- The nucleus matches Hugging Face's own `TemperatureLogitsWarper` + `TopPLogitsWarper` exactly at every step; a test checks this.
- **Temperature 0**: dividing by zero is never attempted. `--mode sampling --temperature 0` is rejected with a message pointing to `--mode greedy` (the T → 0 limit). Greedy mode ignores temperature and top-p and records them as `null`.
- **No other logits processors are applied.** Qwen's `generation_config` sets `repetition_penalty=1.1, top_k=20, top_p=0.8`, and `model.generate()` applies these even with `do_sample=False`. We verified that `generate()` with default settings writes " constant at 37 psi. No gas was…", which differs from our greedy trace from step 4 on (the repetition penalty lowers tokens already in the context, such as the digits in "12,450"). With `repetition_penalty=1.0`, `generate()` reproduces our greedy trace token for token. This project deliberately shows the plain pipeline.
- Each step runs a full forward pass over the whole sequence (no KV cache) under `torch.inference_mode()`, which matches `logits = model(context)` literally.
- **Reproducibility.** Python `random`, NumPy, and PyTorch are all seeded, and re-running on the same machine reproduced the trace exactly. Logits can differ in the last digits across devices, dtypes, and library versions, so bit-identical traces are only expected on the same setup.

### Prompt formatting matters

`--prompt-mode raw-text` (default) feeds the DDR text as-is. `--prompt-mode chat-template` wraps it with the tokenizer's official chat template (system + user instruction), placing the DDR text at the start of the assistant turn so the model continues it. **Different prompt formatting gives a different probability distribution.** With the chat template, step 1 picks ` stable` (54.7%) instead of ` constant` (31.0% in raw-text mode). The trace always stores both `display_context` (what viewers see) and `actual_model_input` (the exact decoded model input, including any special tokens), and flags which input tokens belong to the DDR text.

### Trace format (`data/generated/inference_trace.json`, schema 1.1)

- `metadata`: model, revision, `tokenizer_vocab_size`, device, dtype, modes, temperature, top_p, seed, versions, and the exact method used for each quantity
- `source_context.synthetic`: `true` only when the contexts file declares it; `null` for `--context` text or files that don't say. The video labels the text accordingly and never assumes "synthetic".
- `display_context`, `actual_model_input`, `input_tokens[]` (id, raw tokenizer string, exact decoded text, UI-safe display text, `in_tokenizer`)
- `steps[]`, each with `context_before`/`context_after`, `appended_text` (the text this step added; a character split across tokens, such as an emoji, is credited to the token that completes it), `nucleus_size`, `nucleus_complete`, omitted probability mass, and `top_candidates[]`: the top 10 **plus the nucleus** (up to `--max-export`, default 1,000) **plus the selected token**, each with `raw_logit`, `scaled_logit`, `model_probability`, `temperature_probability`, `cumulative_probability`, `inside_top_p`, `sampling_probability`, `selected`
- `selected_token` and `generated_text`

A very large nucleus (for example top-p = 1.0, where every one of the 151,936 output rows is eligible) is exported only up to `--max-export` and marked `nucleus_complete: false`; the maths still runs over the full vocabulary. Output rows beyond the tokenizer (padding) are exported with `in_tokenizer: false` and empty text rather than an invented token string. Schema 1.0 traces are rejected with a message to re-capture.

The full-vocabulary logits for every step are saved next to the trace as `*.logits.npz` (git-ignored), and the tests recompute every JSON probability from them. The trace is validated on write (`src/inference/schemas.py`) and again on load in the video (`src/video/data/loadInferenceTrace.ts`). Validation checks structure and consistency: probabilities must follow from the recorded logits, the selected token must match its candidate record, all numbers must be finite, and the per-step text must reproduce the generated text.

---

## Usage

### Install

```bash
# Python 3.11+ (exact tested versions are pinned in requirements.txt)
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Node 22.12+ (required by the locked Vitest; 22.18 used here)
npm install
```

**Model download.** On first run, Hugging Face downloads `Qwen/Qwen2.5-1.5B-Instruct` (~3 GB) into `~/.cache/huggingface`, at the pinned commit `989aa798…` so a later upstream update can't silently change the weights (override with `--revision`; other models default to their latest revision). Weights are loaded from safetensors files only, never from pickled `.bin` files. After that everything runs offline (`HF_HUB_OFFLINE=1`). Float32 needs ~7 GB of RAM; use `--dtype bfloat16` on smaller machines (this changes the logits slightly). The device is chosen automatically in the order CUDA → MPS → CPU (override with `--device`).

### Capture the traces

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
| Temperature / top-p / seed | `--temperature 0.9 --top-p 0.95 --seed 7` |
| Greedy vs sampling | `--mode greedy` / `--mode sampling` |
| Number of tokens | `--steps 8`, or `--until-sentence-end` to stop when the model ends the sentence (`--steps` is then the cap). The video's loop scene grows automatically. |
| Prompt format | `--prompt-mode chat-template` |

The video always renders `data/generated/inference_trace.json`, which must be a sampling trace because the video explains temperature and top-p. After capturing a new trace, re-render; there is nothing to edit by hand.

### Preview and render

```bash
scripts/preview_video.sh          # Remotion Studio
scripts/render_video.sh           # → out/next_token_ddr.mp4
node scripts/render_stills.mjs 1200 2800   # QA stills → out/stills/
```

### Tests and checks

```bash
.venv/bin/python -m pytest        # 87 tests: maths, top-p vs Hugging Face, tokenization, schema and tamper detection, recompute from logits,
                                  #   review regressions (incl. one top-p = 1 capture with the cached 0.5B model; deselect with -m 'not slow')
npm test                          # 9 tests: loader serves the trace unchanged, rejects old/tampered traces, keeps low-ranked selections visible
npm run typecheck && npm run lint && .venv/bin/ruff check src tests
```

---

## Repository layout

```
data/input/ddr_contexts.json          synthetic DDR examples
data/generated/inference_trace*.json  captured traces (source of truth)
src/inference/                        device.py · tokenizer_utils.py · sampling.py · schemas.py · capture_trace.py
src/video/data/loadInferenceTrace.ts  the ONLY place the video reads model output
src/video/components/                 ProbabilityChart/Bar, TokenChip, TopPNucleus, SelectionClaw, HUD, CodeEditor, …
src/video/scenes/                     S1Question … S10Final
tests/                                pytest suite
scripts/                              capture / preview / render helpers
```

## What is simplified

- The **"LOCAL LLM" block** and **selection claw** are labelled *CONCEPTUAL VIEW*. They don't depict real neural activity, and the claw's sweep is decorative: the actual choice is one weighted random draw.
- The **code card** is labelled *SIMPLIFIED* pseudocode and is not the Hugging Face implementation.
- Charts show the top 8 candidates. The share held by the rest of the vocabulary is printed below the chart.

## Limitations

- A 1.5B-parameter general model has no drilling-domain training guarantees. Its continuations are plausible text, not engineering judgement.
- The sentence-end rule is a heuristic applied to the generated text so far: `!`, `?` or a newline ends the sentence, and so does a `.` unless it follows a digit (possible decimal point) or a DDR abbreviation such as `in.`, `ft.` or `approx.`. In those cases generation continues.
- The loop scene plays the first three later steps at full pace and the rest in fast-forward, to keep the video near 60 s.
- The greedy trace is captured for comparison, but the current video renders only the sampling trace.

## Licence

Code and documentation: [MIT](LICENSE). The model, Qwen2.5-1.5B-Instruct, is released by the Qwen team under Apache 2.0 and is downloaded separately, not redistributed here.
