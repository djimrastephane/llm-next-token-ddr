// Mirrors the Python trace written by src/inference/capture_trace.py (schema 1.0).

export type TokenInfo = {
  token_id: number;
  raw_token: string;
  decoded_token: string;
  display_token: string;
  is_special: boolean;
};

export type InputToken = TokenInfo & { position: number; in_display_context: boolean };

export type Candidate = TokenInfo & {
  rank: number;
  raw_logit: number;
  scaled_logit: number;
  model_probability: number; // softmax(raw logits), temperature 1
  temperature_probability: number; // softmax(raw logits / T): what top-p is applied to
  cumulative_probability: number; // running sum of temperature_probability
  inside_top_p: boolean | null;
  sampling_probability: number | null; // renormalized within the nucleus; 0 outside
  selected: boolean;
};

export type SelectedToken = TokenInfo & {
  rank: number;
  raw_logit: number;
  model_probability: number;
  temperature_probability: number;
  sampling_probability: number | null;
};

export type Step = {
  step: number;
  context_before: string;
  vocab_size: number;
  forward_ms: number;
  nucleus_size: number | null;
  nucleus_temperature_mass: number | null;
  omitted_model_probability_mass: number;
  omitted_temperature_probability_mass: number;
  top_candidates: Candidate[];
  selected_token: SelectedToken;
  context_after: string;
};

export type Metadata = {
  model: string;
  model_display_name: string;
  model_revision: string | null;
  device: string;
  dtype: string;
  generation_mode: "greedy" | "sampling";
  prompt_mode: "raw-text" | "chat-template";
  temperature: number | null;
  top_p: number | null;
  seed: number;
  steps_requested: number;
  steps_generated: number;
  stop_reason: string;
};

export type InferenceTrace = {
  schema_version: string;
  metadata: Metadata;
  source_context: {
    id: string;
    category: string;
    title: string;
    facts: Record<string, string>;
    synthetic: boolean | null;
  };
  display_context: string;
  actual_model_input: string;
  display_context_token_start: number;
  input_tokens: InputToken[];
  steps: Step[];
  generated_text: string;
  final_display_context: string;
};
