export interface HealthResponse {
  ok: boolean;
  ckpt: string;
  model_loaded: boolean;
}

export interface GenerateRequest {
  prompt: string;
  max_new_tokens: number;
  temperature: number;
  top_k: number;
  top_p: number;
  repetition_penalty: number;
  repeat_window: number;
  seed: number | null;
}

export interface GenerateResponse {
  text: string;
}
