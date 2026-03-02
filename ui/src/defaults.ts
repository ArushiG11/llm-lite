export const defaultGenerateParams = {
  prompt: "",
  max_new_tokens: 200,
  temperature: 0.9,
  top_k: 50,
  top_p: 0.9,
  repetition_penalty: 1.15,
  repeat_window: 64,
  seed: 42 as number | null,
} as const;
