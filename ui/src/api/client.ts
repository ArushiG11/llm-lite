import type { GenerateRequest, GenerateResponse, HealthResponse } from "./types";

/** Same-origin API base. When served at /ui, requests go to origin (e.g. /v1/generate). */
function getApiBase(): string {
  if (typeof window === "undefined") return "";
  return window.location.origin;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const r = await fetch(`${getApiBase()}/healthz`);
  return r.json();
}

export async function generate(params: GenerateRequest): Promise<GenerateResponse> {
  const r = await fetch(`${getApiBase()}/v1/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail ?? r.statusText);
  }
  return r.json();
}

export async function streamGenerate(
  params: GenerateRequest,
  onChunk: (chunk: string) => void
): Promise<void> {
  const r = await fetch(`${getApiBase()}/v1/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error((err as { detail?: string }).detail ?? r.statusText);
  }
  const reader = r.body?.getReader();
  if (!reader) throw new Error("No response body");
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") return;
        onChunk(payload);
      }
    }
  }
}
