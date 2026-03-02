import { useEffect, useState, useCallback } from "react";
import { fetchHealth, generate, streamGenerate } from "./api";
import type { GenerateRequest } from "./api";
import { defaultGenerateParams } from "./defaults";

type Status = "checking" | "ready" | "error";

function App() {
  const [status, setStatus] = useState<Status>("checking");
  const [statusMessage, setStatusMessage] = useState("Checking…");
  const [prompt, setPrompt] = useState<string>(defaultGenerateParams.prompt);
  const [stream, setStream] = useState(true);
  const [params, setParams] = useState<{
    max_new_tokens: number;
    temperature: number;
    top_k: number;
    top_p: number;
    repetition_penalty: number;
  }>({
    max_new_tokens: defaultGenerateParams.max_new_tokens,
    temperature: defaultGenerateParams.temperature,
    top_k: defaultGenerateParams.top_k,
    top_p: defaultGenerateParams.top_p,
    repetition_penalty: defaultGenerateParams.repetition_penalty,
  });
  const [output, setOutput] = useState<string>("");
  const [streaming, setStreaming] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy");

  useEffect(() => {
    let cancelled = false;
    fetchHealth()
      .then((data) => {
        if (cancelled) return;
        if (data.model_loaded) {
          setStatus("ready");
          setStatusMessage("Model loaded");
        } else {
          setStatus("error");
          setStatusMessage("Model not loaded (train and restart API)");
        }
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setStatus("error");
        setStatusMessage(`Cannot reach API (${e.message ?? "network error"})`);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const buildRequest = useCallback((): GenerateRequest => {
    return {
      prompt,
      max_new_tokens: params.max_new_tokens,
      temperature: params.temperature,
      top_k: params.top_k,
      top_p: params.top_p,
      repetition_penalty: params.repetition_penalty,
      repeat_window: defaultGenerateParams.repeat_window,
      seed: defaultGenerateParams.seed,
    };
  }, [prompt, params]);

  const handleGenerate = useCallback(() => {
    const req = buildRequest();
    setStreaming(true);
    if (stream) {
      setOutput(req.prompt);
      streamGenerate(req, (chunk) => setOutput((prev) => prev + chunk))
        .catch((e: Error) => setOutput(`Error: ${e.message}`))
        .finally(() => setStreaming(false));
    } else {
      setOutput("Generating…");
      generate(req)
        .then((res) => setOutput(res.text ?? ""))
        .catch((e: Error) => setOutput(`Error: ${e.message}`))
        .finally(() => setStreaming(false));
    }
  }, [buildRequest, stream]);

  const handleCopy = useCallback(() => {
    if (!output.trim()) return;
    navigator.clipboard
      .writeText(output)
      .then(() => {
        setCopyLabel("Copied");
        setTimeout(() => setCopyLabel("Copy"), 1500);
      })
      .catch(() => {
        setCopyLabel("Copy failed");
        setTimeout(() => setCopyLabel("Copy"), 1500);
      });
  }, [output]);

  const hasOutput = output.length > 0;

  return (
    <div className="layout">
      <header className="header">
        <h1 className="logo">llm-lite</h1>
        <p className="tagline">Generate text with your trained causal transformer</p>
        <div className={`status ${status === "ready" ? "ready" : status === "error" ? "error" : ""}`} aria-live="polite">
          <span className="status-dot" />
          <span className="status-text">{statusMessage}</span>
        </div>
      </header>

      <main className="main">
        <section className="card input-card">
          <label htmlFor="prompt" className="label">
            Prompt
          </label>
          <textarea
            id="prompt"
            className="prompt-input"
            placeholder="Enter your prompt…"
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
          <div className="controls-row">
            <label className="checkbox-label">
              <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
              <span>Stream response</span>
            </label>
            <button type="button" className="btn btn-primary" onClick={handleGenerate} disabled={streaming}>
              Generate
            </button>
          </div>
          <details className="advanced">
            <summary>Generation settings</summary>
            <div className="settings-grid">
              <div className="setting">
                <label htmlFor="maxNewTokens">Max new tokens</label>
                <input
                  type="number"
                  id="maxNewTokens"
                  min={1}
                  max={1000}
                  value={params.max_new_tokens}
                  onChange={(e) => setParams((p) => ({ ...p, max_new_tokens: parseInt(e.target.value, 10) || 200 }))}
                />
              </div>
              <div className="setting">
                <label htmlFor="temperature">Temperature</label>
                <input
                  type="number"
                  id="temperature"
                  min={0}
                  max={2}
                  step={0.1}
                  value={params.temperature}
                  onChange={(e) => setParams((p) => ({ ...p, temperature: parseFloat(e.target.value) || 0.9 }))}
                />
              </div>
              <div className="setting">
                <label htmlFor="topK">Top-k</label>
                <input
                  type="number"
                  id="topK"
                  min={0}
                  max={500}
                  value={params.top_k}
                  onChange={(e) => setParams((p) => ({ ...p, top_k: parseInt(e.target.value, 10) || 50 }))}
                />
              </div>
              <div className="setting">
                <label htmlFor="topP">Top-p</label>
                <input
                  type="number"
                  id="topP"
                  min={0}
                  max={1}
                  step={0.05}
                  value={params.top_p}
                  onChange={(e) => setParams((p) => ({ ...p, top_p: parseFloat(e.target.value) || 0.9 }))}
                />
              </div>
              <div className="setting">
                <label htmlFor="repetitionPenalty">Repetition penalty</label>
                <input
                  type="number"
                  id="repetitionPenalty"
                  min={1}
                  max={2}
                  step={0.05}
                  value={params.repetition_penalty}
                  onChange={(e) =>
                    setParams((p) => ({ ...p, repetition_penalty: parseFloat(e.target.value) || 1.15 }))
                  }
                />
              </div>
            </div>
          </details>
        </section>

        <section className="card output-card">
          <div className="output-header">
            <span className="label">Output</span>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={handleCopy}
              disabled={!hasOutput}
              title="Copy to clipboard"
            >
              {copyLabel}
            </button>
          </div>
          <div className={`output ${hasOutput ? "has-content" : ""}`} aria-live="polite">
            {!hasOutput && <p className="output-placeholder">Generated text will appear here.</p>}
            {hasOutput && (
              <div className="output-content">
                {output}
                {streaming && <span className="cursor" aria-hidden />}
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="footer">
        <a href="/docs" target="_blank" rel="noopener noreferrer">
          API docs
        </a>
        <span className="sep">·</span>
        <a href="/healthz" target="_blank" rel="noopener noreferrer">
          Health
        </a>
      </footer>
    </div>
  );
}

export default App;
