import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
  server: {
    proxy: {
      "/healthz": "http://127.0.0.1:8000",
      "/v1": "http://127.0.0.1:8000",
    },
    // Avoids stale file-handle reads on macOS + Node 22.
    fs: { cachedChecks: false },
  },
});
