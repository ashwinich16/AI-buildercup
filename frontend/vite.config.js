import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, API calls go to the FastAPI server on port 8000.
// In production, FastAPI serves the built files itself (one Cloud Run service).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
