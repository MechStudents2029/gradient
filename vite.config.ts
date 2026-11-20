import { defineConfig } from "vite";

// The visualizer lives in web/ and imports the engine straight from src/.
export default defineConfig({
  root: "web",
  base: "./",
  build: { outDir: "../dist", emptyOutDir: true },
});
