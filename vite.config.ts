import { resolve } from "node:path";
import { defineConfig } from "vite";

// The demo pages live in web/ and import the engine straight from src/.
export default defineConfig({
  root: "web",
  base: "./",
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        graph: resolve(import.meta.dirname, "web/index.html"),
        taylor: resolve(import.meta.dirname, "web/taylor.html"),
      },
    },
  },
});
