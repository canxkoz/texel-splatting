import { defineConfig } from "vite";

export default defineConfig({
    server: { port: 3000 },
    base: "./",
    build: {
        target: "esnext",
        outDir: "site/demo",
        emptyOutDir: true,
    },
});
