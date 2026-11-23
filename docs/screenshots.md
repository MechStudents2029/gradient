# Screenshots

The images in `docs/screenshots/` are taken from a local production build, not
the live site, so they always match the code in the repo.

```sh
npm run build
npx vite preview --port 4173 --strictPort &
```

Then drive the three pages with headless Chrome via `playwright-core`
(`npm i playwright-core` in a scratch folder, so it stays out of this
project's dependencies):

```js
import { chromium } from "playwright-core";

const browser = await chromium.launch({ executablePath: "/usr/bin/google-chrome" });
const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
const base = "http://localhost:4173/";

// Graph view: squared error, a few steps into the backward pass.
await page.goto(base);
await page.selectOption("#preset", { label: "squared error" });
for (let i = 0; i < 4; i++) await page.click("#step");
await page.screenshot({ path: "docs/screenshots/graph-view.png", fullPage: true });

// Taylor view: the default sin x preset.
await page.goto(base + "taylor.html");
await page.waitForTimeout(1200);
await page.screenshot({ path: "docs/screenshots/taylor-view.png", fullPage: true });

// Training: XOR, seed 1, after ~8 s of training (about 800 epochs).
await page.goto(base + "train.html");
await page.click("#start");
await page.waitForTimeout(8000);
await page.screenshot({ path: "docs/screenshots/train-view.png", fullPage: true });

await browser.close();
```

The squared-error preset is used for the graph shot because it has edges that
skip a column (`y`, `b`, and the constant `-1`), which shows the edge routing.
Training is seeded, so the XOR boundary looks the same on every run; only the
epoch count depends on how fast the frames come in.
