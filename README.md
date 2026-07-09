# Signal — AI Stock Screener

A stock screener that pulls daily price data, computes technical signals
(momentum, moving averages, volatility, volume trends), fits a small machine
learning model to estimate short-term bullish potential, and suggests
take-profit targets. It ships in two forms that share the same logic and the
same defaults:

| | Where it runs | Data | ML model |
|---|---|---|---|
| **`python/`** — terminal edition | Your machine, via `python screener.py` | Yahoo Finance via [`yfinance`](https://pypi.org/project/yfinance/) (free, no key) | scikit-learn `GradientBoostingClassifier` |
| **`web/`** — dashboard edition | Entirely in the browser, no backend | Twelve Data free API (optional, needs a free key) or a built-in offline simulator | Logistic regression trained from scratch in plain JS |

Neither version needs a paid API key to try out. The web app defaults to a
seeded, deterministic "simulated" market so it works instantly, offline, with
zero setup — flip a switch to point it at real data once you have a free
Twelve Data key.

**Nothing here is financial advice.** Bullish probabilities and take-profit
targets are model estimates based on historical patterns, not guarantees.

---

## 1. Terminal edition (Python)

### Setup

```bash
cd python
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

### Run it

```bash
# Scan the built-in default watchlist
python screener.py

# Scan your own tickers, show the top 5
python screener.py --tickers AAPL,MSFT,NVDA,AMD,SHOP --top 5

# Also show tickers that move like NVDA (return correlation)
python screener.py --similar NVDA

# Keep scanning on a loop, terminal-scanner style
python screener.py --watch --interval 60
```

Run `python screener.py --help` for every flag (history window, the forward
horizon and return threshold used to label "bullish" when training, a
take-profit multiplier, and more).

### How it works

1. **`data.py`** pulls daily OHLCV bars per ticker with `yfinance`. If a live
   fetch fails (no network, rate-limited, ticker not found, or you're running
   in an offline/sandboxed environment), it automatically falls back to a
   seeded synthetic price series so the tool still runs end-to-end — every
   synthetic row is clearly labeled `synthetic` in the output so it's never
   mistaken for real data.
2. **`indicators.py`** computes SMA/EMA, RSI, MACD, Bollinger %B, ATR,
   annualized realized volatility, and a volume trend ratio.
3. **`model.py`** labels every historical bar across your whole watchlist
   (did price rise more than `--target-return` within `--horizon` trading
   days?), pools the rows, and fits a small `GradientBoostingClassifier`.
   It then scores each ticker's latest bar into a bullish probability, and
   derives two take-profit targets plus a stop-loss from ATR, scaled by how
   confident the model is.
4. `--similar TICKER` ranks the rest of your watchlist by correlation of
   daily returns with that ticker.

---

## 2. Dashboard edition (web)

A static site — plain HTML/CSS/JS, no build tools, no framework, no backend.
That's deliberate: it makes local testing trivial and deployment to GitHub
Pages or Vercel a non-event.

### Run it locally

Any static file server works. From the `web/` folder:

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

or, if you have Node:

```bash
npx serve web
```

You can also just double-click `web/index.html` — it works from the
filesystem too, since simulated mode makes no network calls.

### Using it

- **Watchlist & settings** (left panel) — edit tickers, pick a history
  window, and tune the forward horizon / target return / take-profit reach
  the model uses.
- **Simulated vs. Live data** — Simulated is on by default (instant, offline,
  deterministic per ticker). Switch to **Live** and paste a free API key from
  [twelvedata.com](https://twelvedata.com/pricing) to pull real daily bars
  directly from your browser. Any ticker whose live fetch fails automatically
  falls back to simulated data for that ticker, and says so.
- **Run Scan** — fetches/generates bars, computes indicators, trains a fresh
  logistic regression in-browser, and ranks every ticker by bullish
  probability.
- **Ticker detail** — click a row to see its price/SMA chart, full signal
  breakdown, take-profit and stop levels, and a "stocks that move like this
  one" panel driven by return correlation.
- **Scanner terminal** (right panel) — a live, scrolling log of the actual
  scan pipeline (fetch → indicators → training epochs → scores). It's not
  decorative text; every line is a real event from that run.

### How the in-browser ML works

`js/model.js` standardizes the technical features and fits a logistic
regression by batch gradient descent, trained fresh in your browser on
pooled historical rows from your current watchlist — the same labeling
scheme as the Python edition (forward return over `horizon` days versus
`targetReturn`). No ML library or server round-trip required.

### Optional: local smoke test

A small Node/JSDOM test drives the whole UI (load → scan → select a row →
verify the terminal streamed output) without a real browser:

```bash
cd web
npm install jsdom   # only needed to run this test
node test_dom.js
```

---

## 3. Deploying the web dashboard

Because it's a static site, both options are effectively "point the host at
the `web/` folder."

### GitHub Pages

A ready-to-go workflow lives at `.github/workflows/deploy-pages.yml`. It
publishes the contents of `web/` (minus `node_modules` and the test file)
every time you push to `main`.

1. Push this repo to GitHub.
2. In **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Push to `main` (or run the workflow manually from the **Actions** tab).
4. Your dashboard will be live at `https://<you>.github.io/<repo>/`.

### Vercel

No config file needed — just point Vercel at the `web/` folder as the
project root:

```bash
npm i -g vercel   # if you don't have it already
cd web
vercel            # first deploy, follow the prompts
vercel --prod     # promote to production
```

Or via the dashboard: **Import Project** → select this repo → set **Root
Directory** to `web` → deploy. No build command or output directory is
needed since it's static files.

---

## Project structure

```
ai-stock-screener/
├── python/
│   ├── screener.py       # CLI entry point
│   ├── data.py           # yfinance fetch + synthetic fallback
│   ├── indicators.py     # technical signal calculations
│   ├── model.py          # training set, GradientBoostingClassifier, scoring
│   └── requirements.txt
├── web/
│   ├── index.html
│   ├── css/style.css     # dark, gold-accented "terminal" theme
│   ├── js/
│   │   ├── indicators.js # same formulas as indicators.py, in JS
│   │   ├── data.js       # Twelve Data fetch + seeded simulator
│   │   ├── model.js      # logistic regression trained in-browser
│   │   ├── terminal.js   # scrolling live-scan log panel
│   │   └── app.js        # wires it all together
│   └── test_dom.js       # optional JSDOM smoke test
└── .github/workflows/deploy-pages.yml
```

## Disclaimer

This project is for educational/informational purposes. It does not provide
investment advice, and bullish-probability scores or take-profit/stop levels
are statistical estimates from a small model over historical patterns — they
can be wrong, and past patterns are no guarantee of future returns. Always do
your own research.
