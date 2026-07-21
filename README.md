# telegram-depth-anything-bot

A single-user Telegram bot that runs **Depth Anything V2** monocular depth estimation on any photo you send it. Heavy compute is offloaded to GitHub Actions (free unlimited minutes on public repos), while a tiny Vercel webhook acts as the firewall + dispatcher.

```
Telegram photo ──▶ Vercel webhook (200ms) ──▶ repository_dispatch ──▶ GitHub Actions runner
                                          │                            │
                                          │                            ├─ pulls HF weights (cached)
                                          │                            ├─ runs Depth Anything V2
                                          │                            └─ sendPhoto back to chat
                                          └─ user locked via ALLOWED_TELEGRAM_USER_ID
```

## Architecture

| Layer | Tech | Role |
|------|------|------|
| Edge | Vercel serverless function (`api/webhook.js`) | Auth firewall + dispatcher. Returns HTTP 200 within ~500ms. |
| Queue | GitHub `repository_dispatch` API | Decouples webhook from compute. Survives retries. |
| Compute | GitHub Actions (`ubuntu-latest`) | Pulls HF weights, runs inference, replies via Telegram Bot API. |
| ML | Hugging Face `transformers` pipeline | Depth Anything V2 Small (`/fast`) or Large (`/hd`). |

### Why this shape

- **Vercel webhook is the only public endpoint.** It does a constant-time user-ID check and silently drops everyone else with HTTP 200 — attackers cannot even tell the bot is alive.
- **The repo is public** so GitHub Actions minutes are unlimited. The dispatch event still requires the PAT, so a stranger cannot fire it through the GitHub API.
- **Heavy model weights are cached** under `~/.cache/huggingface` so warm runs boot in <2 min instead of pulling ~1.5GB every time.
- **Vercel locks the user ID**, GitHub locks the dispatch, Telegram locks the bot token — three independent gates, all secrets in env vars, zero secrets in code.

## Files

```
.
├── api/
│   └── webhook.js                 # Vercel serverless endpoint
├── .github/workflows/
│   └── depth_pipeline.yml         # Async runner
├── scripts/
│   ├── process_depth.py           # Telegram <-> HF inference bridge
│   └── requirements.txt
├── package.json
├── vercel.json
├── .env.example                   # All required env vars
└── README.md
```

## Setup

You only need to do this once. After the first deployment, sending a photo to the bot is enough.

### 1. Telegram bot

1. Talk to [@BotFather](https://t.me/BotFather), send `/newbot`, pick a name. Save the token as `TELEGRAM_BOT_TOKEN`.
2. Talk to [@userinfobot](https://t.me/userinfobot) to get your numeric user ID. Save as `ALLOWED_TELEGRAM_USER_ID`.

### 2. Public GitHub repo

```bash
gh repo create telegram-depth-anything-bot --public --source=. --push
```

Then in **Settings → Secrets and variables → Actions**, add:

| Secret | Value |
|--------|-------|
| `TELEGRAM_BOT_TOKEN` | Same token as above. Used by the runner to call `sendPhoto`. |

(`GH_PAT_TOKEN` is not needed in the runner; the workflow uses the default `GITHUB_TOKEN` for checkout, and the dispatch is fired from Vercel.)

### 3. Vercel deployment

1. Push the repo to GitHub (already done above).
2. In Vercel, **New Project → import the repo**.
3. Set Environment Variables (Project → Settings → Environment Variables):

| Key | Value |
|-----|-------|
| `TELEGRAM_BOT_TOKEN` | bot token |
| `ALLOWED_TELEGRAM_USER_ID` | your numeric Telegram user id |
| `GH_PAT_TOKEN` | GitHub classic PAT with `repo` + `workflow` scopes |
| `GITHUB_REPO_OWNER` | e.g. `your-github-username` |
| `GITHUB_REPO_NAME` | `telegram-depth-anything-bot` |

4. Deploy. Copy the production URL (e.g. `https://<your-project>.vercel.app`).

### 4. Wire Telegram to Vercel

```bash
curl "https://api.telegram.org/bot<TOKEN>/setWebhook" \
     -d "url=https://<your-project>.vercel.app/api/webhook" \
     -d "allowed_updates=%5B%22message%22%5D"
```

Verify:

```bash
curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"
```

You should see `"url": "https://<your-project>.vercel.app/api/webhook"` and `"pending_update_count": 0`.

### 5. Send a photo

Send a photo to your bot with caption `/hd` (large model, ~1–3 min) or `/fast` (small model, ~30–60s). You can combine model and color flags, e.g. `/fast /color` or `/hd /gray`. With no flags you get **HD + grayscale** by default.

## Usage

Flags can be combined in any order in a photo caption (or sent as a standalone text command before uploading).

### Model flags

| Flag | Model |
|------|-------|
| `/hd` or `hd` | Depth Anything V2 **Large** (default) |
| `/fast` or `fast` | Depth Anything V2 **Small** |

### Color flags

| Flag | Output |
|------|--------|
| `/gray`, `/grayscale`, or `gray` | Pure 0–255 grayscale PNG (default) |
| `/color`, `/inferno`, or `color` | Inferno-colormapped PNG |

### Examples

| Caption | Result |
|---------|--------|
| *(no caption)* | HD + grayscale |
| `/fast` | Small + grayscale |
| `/hd /color` | Large + inferno colormap |
| `/fast /gray` | Small + grayscale |
| `hd color` | Large + inferno (slashes optional) |
| `/color` | Large + inferno (model defaults to HD) |
| `/start` | Greeting + usage. |
| `/help`  | Usage summary. |

## Models

- **Small**: [`depth-anything/Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) — 25M params, ~3s inference on CPU.
- **Large**: [`depth-anything/Depth-Anything-V2-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) — 335M params, ~10–20s inference on CPU.

## Security notes

- The Vercel webhook **always returns HTTP 200**, even for unauthorized users. This is intentional — returning 401/403 leaks that the endpoint exists and validates IDs.
- `ALLOWED_TELEGRAM_USER_ID` is checked first. If it doesn't match, the function exits before touching the GitHub API, so abuse from a stranger costs you zero GitHub Actions minutes.
- The PAT lives in Vercel env vars only. It is never written to the repo. Rotate it by revoking the classic token in GitHub settings and updating the Vercel env var.
- The repo being public does NOT leak the PAT — it's only used to authenticate the `repository_dispatch` POST, which Vercel performs server-side.

## Limits

- GitHub Actions workflow run: 15 min hard timeout (set in `depth_pipeline.yml`). The Large model on a cold cache can take ~6–9 min; warm runs are ~3–4 min.
- Vercel function: 10s max duration. We never approach this — we ack within ~500ms.
- Telegram `getFile`: 20MB file size limit. Photos are well below this.
- Concurrency: the workflow uses `cancel-in-progress: false`, so back-to-back sends queue up rather than clobber each other.

## Cost

| Component | Cost |
|-----------|------|
| Vercel | Free (well under Hobby limits) |
| GitHub Actions | Free (unlimited on public repos) |
| Hugging Face | Free (model is open weights) |
| Telegram Bot API | Free |

Total: $0/mo for the configured single-user workload.

## Troubleshooting

- **Bot doesn't reply at all.** Check `getWebhookInfo` — if `last_error_message` is set, the Vercel function is throwing. Tail Vercel logs.
- **Bot says "Queued" but never delivers the depth map.** Check the Actions tab in GitHub — the workflow run is likely red. Open the log; 99% of the time it's a missing `TELEGRAM_BOT_TOKEN` secret or a HF download timeout (re-run the job, the cache will speed it up the second time).
- **Stranger sent a photo and burned my Actions minutes.** They can't — the Vercel firewall drops them with HTTP 200 before the dispatch fires. Verify `ALLOWED_TELEGRAM_USER_ID` is your numeric ID, not your `@username`.
- **Cold cache takes forever.** First run after cache invalidation downloads ~1.5GB for the Large model. Subsequent runs use the cache and boot in <2 min.

## License

MIT.
