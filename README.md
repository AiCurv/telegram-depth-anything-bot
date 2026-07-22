# telegram-depth-anything-bot

A single-user Telegram bot that runs **Depth Anything V2** monocular depth estimation on any photo (or album of up to 10) you send it. Heavy compute is offloaded to GitHub Actions (free unlimited minutes on public repos), while a Cloudflare Worker + KV acts as the firewall, album buffer, and dispatcher.

```
Telegram photo ──▶ Cloudflare Worker (200ms) ──▶ inline keyboard with 2 buttons
                       │                                  │
                       │                                  └─ user taps 🎨 HD Color or 🏁 HD Grayscale
                       │                                       │
                       │                                       └─ callback_query ──▶ Worker
                       │                                              │
                       │                                              ├─ answerCallbackQuery (instant, kills spinner)
                       │                                              ├─ editMessageText ("⏳ Generating...")
                       │                                              └─ repository_dispatch ──▶ GitHub Actions runner
                       │                                                                          │
                       │                                                                          ├─ pulls HF weights (cached)
                       │                                                                          ├─ runs Depth Anything V2 Large
                       │                                                                          └─ sendPhoto / sendMediaGroup back to chat
                       │
                       ├─ user locked via ALLOWED_TELEGRAM_USER_ID
                       └─ KV buffers album photos for 2500ms → ONE inline keyboard per album
```

## Architecture

| Layer | Tech | Role |
|------|------|------|
| Edge | Cloudflare Worker (`src/worker.js`) | Auth firewall + KV album buffer + inline keyboard dispatcher. Returns HTTP 200 within ~200ms. |
| State | Cloudflare KV (`ALBUM_BUFFER` namespace) | Per-photo keys (album buffer) + `session:{chat_id}:{msg_id}` keys (10-min TTL, hold photo_ids for callback retrieval). |
| Queue | GitHub `repository_dispatch` API | Decouples webhook from compute. Survives retries. |
| Compute | GitHub Actions (`ubuntu-latest`) | Pulls HF weights, runs inference (model loaded once per album), replies via Telegram Bot API. |
| ML | Hugging Face `transformers` pipeline | Depth Anything V2 **Large** (hardcoded — Small retired). |

### Why this shape

- **Cloudflare Worker is the only public endpoint.** It does a constant-time user-ID check and silently drops everyone else with HTTP 200 — attackers cannot even tell the bot is alive.
- **KV buffers albums across instances.** Each album photo is stored under its own KV key (`album:{media_group_id}:{message_id}`); a 1500ms debounced flush gathers all siblings via `list({ prefix })` and fires ONE dispatch per album. This fixes the previous Vercel in-memory buffer, which lost photos when album POSTs landed on different Vercel isolates.
- **The repo is public** so GitHub Actions minutes are unlimited. The dispatch event still requires the PAT, so a stranger cannot fire it through the GitHub API.
- **Heavy model weights are cached** under `~/.cache/huggingface` so warm runs boot in <2 min instead of pulling ~1.5GB every time.
- **Cloudflare locks the user ID**, GitHub locks the dispatch, Telegram locks the bot token — three independent gates, all secrets in Worker secrets, zero secrets in code.

## Files

```
.
├── src/
│   └── worker.js                  # Cloudflare Worker: auth + KV album buffer + dispatcher
├── .github/workflows/
│   └── depth_pipeline.yml         # Async runner
├── scripts/
│   ├── process_depth.py           # Telegram <-> HF inference bridge (batch-aware)
│   └── requirements.txt
├── wrangler.toml                  # Cloudflare Worker + KV namespace config
├── package.json
├── .env.example                   # All required Worker secrets
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
| `TELEGRAM_BOT_TOKEN` | Same token as above. Used by the runner to call `sendPhoto` / `sendMediaGroup`. |

(`GH_PAT_TOKEN` is not needed in the runner; the workflow uses the default `GITHUB_TOKEN` for checkout, and the dispatch is fired from the Cloudflare Worker.)

### 3. Cloudflare Worker deployment

1. Install wrangler: `npm install -g wrangler`
2. Authenticate with your Cloudflare API token + account ID:

   ```bash
   export CLOUDFLARE_API_TOKEN=cfat_...
   export CLOUDFLARE_ACCOUNT_ID=...
   ```

3. Create the KV namespace:

   ```bash
   wrangler kv namespace create ALBUM_BUFFER
   ```

   Paste the returned `id` into `wrangler.toml` under `[[kv_namespaces]] → id`.

4. Set Worker secrets:

   ```bash
   wrangler secret put TELEGRAM_BOT_TOKEN
   wrangler secret put ALLOWED_TELEGRAM_USER_ID
   wrangler secret put GH_PAT_TOKEN
   wrangler secret put GITHUB_REPO_OWNER
   wrangler secret put GITHUB_REPO_NAME
   ```

   (`GH_PAT_TOKEN` is a GitHub classic PAT with `repo` + `workflow` scopes.)

5. Deploy:

   ```bash
   wrangler deploy
   ```

   Copy the resulting Worker URL (e.g. `https://telegram-depth-anything-bot.<your-subdomain>.workers.dev`).

### 4. Wire Telegram to the Worker

The webhook must allow BOTH `message` and `callback_query` updates — the
latter is what Telegram sends when the user taps an inline keyboard button.

```bash
curl "https://api.telegram.org/bot<TOKEN>/setWebhook" \
     -d "url=https://<your-worker>.workers.dev/" \
     -d "allowed_updates=%5B%22message%22%2C%22callback_query%22%5D"
```

Verify:

```bash
curl "https://api.telegram.org/bot<TOKEN>/getWebhookInfo"
```

You should see `"url": "https://<your-worker>.workers.dev/"` and `"pending_update_count": 0`.

### 5. Send a photo

Send a photo (or album of up to 10) to the bot. Within ~3 seconds the bot replies with two inline buttons:

- 🎨 **HD Color** — inferno colormap (warm = near, cool = far)
- 🏁 **HD Grayscale** — pure 0–255 grayscale

Tap one. The button spinner disappears instantly (the Worker calls `answerCallbackQuery` as line 1 of its callback handler), the button card updates to `⏳ Generating HD Depth Map...`, and ~1–3 minutes later the depth map arrives as a reply.

## Albums

Send multiple photos as a single Telegram album and the bot will batch-process them. The Cloudflare Worker buffers album photos (sharing the same `media_group_id`) in KV for 2500ms, then the leader (smallest message_id) writes a single session entry to KV and shows ONE inline keyboard for the whole album. When you tap a color button, all photos in the album are processed in one GitHub Actions run, and the result comes back as a single `sendMediaGroup` album reply.

**How KV buffering works:** each album POST writes its photo to a unique KV key (`album:{media_group_id}:{message_id}`) with photo metadata in KV's inline metadata field. Each POST also schedules `ctx.waitUntil(flush at t+2500ms)`. The flush elects a leader (smallest message_id) — only the leader proceeds. The leader does a 4-try re-gather loop (500ms apart) to wait for KV eventual consistency across colos, then writes the consolidated session (`session:{chat_id}:{first_msg_id}`, 10-min TTL) and sends the inline keyboard. A sentinel key (`album:{id}:flushed`) suppresses double-flush if a sibling's retry loop sees a stable count later.

**Why answerCallbackQuery must be line 1 of the callback handler:** Telegram's client shows a small loading spinner on the tapped inline keyboard button. The spinner auto-stops when `answerCallbackQuery` is received OR after ~10 seconds (whichever comes first). If the Worker does any slow work first (KV read, GitHub dispatch), the user sees the spinner for the full 10s, which feels broken. Calling `answerCallbackQuery` as the first HTTP call kills the spinner in <100ms.

**Limitations:** KV is eventually consistent across Cloudflare colos, so in the rare case that album photos land on different colos, a sibling's write might not yet be visible when the flush fires. The leader's retry loop handles this. Orphaned KV keys auto-expire after 5 minutes. For bulletproof cross-colo batching, use Durable Objects (requires Workers Paid plan, $5/mo).

## Usage

No more caption flags. Send a photo (or album), tap a button. That's it.

### The flow

1. **Send a photo or album.** No caption needed.
2. **The bot replies with an inline keyboard** containing two buttons:
   - 🎨 **HD Color** — inferno colormap
   - 🏁 **HD Grayscale** — pure 0–255 grayscale PNG
3. **Tap a button.** The spinner disappears instantly. The button card updates to `⏳ Generating HD Depth Map...`.
4. **Wait ~1–3 minutes.** The depth map (or album of depth maps) arrives as a reply.

### What happened to /hd /fast /gray /color?

The caption-flag parser has been removed. The model is **always** `Depth-Anything-V2-Large-hf` (the Small model has been retired — it wasn't worth the quality tradeoff for the speed savings on a single-user bot). Color is chosen via the inline keyboard buttons, which is faster and less error-prone than typing flags.

### /start and /help

These still work and describe the new button-based flow.

## Models

Only one model is in use:

- **Large**: [`depth-anything/Depth-Anything-V2-Large-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) — 335M params, ~10–20s inference on CPU. Hardcoded in `scripts/process_depth.py` and `src/worker.js`.

The Small model (25M params, ~3s inference) has been retired. The speed savings weren't worth the quality tradeoff for a single-user bot, and removing it simplified the UI to two buttons instead of four.

## Security notes

- The Worker **always returns HTTP 200**, even for unauthorized users. This is intentional — returning 401/403 leaks that the endpoint exists and validates IDs.
- `ALLOWED_TELEGRAM_USER_ID` is checked first. If it doesn't match, the Worker exits before touching the GitHub API or KV, so abuse from a stranger costs you zero GitHub Actions minutes and zero KV operations.
- The PAT lives in Cloudflare Worker secrets only. It is never written to the repo. Rotate it by revoking the classic token in GitHub settings and re-running `wrangler secret put GH_PAT_TOKEN`.
- The repo being public does NOT leak the PAT — it's only used to authenticate the `repository_dispatch` POST, which the Worker performs server-side.

## Limits

- GitHub Actions workflow run: 20 min hard timeout (set in `depth_pipeline.yml`). The Large model on a cold cache can take ~6–9 min; warm runs are ~3–4 min. A 10-photo album on a warm cache adds ~15s/photo = ~150s.
- Cloudflare Worker free plan: 100,000 requests/day, 10ms CPU per invocation. We use `ctx.waitUntil` for the 1500ms debounce which doesn't count against CPU time.
- Cloudflare KV free plan: 100,000 reads/day, 1,000 writes/day, 1,000 deletes/day, 1,000 list operations/day. A 10-photo album uses ~10 writes + 1 list + 10 deletes = ~21 operations — well within limits for a personal bot.
- Telegram `getFile`: 20MB file size limit. Photos are well below this.
- Telegram `sendMediaGroup`: max 10 items per album. The Worker caps at 10 photos per dispatch defensively.

## Cost

| Component | Cost |
|-----------|------|
| Cloudflare Workers + KV | Free (well within free plan limits for a single-user bot) |
| GitHub Actions | Free (unlimited on public repos) |
| Hugging Face | Free (model is open weights) |
| Telegram Bot API | Free |

Total: $0/mo for the configured single-user workload.

## Troubleshooting

- **Bot doesn't reply at all.** Check `getWebhookInfo` — if `last_error_message` is set, the Worker is throwing. Tail Worker logs with `wrangler tail`. Also verify that `allowed_updates` includes BOTH `message` and `callback_query` (the inline keyboard taps won't reach the Worker otherwise).
- **Button tap shows a 10-second spinner before anything happens.** This means `answerCallbackQuery` is not being called first in the callback handler. Check `wrangler tail` for `[callback]` log lines — the handler should call `answerCallbackQuery` before any KV read or GitHub dispatch. The current code does this correctly; if you see this symptom, you may be running an older deploy.
- **Bot says "Queued" / "⏳ Generating..." but never delivers the depth map.** Check the Actions tab in GitHub — the workflow run is likely red. Open the log; 99% of the time it's a missing `TELEGRAM_BOT_TOKEN` secret or a HF download timeout (re-run the job, the cache will speed it up the second time).
- **Album came back as individual photos instead of one album.** This means the KV buffer didn't gather all siblings — check `wrangler tail` for `[flush] album ...` messages. Could be a KV eventual-consistency delay across colos; for a guaranteed fix, upgrade to Workers Paid and use Durable Objects.
- **Button says "This button expired."** The session entry (10-min TTL) expired before you tapped. Send the photo(s) again to get a fresh inline keyboard.
- **Stranger sent a photo and burned my Actions minutes.** They can't — the Worker firewall drops them with HTTP 200 before the dispatch fires. Verify `ALLOWED_TELEGRAM_USER_ID` is your numeric ID, not your `@username`.
- **Cold cache takes forever.** First run after cache invalidation downloads ~1.5GB for the Large model. Subsequent runs use the cache and boot in <2 min.

## License

MIT.
