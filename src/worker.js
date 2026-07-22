/**
 * src/worker.js
 *
 * Cloudflare Worker that:
 *   1. Receives Telegram webhook payloads (messages AND callback_query events).
 *   2. Validates the sender against ALLOWED_TELEGRAM_USER_ID (single-user whitelist).
 *   3. Silently drops anyone else with HTTP 200 (so the bot cannot be probed).
 *   4. When photo(s) arrive:
 *      a. If album (media_group_id present): buffers photo file_ids in KV per
 *         photo key under `album:{media_group_id}:{message_id}` with a 2500ms
 *         debounce, then the leader (smallest message_id) flushes.
 *      b. If single photo: skips the debounce.
 *      c. After flush: writes a SESSION entry under `session:{chat_id}:{first_message_id}`
 *         (10-minute TTL) containing all file_ids, then sends a sendMessage
 *         with an INLINE KEYBOARD offering two buttons:
 *           - "🎨 HD Color"     callback_data=`process:inferno:{first_message_id}`
 *           - "🏁 HD Grayscale" callback_data=`process:gray:{first_message_id}`
 *   5. When `callback_query` arrives (user tapped a button):
 *      STEP 1 (IMMEDIATE, INLINE): Call answerCallbackQuery to dismiss the
 *              Telegram button loading spinner. This MUST be the first HTTP
 *              call — Telegram shows a 10s spinner otherwise.
 *      STEP 2 (INLINE): editMessageText on the inline keyboard message to
 *              show "⏳ Generating HD Depth Map...".
 *      STEP 3 (in ctx.waitUntil): Read the session from KV, fire
 *              repository_dispatch to GitHub Actions with
 *              model=Depth-Anything-V2-Large-hf and cmap (inferno|gray).
 *
 * Why KV instead of in-memory?
 *   Vercel serverless functions (the previous architecture) do not share
 *   in-memory state across invocations. Album photos that landed on different
 *   Vercel instances were dispatched independently, so the user got N separate
 *   replies instead of one album reply. Cloudflare KV provides durable,
 *   cross-instance state — every Worker invocation, regardless of which colo
 *   it lands on, can read and append to the same album buffer.
 *
 * Why per-photo keys instead of read-modify-write on a single key?
 *   KV's free tier has a soft limit of 1 write per second per key, and KV is
 *   eventually consistent. A read-modify-write loop on a single album key
 *   would race when multiple album photos arrive within ~100ms of each other
 *   (the typical Telegram album cadence). Per-photo keys eliminate the
 *   read-modify-write race: each photo writes to its OWN key, and the flush
 *   path uses `list({ prefix })` to gather all siblings atomically.
 *
 * Last-photo-flushes pattern:
 *   Each album POST schedules `ctx.waitUntil(flush at t+2500ms)`. The flush
 *   reads all photos for this album and elects a leader (smallest message_id).
 *   Only the leader dispatches. The sentinel key (`album:{id}:dispatched`)
 *   suppresses double-dispatch if a sibling's flush fires after the leader.
 *
 * Why answerCallbackQuery MUST be the first HTTP call?
 *   Telegram's client shows a small loading spinner on the tapped inline
 *   keyboard button. The spinner auto-stops when answerCallbackQuery is
 *   received OR after ~10 seconds (whichever comes first). If the Worker
 *   does any slow work first (KV read, GitHub dispatch), the user sees the
 *   spinner for the full 10s, which feels broken. Calling answerCallbackQuery
 *   as line 1 of the callback handler kills the spinner in <100ms.
 *
 * Security model:
 *   - All secrets live in Cloudflare Worker secrets; NONE are in the repo.
 *   - Unauthorized users get an identical 200 OK — no information leakage.
 *   - The repository is public so we get unlimited GitHub Actions minutes, but
 *     the dispatch event requires GH_PAT_TOKEN so a stranger cannot fire it
 *     directly through GitHub's API.
 *
 * Model:
 *   Hardcoded to depth-anything/Depth-Anything-V2-Large-hf. The Small model
 *   has been retired from the entire pipeline. The user no longer types
 *   /hd or /fast — every run is HD. Color (inferno|gray) is chosen via the
 *   inline keyboard buttons.
 */

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_LARGE = 'depth-anything/Depth-Anything-V2-Large-hf';

// How long to wait after the LATEST photo of a media group before flushing
// the accumulated batch. Telegram sends album photos in quick succession
// (typically <500ms apart), but we leave a healthy margin so slow networks
// still get fully batched. 2500ms is short enough to feel responsive and long
// enough to catch a 10-photo album.
//
// Note: this is the INITIAL debounce. The flush function also does a 4-try
// re-check loop with 500ms pauses, because Cloudflare KV is eventually
// consistent across colos — a sibling write may not be visible to a flush
// running in a different colo for up to ~1-2 seconds.
const MEDIA_GROUP_BUFFER_MS = 2500;

// How many times the flush should re-check KV for additional photos before
// giving up and flushing whatever it has. Each retry is 500ms apart.
// Total worst-case wait: MEDIA_GROUP_BUFFER_MS + FLUSH_RETRIES * 500ms.
const FLUSH_RETRIES = 4;
const FLUSH_RETRY_DELAY_MS = 500;

// Hard ceiling on the number of photos in a single session. Telegram's
// sendMediaGroup API limits albums to 10 items; if a user somehow uploads
// more (Telegram shouldn't allow it, but defensive), we cap here.
const MAX_PHOTOS_PER_BATCH = 10;

// KV key TTLs.
//
// KV_ALBUM_BUFFER_TTL_S: per-photo keys during album buffering. Short — the
//   flush deletes them after dispatch, but if the leader crashes they auto-
//   expire so orphans don't accumulate.
//
// SESSION_TTL_S: the session entry that the callback_query reads to fire the
//   GitHub dispatch. Must be long enough that the user has time to tap a
//   button. 10 minutes is generous — if they haven't tapped in 10 min, they
//   probably never will, and the session expires cleanly.
//
// SENTINEL_TTL_S: marks an album as already-flushed so sibling flush timers
//   no-op. 60s is enough for all sibling timers to have fired.
const KV_ALBUM_BUFFER_TTL_S = 300;
const SESSION_TTL_S = 600;
const SENTINEL_TTL_S = 60;

// ---------------------------------------------------------------------------
// Telegram + GitHub glue
// ---------------------------------------------------------------------------

const TG_API = (botToken) => `https://api.telegram.org/bot${botToken}`;

/**
 * Send a short text message to a Telegram chat. Fire-and-forget, errors are
 * swallowed so we never block the 200 OK response to Telegram.
 */
async function notifyTelegram(botToken, chatId, text) {
  try {
    await fetch(`${TG_API(botToken)}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text,
        parse_mode: 'Markdown',
        disable_web_page_preview: true,
      }),
    });
  } catch (e) {
    console.error('notifyTelegram failed:', e.message);
  }
}

/**
 * Send a sendMessage with an inline keyboard offering two color choices.
 * Returns the sent message's message_id (so the caller can later edit it),
 * or null if the send failed.
 *
 * The inline keyboard has TWO buttons in ONE row:
 *   [🎨 HD Color] [🏁 HD Grayscale]
 * Each button's callback_data encodes the chosen cmap and the first photo's
 * message_id (used to look up the session in KV when the button is tapped).
 *
 * `photoMessageId` is the message_id of the FIRST photo in the album (or the
 * only photo for single-photo uploads). It becomes the suffix of:
 *   - the KV session key:   session:{chatId}:{photoMessageId}
 *   - the callback_data:    process:{cmap}:{photoMessageId}
 */
async function sendInlineKeyboard(botToken, chatId, photoMessageId, isAlbum, photoCount) {
  const label = isAlbum
    ? `📸 Got your album of ${photoCount} photo${photoCount === 1 ? '' : 's'}.`
    : '📸 Got your photo.';

  const text =
    `${label}\n\n` +
    `Tap a button to generate an *HD depth map* with the chosen render:\n\n` +
    `• *HD Color* — inferno colormap (warm = near, cool = far)\n` +
    `• *HD Grayscale* — pure 0–255 grayscale\n\n` +
    `Model: \`Depth-Anything-V2-Large-hf\``;

  const inlineKeyboard = [
    [
      { text: '🎨 HD Color', callback_data: `process:inferno:${photoMessageId}` },
      { text: '🏁 HD Grayscale', callback_data: `process:gray:${photoMessageId}` },
    ],
  ];

  try {
    const res = await fetch(`${TG_API(botToken)}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text,
        parse_mode: 'Markdown',
        disable_web_page_preview: true,
        reply_markup: { inline_keyboard: inlineKeyboard },
      }),
    });
    const body = await res.json();
    if (!body.ok) {
      console.error('sendInlineKeyboard failed:', body);
      return null;
    }
    return body.result?.message_id ?? null;
  } catch (e) {
    console.error('sendInlineKeyboard threw:', e.message);
    return null;
  }
}

/**
 * IMMEDIATELY answer a callback_query. This is the single most important call
 * in the callback handler — it dismisses the Telegram button loading spinner
 * within ~100ms. Without it, the spinner runs for the full 10s timeout and
 * the bot feels broken.
 *
 * MUST be the FIRST HTTP call in the callback handler. Do not do KV reads,
 * GitHub dispatches, or any other slow work before this.
 *
 * `text` is shown as a small toast at the top of the chat for ~2s.
 */
async function answerCallbackQuery(botToken, callbackQueryId, text) {
  try {
    await fetch(`${TG_API(botToken)}/answerCallbackQuery`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        callback_query_id: callbackQueryId,
        text,
      }),
    });
  } catch (e) {
    // Don't throw — even if this fails, we want to proceed with the dispatch.
    console.error('answerCallbackQuery failed:', e.message);
  }
}

/**
 * Edit the text (and optionally reply_markup) of an existing inline keyboard
 * message. Used to swap the "tap a button" prompt for "⏳ Generating..." after
 * the user taps.
 */
async function editMessageText(botToken, chatId, messageId, text, opts = {}) {
  const body = {
    chat_id: chatId,
    message_id: messageId,
    text,
    parse_mode: opts.parse_mode || 'Markdown',
    disable_web_page_preview: true,
  };
  // If a new inline_keyboard is provided, include it. Pass `null` to remove
  // the keyboard entirely ( Telegram accepts reply_markup: undefined to leave
  // it unchanged, or {inline_keyboard: []} to remove it).
  if (opts.inline_keyboard !== undefined) {
    body.reply_markup = { inline_keyboard: opts.inline_keyboard };
  }
  try {
    await fetch(`${TG_API(botToken)}/editMessageText`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    console.error('editMessageText failed:', e.message);
  }
}

/**
 * Trigger the GitHub Actions workflow via the `repository_dispatch` event.
 * The workflow in `.github/workflows/depth_pipeline.yml` listens for the
 * event type `depth_request`.
 */
async function triggerGitHubAction(env, payload) {
  const owner = env.GITHUB_REPO_OWNER;
  const repo = env.GITHUB_REPO_NAME;
  const token = env.GH_PAT_TOKEN;

  const url = `https://api.github.com/repos/${owner}/${repo}/dispatches`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      'User-Agent': 'telegram-depth-anything-bot',
    },
    body: JSON.stringify({
      event_type: 'depth_request',
      client_payload: payload,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`GitHub dispatch failed (${res.status}): ${body}`);
  }
}

// ---------------------------------------------------------------------------
// KV album buffer (per-photo keys, leader-elected flush)
// ---------------------------------------------------------------------------
//
// Storage layout (KV namespace ALBUM_BUFFER):
//   album:{media_group_id}:{message_id}  -> value="", metadata={photo data}
//   album:{media_group_id}:dispatched    -> value="1", short TTL
//   session:{chat_id}:{first_message_id} -> value=JSON.stringify({photo_ids, ...}),
//                                            10-minute TTL (used by callback_query)
//
// Each photo writes its OWN key, so there are no read-modify-write races and
// no per-key write-rate-limit issues. The flush path uses `list({ prefix })`
// which returns metadata inline — no per-key `get()` calls needed.
//
// KV metadata is capped at 1024 bytes per entry. Our photo metadata is
// typically ~500 bytes (file_id ~80 chars, plus a few small fields and a
// truncated caption), well within the limit.

/**
 * Build the KV key for a single photo in an album (or a single-photo "album").
 */
function photoKey(mediaGroupId, messageId) {
  return `album:${mediaGroupId}:${messageId}`;
}

/**
 * Build the KV sentinel key that marks an album as already-flushed.
 */
function flushedSentinelKey(mediaGroupId) {
  return `album:${mediaGroupId}:flushed`;
}

/**
 * Build the KV session key that the callback_query reads to fire the dispatch.
 */
function sessionKey(chatId, firstMessageId) {
  return `session:${chatId}:${firstMessageId}`;
}

/**
 * Write a single photo's metadata to KV under its own key.
 *
 * The photo data goes into KV `metadata` (not the value) so that `list()`
 * returns it inline — the flush path doesn't need to call `get()` per photo,
 * saving N KV reads per album.
 *
 * KV metadata is capped at 1024 bytes. We truncate the caption defensively.
 */
async function writePhotoEntry(env, mediaGroupId, photo) {
  const safeCaption = (photo.caption || '').slice(0, 600);

  const metadata = {
    chatId: photo.chatId,
    messageId: photo.messageId,
    fileId: photo.fileId,
    caption: safeCaption,
    receivedAt: Date.now(),
  };

  const key = photoKey(mediaGroupId, photo.messageId);
  await env.ALBUM_BUFFER.put(key, '', {
    expirationTtl: KV_ALBUM_BUFFER_TTL_S,
    metadata,
  });
}

/**
 * Gather all photos for an album by listing KV keys with the album prefix.
 *
 * Returns an array of { name, metadata } entries, sorted by messageId for
 * stable ordering (Telegram delivers album photos in message_id order, but
 * KV list() does not guarantee any particular order).
 *
 * If the flushed sentinel exists, callers should treat the album as
 * already-handled (checked separately via isAlreadyFlushed).
 */
async function gatherAlbum(env, mediaGroupId) {
  const prefix = `album:${mediaGroupId}:`;
  const sentinel = flushedSentinelKey(mediaGroupId);
  const entries = [];
  let cursor;

  do {
    const result = await env.ALBUM_BUFFER.list({ prefix, cursor, limit: 100 });
    for (const k of result.keys) {
      if (k.name === sentinel) continue;
      if (k.metadata) {
        entries.push({ name: k.name, metadata: k.metadata });
      }
    }
    cursor = result.list_complete ? null : result.cursor;
  } while (cursor);

  entries.sort((a, b) => a.metadata.messageId - b.metadata.messageId);
  return entries;
}

/**
 * Delete all photo keys for an album (called after a successful flush).
 *
 * Best-effort: if a delete fails, the key will auto-expire after KV_ALBUM_BUFFER_TTL_S.
 */
async function deleteAlbumKeys(env, mediaGroupId) {
  const prefix = `album:${mediaGroupId}:`;
  const sentinel = flushedSentinelKey(mediaGroupId);
  let cursor;

  do {
    const result = await env.ALBUM_BUFFER.list({ prefix, cursor, limit: 100 });
    for (const k of result.keys) {
      if (k.name === sentinel) continue;
      await env.ALBUM_BUFFER.delete(k.name);
    }
    cursor = result.list_complete ? null : result.cursor;
  } while (cursor);
}

/**
 * Set the flushed sentinel so sibling flush timers know to no-op.
 */
async function markFlushed(env, mediaGroupId) {
  await env.ALBUM_BUFFER.put(flushedSentinelKey(mediaGroupId), '1', {
    expirationTtl: SENTINEL_TTL_S,
  });
}

/**
 * Check if the flushed sentinel exists.
 */
async function isAlreadyFlushed(env, mediaGroupId) {
  const val = await env.ALBUM_BUFFER.get(flushedSentinelKey(mediaGroupId));
  return val !== null;
}

// ---------------------------------------------------------------------------
// KV session storage (for callback_query retrieval)
// ---------------------------------------------------------------------------

/**
 * Write the consolidated session (photo_ids + metadata) to KV so the
 * callback_query handler can fire the GitHub dispatch when the user taps a
 * color button.
 *
 * Key:   session:{chatId}:{firstMessageId}
 * TTL:   SESSION_TTL_S (10 minutes)
 * Value: JSON.stringify({ chat_id, message_id, photo_ids, is_album,
 *                          photo_count, caption })
 */
async function writeSession(env, chatId, firstMessageId, session) {
  await env.ALBUM_BUFFER.put(sessionKey(chatId, firstMessageId), JSON.stringify(session), {
    expirationTtl: SESSION_TTL_S,
  });
}

/**
 * Read and parse a session entry. Returns null if missing or invalid JSON.
 */
async function readSession(env, chatId, firstMessageId) {
  const raw = await env.ALBUM_BUFFER.get(sessionKey(chatId, firstMessageId));
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/**
 * Delete a session entry (after the dispatch has fired, to prevent the user
 * from tapping the button again and double-dispatching).
 */
async function deleteSession(env, chatId, firstMessageId) {
  await env.ALBUM_BUFFER.delete(sessionKey(chatId, firstMessageId));
}

// ---------------------------------------------------------------------------
// Flush: gather album, write session, show inline keyboard
// ---------------------------------------------------------------------------

/**
 * Flush a buffered album: gather all photo entries from KV, write a session
 * entry, send the inline keyboard, and clean up the per-photo KV keys.
 *
 * Called either:
 *   - immediately (in `ctx.waitUntil`) for single-photo messages, OR
 *   - after MEDIA_GROUP_BUFFER_MS (in `ctx.waitUntil`) for album messages.
 *
 * Leader election by smallest message_id:
 *   Each album POST schedules its own flush. To avoid N sibling flushes each
 *   showing its own inline keyboard, only the photo with the SMALLEST
 *   message_id in the album flushes. All other siblings no-op.
 *
 *   Telegram delivers album photos in message_id order, so the smallest
 *   message_id is the FIRST photo of the album. Its flush is the "leader".
 *
 * KV eventual consistency handling:
 *   Cloudflare KV is eventually consistent across colos. When 4 album photos
 *   arrive within ~100ms, they may land on different colos, and each colo's
 *   KV read may not see writes from other colos for up to ~1-2 seconds.
 *
 *   The leader's flush does a retry loop: gather, wait, gather again. If the
 *   count is still growing, keep retrying (up to FLUSH_RETRIES). When the
 *   count is stable, flush whatever we have.
 *
 *   The sentinel key prevents double-flush if the leader crashes mid-flush
 *   and a sibling's retry loop later sees a stable count.
 *
 * `myMessageId` is the message_id of the photo that triggered THIS flush.
 * For single photos, it's the only photo's message_id (always the leader).
 * For albums, the leader check is: am I the smallest message_id in the album?
 */
async function flushAlbum(env, mediaGroupId, isAlbum, myMessageId) {
  // Check sentinel first — a sibling flush may have already flushed.
  if (await isAlreadyFlushed(env, mediaGroupId)) {
    console.log(`[flush] ${mediaGroupId} already flushed, skipping`);
    return;
  }

  let entries = await gatherAlbum(env, mediaGroupId);
  if (entries.length === 0) {
    console.log(`[flush] ${mediaGroupId} has no photos (already cleaned up)`);
    return;
  }

  // Leader election: only the photo with the smallest message_id flushes.
  if (isAlbum) {
    const minMessageId = Math.min(...entries.map((e) => e.metadata.messageId));
    if (myMessageId !== minMessageId) {
      console.log(
        `[flush] ${mediaGroupId} I am msg ${myMessageId}, leader is msg ${minMessageId}, deferring`
      );
      return;
    }
    console.log(`[flush] ${mediaGroupId} I am the leader (msg ${myMessageId})`);
  }

  // Retry loop: re-gather a few times to let KV propagate sibling writes.
  let lastCount = entries.length;
  for (let attempt = 1; attempt <= FLUSH_RETRIES; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, FLUSH_RETRY_DELAY_MS));

    if (await isAlreadyFlushed(env, mediaGroupId)) {
      console.log(`[flush] ${mediaGroupId} flushed by sibling during retry ${attempt}`);
      return;
    }

    const reEntries = await gatherAlbum(env, mediaGroupId);
    if (reEntries.length === 0) {
      console.log(`[flush] ${mediaGroupId} cleaned up during retry ${attempt}`);
      return;
    }

    if (reEntries.length > lastCount) {
      console.log(
        `[flush] ${mediaGroupId} retry ${attempt}: ` +
        `count grew ${lastCount} -> ${reEntries.length}, continuing`
      );
      entries = reEntries;
      lastCount = reEntries.length;
      continue;
    }

    if (reEntries.length !== entries.length) {
      entries = reEntries;
    }
    console.log(
      `[flush] ${mediaGroupId} retry ${attempt}: count stable at ${reEntries.length}, flushing`
    );
    break;
  }

  // Cap at MAX_PHOTOS_PER_BATCH defensively.
  const capped = entries.slice(0, MAX_PHOTOS_PER_BATCH);
  const photoIds = capped.map((e) => e.metadata.fileId);

  // First non-empty caption wins. Telegram normally only puts the caption on
  // the first photo of an album, but defensively we scan all of them.
  const firstCaption =
    capped.map((e) => e.metadata.caption).find((c) => c && c.trim().length > 0) || '';

  const first = capped[0].metadata;
  const chatId = first.chatId;
  const firstMessageId = first.messageId;
  const count = photoIds.length;
  const label = isAlbum ? `album (${count} photos)` : 'photo';

  console.log(
    `[flush] ${label}: photos=${count} media_group_id=${mediaGroupId} ` +
    `-> showing inline keyboard`
  );

  // Set the flushed sentinel BEFORE sending the keyboard. This minimizes the
  // race window where a sibling flush could also send a keyboard.
  await markFlushed(env, mediaGroupId);

  // Write the session entry so the callback_query handler can fire the
  // GitHub dispatch when the user taps a color button.
  const session = {
    chat_id: chatId,
    message_id: firstMessageId,
    photo_ids: photoIds,
    is_album: isAlbum,
    photo_count: count,
    caption: firstCaption,
    created_at: Date.now(),
  };
  await writeSession(env, chatId, firstMessageId, session);

  // Send the inline keyboard. The bot token is read here (not earlier) so
  // that any token-loading cost happens after KV writes, keeping the
  // critical path tight.
  const buttonMessageId = await sendInlineKeyboard(
    env.TELEGRAM_BOT_TOKEN,
    chatId,
    firstMessageId,
    isAlbum,
    count
  );

  if (buttonMessageId === null) {
    // sendMessage failed — notify the user via plain text and clean up.
    console.error(`[flush] sendInlineKeyboard failed for ${label}`);
    await notifyTelegram(
      env.TELEGRAM_BOT_TOKEN,
      chatId,
      `⚠️ Couldn't show the color picker for your ${label}. Try sending it again.`
    );
    await deleteSession(env, chatId, firstMessageId);
    await deleteAlbumKeys(env, mediaGroupId);
    return;
  }

  console.log(
    `[flush] inline keyboard sent: button_msg=${buttonMessageId} ` +
    `session=session:${chatId}:${firstMessageId}`
  );

  // Clean up the per-photo keys. Sentinel auto-expires.
  await deleteAlbumKeys(env, mediaGroupId);
}

// ---------------------------------------------------------------------------
// Callback query handler — fires when the user taps an inline keyboard button
// ---------------------------------------------------------------------------

/**
 * Handle a callback_query from a tapped inline keyboard button.
 *
 * CRITICAL: answerCallbackQuery MUST be the first HTTP call. Telegram shows a
 * loading spinner on the tapped button that auto-dismisses on
 * answerCallbackQuery or after 10s. Doing any slow work first makes the bot
 * feel broken.
 *
 * Steps (per spec):
 *   1. IMMEDIATE: answerCallbackQuery with text 'Processing depth map...'
 *   2. INLINE:    editMessageText on the keyboard message →
 *                 '⏳ Generating HD Depth Map...'
 *   3. ctx.waitUntil: read session from KV, fire repository_dispatch with
 *                 model=Depth-Anything-V2-Large-hf and cmap (inferno|gray).
 *
 * callback_data format: `process:{cmap}:{photo_message_id}`
 *   - cmap: 'inferno' or 'gray'
 *   - photo_message_id: the message_id of the FIRST photo (used as the KV
 *     session key suffix)
 */
async function handleCallbackQuery(env, ctx, callbackQuery) {
  const callbackQueryId = callbackQuery.id;
  const fromId = callbackQuery.from?.id;
  const chatId = callbackQuery.message?.chat?.id;
  const buttonMessageId = callbackQuery.message?.message_id;
  const data = callbackQuery.data || '';
  const botToken = env.TELEGRAM_BOT_TOKEN;
  const allowedUserId = parseInt(env.ALLOWED_TELEGRAM_USER_ID, 10);

  // Auth firewall — same silent-drop behavior as the message path.
  if (!fromId || fromId !== allowedUserId) {
    // Still answer the callback so the spinner stops, but do nothing else.
    if (callbackQueryId) {
      await answerCallbackQuery(botToken, callbackQueryId, 'Unauthorized');
    }
    return;
  }

  // Parse callback_data: 'process:{cmap}:{photo_message_id}'
  const parts = data.split(':');
  if (parts.length !== 3 || parts[0] !== 'process') {
    console.warn(`[callback] unparseable callback_data: ${data}`);
    await answerCallbackQuery(botToken, callbackQueryId, 'Bad request');
    return;
  }
  const cmap = parts[1];
  const photoMessageId = parseInt(parts[2], 10);
  if ((cmap !== 'inferno' && cmap !== 'gray') || Number.isNaN(photoMessageId)) {
    console.warn(`[callback] invalid cmap or photoMessageId in: ${data}`);
    await answerCallbackQuery(botToken, callbackQueryId, 'Bad request');
    return;
  }

  // ─── STEP 1 (IMMEDIATE) ────────────────────────────────────────────────
  // answerCallbackQuery FIRST. This kills the Telegram button spinner in
  // <100ms. Do NOT do any KV reads or GitHub dispatches before this line.
  await answerCallbackQuery(botToken, callbackQueryId, 'Processing depth map...');

  // ─── STEP 2 (INLINE) ───────────────────────────────────────────────────
  // Edit the inline keyboard message to show "⏳ Generating..." and remove
  // the buttons (so the user can't tap again and double-dispatch).
  const cmapLabel = cmap === 'inferno' ? 'inferno colormap' : 'grayscale';
  await editMessageText(
    botToken,
    chatId,
    buttonMessageId,
    `⏳ Generating HD Depth Map...\n\n` +
    `Render: *${cmapLabel}*\n` +
    `Model: \`Depth-Anything-V2-Large-hf\`\n` +
    `ETA: 1–3 min (warm cache faster)`,
    { inline_keyboard: [] }  // remove the buttons
  );

  // ─── STEP 3 (ctx.waitUntil) ────────────────────────────────────────────
  // Fire the GitHub dispatch in the background. We move this to
  // ctx.waitUntil so the Worker can return 200 to Telegram immediately,
  // keeping the webhook healthy.
  ctx.waitUntil(
    (async () => {
      try {
        const session = await readSession(env, chatId, photoMessageId);
        if (!session) {
          console.warn(
            `[callback] session not found for session:${chatId}:${photoMessageId} ` +
            `(likely expired after ${SESSION_TTL_S}s)`
          );
          await notifyTelegram(
            botToken,
            chatId,
            '⚠️ This button expired. Please send the photo(s) again to get a fresh depth map.'
          );
          return;
        }

        console.log(
          `[callback] dispatching: cmap=${cmap} photos=${session.photo_count} ` +
          `is_album=${session.is_album}`
        );

        const dispatchPayload = {
          chat_id: session.chat_id,
          message_id: session.message_id,
          photo_ids: session.photo_ids,
          model: MODEL_LARGE,
          cmap,
          is_album: session.is_album,
          photo_count: session.photo_count,
          caption: session.caption || '',
        };

        await triggerGitHubAction(env, dispatchPayload);

        // Delete the session so a second tap on a stale button (e.g. if the
        // user re-tapped before editMessageText removed the buttons) doesn't
        // fire a second dispatch.
        await deleteSession(env, chatId, photoMessageId);
      } catch (err) {
        console.error('[callback] dispatch error:', err);
        await notifyTelegram(
          botToken,
          chatId,
          `⚠️ Could not start the depth job. The dispatcher failed: ${err.message?.slice(0, 200)}`
        );
      }
    })()
  );
}

// ---------------------------------------------------------------------------
// Main handler
// ---------------------------------------------------------------------------

export default {
  async fetch(request, env, ctx) {
    // Telegram only ever POSTs.
    if (request.method !== 'POST') {
      return new Response(null, { status: 200 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response(null, { status: 200 });
    }

    // ─── callback_query branch (inline keyboard button tap) ──────────────
    // Handle this BEFORE the message branch — callback_query events have a
    // 10s button-spinner timeout that the message path doesn't, so it's the
    // more time-critical path.
    if (body.callback_query) {
      // STEP 1 (answerCallbackQuery) happens inside handleCallbackQuery as
      // the FIRST HTTP call. We don't do any other work here that could
      // delay it.
      try {
        await handleCallbackQuery(env, ctx, body.callback_query);
      } catch (err) {
        console.error('[callback] handler threw:', err);
      }
      return new Response(
        JSON.stringify({ ok: true, handled: 'callback_query' }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // ─── message branch (photo upload or text command) ───────────────────
    const message = body.message || body.edited_message;
    if (!message) {
      return new Response(null, { status: 200 });
    }

    const fromId = message.from?.id;
    const chatId = message.chat?.id;
    const allowedUserId = parseInt(env.ALLOWED_TELEGRAM_USER_ID, 10);

    // Auth firewall — unknown users get an identical 200 OK so the bot looks
    // indistinguishable from a dead endpoint to a scanner.
    if (!fromId || fromId !== allowedUserId) {
      return new Response(null, { status: 200 });
    }

    // Reject non-photo messages with a friendly hint.
    const photo = Array.isArray(message.photo) && message.photo.length > 0
      ? message.photo[message.photo.length - 1]  // highest resolution
      : null;

    if (!photo) {
      const rawText = (message.text || message.caption || '').trim();
      const botToken = env.TELEGRAM_BOT_TOKEN;
      if (rawText.startsWith('/start')) {
        await notifyTelegram(
          botToken,
          chatId,
          '*Depth bot reporting in.*\n\n' +
          'Send me a photo (or an album!) and I will reply with two buttons:\n\n' +
          '• 🎨 *HD Color* — inferno colormap\n' +
          '• 🏁 *HD Grayscale* — pure 0–255 grayscale\n\n' +
          'Tap one and I\'ll generate a high-quality depth map with the ' +
          '`Depth-Anything-V2-Large` model. ETA ~1–3 min depending on cache warmth.\n\n' +
          '_The /fast small-model option has been retired — every run is HD._'
        );
      } else if (rawText.startsWith('/help')) {
        await notifyTelegram(
          botToken,
          chatId,
          '*Usage:*\n' +
          '1. Attach a photo (or an album of up to 10) to your message.\n' +
          '2. I\'ll reply with two buttons: 🎨 HD Color and 🏁 HD Grayscale.\n' +
          '3. Tap the one you want. I\'ll start the depth run and reply with the result.\n\n' +
          '*Model:* `Depth-Anything-V2-Large-hf` (hardcoded — Small retired).\n\n' +
          '*Albums:* all photos in a single album are buffered for 2.5s and ' +
          'processed in one GitHub Actions run, and you get a single reply album back.\n\n' +
          'Everything runs on GitHub Actions runners; nothing is stored.'
        );
      } else {
        await notifyTelegram(
          botToken,
          chatId,
          'Send me a photo or album — I\'ll show you HD Color and HD Grayscale buttons to pick from.'
        );
      }
      return new Response(JSON.stringify({ ok: true, status: 'no_photo' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    // ─── Buffer the photo (album-aware) ───────────────────────────────────
    const mediaGroupId = message.media_group_id || null;
    const isAlbum = !!mediaGroupId;
    const rawText = (message.text || message.caption || '').trim();

    if (isAlbum) {
      // Album path: write the photo to KV under its own key, then schedule
      // a debounced flush. The leader (smallest message_id) flushes.
      await writePhotoEntry(env, mediaGroupId, {
        chatId,
        messageId: message.message_id,
        fileId: photo.file_id,
        caption: rawText,
      });

      ctx.waitUntil(
        (async () => {
          await new Promise((resolve) => setTimeout(resolve, MEDIA_GROUP_BUFFER_MS));
          try {
            await flushAlbum(env, mediaGroupId, true, message.message_id);
          } catch (err) {
            console.error('[waitUntil] album flush threw:', err);
          }
        })()
      );

      return new Response(
        JSON.stringify({
          ok: true,
          buffered: true,
          is_album: true,
          photo_count_in_batch_so_far: 1,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      );
    }

    // Single-photo path: no buffer, no debounce. Reuse the flush path by
    // synthesizing a per-photo "album" key.
    const singleKey = `single:${chatId}:${message.message_id}`;
    await writePhotoEntry(env, singleKey, {
      chatId,
      messageId: message.message_id,
      fileId: photo.file_id,
      caption: rawText,
    });

    ctx.waitUntil(
      (async () => {
        try {
          await flushAlbum(env, singleKey, false, message.message_id);
        } catch (err) {
          console.error('[waitUntil] single flush threw:', err);
        }
      })()
    );

    return new Response(
      JSON.stringify({
        ok: true,
        buffered: true,
        is_album: false,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    );
  },
};
