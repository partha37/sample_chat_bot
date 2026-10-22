/**
 * Lightweight frontend server.
 *
 * - Serves the static UI from ./public
 * - Reverse-proxies /api/* to the FastAPI backend (streaming SSE included)
 * - Optional live-reload: watches ./public and pushes a reload event
 *
 * Dependencies are deliberately minimal: express, helmet, cors, dotenv,
 * chokidar (dev only).
 */

'use strict';

const path = require('node:path');
const fs = require('node:fs');
const { Readable } = require('node:stream');
const express = require('express');
const helmet = require('helmet');
const cors = require('cors');
const dotenv = require('dotenv');

const ROOT_DIR = path.resolve(__dirname, '..');
dotenv.config({ path: path.join(ROOT_DIR, '.env') });
dotenv.config({ path: path.join(__dirname, '.env'), override: true });

const PORT = Number(process.env.FRONTEND_PORT || 3000);
const BACKEND_URL = (process.env.BACKEND_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const WATCH_ENABLED = process.env.FRONTEND_WATCH !== 'false';

const app = express();
app.disable('x-powered-by');
app.set('trust proxy', true);

app.use(
  helmet({
    // The UI is plain HTML/JS with no inline bundler; CSP would only get in
    // the way of the inline bootstrap styles.
    contentSecurityPolicy: false,
    crossOriginEmbedderPolicy: false,
  })
);
app.use(cors({ origin: true }));
// NOTE: no express.json() here on purpose - the /api proxy forwards the raw
// request body upstream, and a body parser would consume the stream first,
// leaving the proxy with nothing to forward.

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
app.use((req, res, next) => {
  const started = Date.now();
  res.on('finish', () => {
    const ms = Date.now() - started;
    const flag = res.statusCode >= 400 ? '!' : '·';
    console.log(
      `${flag} ${req.method.padEnd(6)} ${req.originalUrl.padEnd(34)} ${String(res.statusCode).padEnd(3)} ${ms}ms`
    );
  });
  next();
});

// ---------------------------------------------------------------------------
// API proxy -> FastAPI backend
// ---------------------------------------------------------------------------
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'transfer-encoding',
  'upgrade',
  'content-encoding',
  'content-length',
  'host',
]);

function readRawBody(req) {
  return new Promise((resolve, reject) => {
    // The stream may already be finished (e.g. another middleware read it).
    if (req.readableEnded) {
      resolve(Buffer.alloc(0));
      return;
    }
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

app.use('/api', async (req, res) => {
  const target = `${BACKEND_URL}${req.originalUrl}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180_000);

  // Stop talking to the backend as soon as the browser goes away.
  req.on('aborted', () => controller.abort());

  try {
    const headers = { Accept: req.headers.accept || 'application/json' };
    if (req.headers['content-type']) headers['Content-Type'] = req.headers['content-type'];

    const options = {
      method: req.method,
      headers,
      signal: controller.signal,
    };

    if (!['GET', 'HEAD'].includes(req.method)) {
      const body = await readRawBody(req);
      if (body.length) options.body = body;
    }

    const upstream = await fetch(target, options);

    res.status(upstream.status);
    upstream.headers.forEach((value, key) => {
      if (!HOP_BY_HOP.has(key.toLowerCase())) res.setHeader(key, value);
    });

    if (!upstream.body) {
      res.end();
      return;
    }

    // Pipe straight through so SSE chunks arrive as they are produced.
    const stream = Readable.fromWeb(upstream.body);
    stream.on('error', (err) => {
      console.error(`[proxy] stream error: ${err.message}`);
      if (!res.headersSent) res.status(502);
      res.end();
    });
    stream.pipe(res);
  } catch (err) {
    clearTimeout(timer);
    if (res.headersSent) {
      res.end();
      return;
    }
    const aborted = err.name === 'AbortError';
    console.error(`[proxy] ${aborted ? 'aborted' : 'error'} ${target}: ${err.message}`);
    res.status(aborted ? 504 : 502).json({
      error: aborted ? 'Backend request timed out' : 'Could not reach the backend',
      detail: err.message,
      backend: BACKEND_URL,
      hint: 'Start the backend with: .\\.venv\\Scripts\\python -m backend.main',
    });
    return;
  } finally {
    clearTimeout(timer);
  }
});

// ---------------------------------------------------------------------------
// Static UI
// ---------------------------------------------------------------------------
const PUBLIC_DIR = path.join(__dirname, 'public');
app.use(express.static(PUBLIC_DIR, { extensions: ['html'], maxAge: 0 }));

// ---------------------------------------------------------------------------
// Live reload (development convenience)
// ---------------------------------------------------------------------------
const reloadClients = new Set();

app.get('/__reload', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.write('event: connected\ndata: {}\n\n');
  reloadClients.add(res);
  req.on('close', () => reloadClients.delete(res));
});

function broadcastReload() {
  for (const client of reloadClients) {
    client.write('event: reload\ndata: {}\n\n');
  }
}

if (WATCH_ENABLED) {
  try {
    const chokidar = require('chokidar');
    let timer = null;
    chokidar
      .watch(PUBLIC_DIR, { ignoreInitial: true })
      .on('all', () => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          console.log('· change detected - reloading clients');
          broadcastReload();
        }, 120);
      });
    console.log('· watching ./public for changes (live reload enabled)');
  } catch (err) {
    console.log(`· live reload disabled (${err.message})`);
  }
}

// ---------------------------------------------------------------------------
// Health + fallbacks
// ---------------------------------------------------------------------------
app.get('/__health', async (_req, res) => {
  let backend = { reachable: false };
  try {
    const response = await fetch(`${BACKEND_URL}/api/health`, {
      signal: AbortSignal.timeout(3000),
    });
    backend = { reachable: response.ok, status: response.status, body: await response.json() };
  } catch (err) {
    backend = { reachable: false, error: err.message };
  }
  res.json({
    status: 'ok',
    service: 'sample-chat-bot-frontend',
    port: PORT,
    backend_url: BACKEND_URL,
    backend,
  });
});

app.use((req, res) => {
  res.status(404).json({ error: 'Not found', path: req.originalUrl });
});

app.use((err, _req, res, _next) => {
  console.error('[error]', err);
  res.status(500).json({ error: 'Internal server error', detail: err.message });
});

app.listen(PORT, () => {
  console.log('');
  console.log('  Sample Chat Bot - Frontend');
  console.log(`  UI      -> http://localhost:${PORT}`);
  console.log(`  Backend -> ${BACKEND_URL}`);
  console.log('');
});
