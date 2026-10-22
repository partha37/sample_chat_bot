# Sample Chat Bot &mdash; LangGraph + Gemini + FastAPI + Node

A complete, runnable chat-bot sample split into three independent layers, each
with its **own debugger**:

| Layer | Folder | Stack | Debugger |
|---|---|---|---|
| **AI layer** | `ai_layer/` | Python, LangGraph, Gemini | Node-level graph tracer, breakpoints, state snapshots, checkpoint replay |
| **Backend** | `backend/` | Python, FastAPI | HTTP request tracer (method, path, status, duration), trace feed |
| **Frontend** | `frontend/` | Node, Express, vanilla JS | Live trace timeline, graph flow view, breakpoint controls, request log |

```
  browser (http://localhost:3000)
        |  fetch + SSE
        v
  frontend/  (Express: static UI + /api proxy + live reload)
        |  HTTP / SSE
        v
  backend/   (FastAPI: /api/chat, /api/chat/stream, /api/debug/*)
        |  in-process calls
        v
  ai_layer/  (LangGraph StateGraph -> Gemini)
```

---

## 1. Prerequisites

* **Python 3.11+** (developed and tested on 3.14)
* **Node.js 18+** (tested on 24)
* A **Gemini API key** &mdash; free at <https://aistudio.google.com/apikey>

## 2. Setup

```powershell
# 1. create the virtual environment + install Python dependencies
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# 2. create your .env and paste in the key
Copy-Item .env.example .env
notepad .env          # set GEMINI_API_KEY=...

# 3. install the frontend dependencies
cd frontend
npm install
cd ..
```

> **No key yet?** The app still runs. Without a key the AI layer falls back to a
> local heuristic router + canned reply, and every debugger keeps working so
> you can explore the graph before wiring up Gemini.

## 3. Run it

**Everything at once** (installs deps, starts both servers, writes logs to `.logs/`):

```powershell
.\scripts\dev.ps1
```

**Or manually, in two terminals:**

```powershell
# terminal 1 - backend
.\.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000

# terminal 2 - frontend
cd frontend
npm start
```

Then open **<http://localhost:3000>**.

## 4. How a turn works

```
START → ingest → router ─┬─(no tool needed)→ respond → finalize → END
                          └─(tool requested)→ act → respond → finalize → END
```

* `ingest` &mdash; normalises the message.
* `router` &mdash; **one** Gemini call with the tools bound. Gemini either
  answers directly or calls a tool.
* `act` &mdash; runs the requested tool.
* `respond` &mdash; reuses the router's answer when no tool ran, otherwise makes
  the second (and last) call with the tool output.
* `finalize` &mdash; packages the reply.

Routing and answering share a single call, so a normal chat turn costs **one**
Gemini request. That matters: the free tier only allows ~5 requests per minute
per model.

## 5. Verify the build

With the backend running:

```powershell
.\.venv\Scripts\python scripts\smoke_test.py
```

It checks health, chat, the SSE stream, every debug endpoint, the
breakpoint pause/resume flow and error handling.

To verify what the browser actually renders (same SSE parsing and the same
"no response" fallback the UI uses):

```powershell
.\.venv\Scripts\python scripts\simulate_browser.py "Hello there!" "What is 25 * 4?"
```

The offline unit checks (no network, no quota needed):

```powershell
.\.venv\Scripts\python scripts\test_quota_parsing.py
.\.venv\Scripts\python scripts\test_model_validation.py
```

To see how many Gemini requests a turn actually spends (important on the free
tier):

```powershell
.\.venv\Scripts\python scripts\count_calls.py "Hello there!"
```

---

## 6. The three debuggers

### 6.1 AI layer debugger (`ai_layer/debugger.py`)

Every node is instrumented, so a turn produces a full trace:

```
enter ingest  ->  exit ingest   (0.1ms)
enter router  ->  exit router   (22.1s)   "model requested tool(s): calculator"
enter act     ->  exit act      (2.2ms)   "Executed 1 tool(s): calculator=100"
enter respond ->  exit respond  (44.2s)   "Generated 27 characters"
enter finalize->  exit finalize (0.1ms)
```

* **State snapshots** &mdash; the graph state is rendered into a JSON-safe
  preview before and after every node (messages summarised, long text
  truncated, secrets redacted).
* **Breakpoints** &mdash; mark any node `break` (pause before it runs and hand
  the state back) or `skip` (jump over it), then resume from the UI.
* **Checkpoint replay** &mdash; `GET /api/debug/sessions/{id}/checkpoints`
  returns the LangGraph checkpoint history for a conversation.
* **Model + quota visibility** &mdash; the trace records which model served the
  turn, and `GET /api/health` lists any model that is currently in a
  rate-limit cooldown.

### 6.2 Backend debugger (`backend/debugger.py`, `backend/middleware.py`)

`RequestTracingMiddleware` records every request: method, path, query string,
status code, duration, client IP. Each response carries `X-Request-Id` and
`X-Response-Time-ms` headers. The **Requests** tab in the UI shows the live
log with total / error / average / max timings.

### 6.3 Frontend debugger (`frontend/public/js/app.js`)

The right-hand **Debugger** panel has four tabs:

| Tab | What it shows |
|---|---|
| **Trace** | Node-by-node timeline, streamed live while the answer is generating. Click a node to expand its state snapshot. |
| **Graph** | The LangGraph flow (ingest → router → act → respond → finalize) with each node lighting up as it runs. |
| **Breakpoints** | Toggle `off` / `break` / `skip` per node; a paused run shows a **Resume** button in the Trace tab. |
| **Requests** | The backend request log with timings. |

The frontend also has a small live-reload watcher (`chokidar`) that refreshes
the page when you edit anything in `frontend/public`.

---

## 7. Project layout

```
sample_chat_bot/
├── ai_layer/                 # AI layer (no web framework imports)
│   ├── config.py             #   settings loaded from the root .env
│   ├── state.py              #   ChatState (the LangGraph state contract)
│   ├── tools.py              #   calculator / current_datetime / text_stats
│   ├── client.py             #   Gemini wrapper: fallbacks, retries, rate-limit cooldowns
│   ├── nodes.py              #   ingest, router (routes + answers), act, respond, finalize
│   ├── graph.py              #   StateGraph wiring, streaming, breakpoints
│   └── debugger.py           #   tracer, breakpoint registry, snapshots
│
├── backend/                  # FastAPI backend
│   ├── main.py               #   app factory, middleware, routers
│   ├── config.py             #   settings
│   ├── middleware.py          #   request tracing + access log
│   ├── debugger.py           #   HTTP request ring buffer
│   ├── routes/
│   │   ├── chat.py           #   /api/chat, /api/chat/stream, /api/chat/resume
│   │   ├── debug.py          #   /api/debug/*
│   │   └── health.py         #   /api/health
│   └── services/ai_service.py
│
├── frontend/                 # Node / Express UI
│   ├── server.js             #   static server + /api proxy + live reload
│   └── public/
│       ├── index.html
│       ├── css/style.css
│       └── js/app.js
│
├── scripts/
│   ├── dev.ps1               # start both servers
│   ├── smoke_test.py         # end-to-end verification
│   ├── simulate_browser.py   # replay what the browser does against /api/chat/stream
│   ├── count_calls.py        # show how many Gemini requests a turn costs
│   ├── inspect_trace.py      # print a session's debugger trace
│   ├── probe_models.py       # which models does this key know about?
│   ├── probe_quota.py        # which models still have quota today?
│   ├── test_quota_parsing.py # unit checks for the quota error parser
│   └── test_model_validation.py # unit checks for retired-model detection
││
├── requirements.txt
└── .env.example
```

## 8. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness + AI configuration status |
| `POST` | `/api/chat` | One turn of the chat graph (returns the answer + trace) |
| `POST` | `/api/chat/stream` | Same, streamed as Server-Sent Events |
| `POST` | `/api/chat/resume` | Resume a run paused at a breakpoint |
| `GET` | `/api/debug/status` | Debugger overview |
| `GET` | `/api/debug/graph` | Graph nodes + edges (used to draw the flow) |
| `GET` | `/api/debug/traces` | Recent graph traces |
| `GET` | `/api/debug/traces/{id}` | Full trace with every event |
| `GET` | `/api/debug/sessions/{id}/checkpoints` | LangGraph checkpoint history |
| `GET/PUT/DELETE` | `/api/debug/breakpoints[/{node}]` | Breakpoint registry |
| `GET/DELETE` | `/api/debug/requests` | HTTP request log |

Interactive docs: <http://127.0.0.1:8000/docs>

## 9. Configuration

All settings live in the root `.env` (see `.env.example`).

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | &mdash; | Your Gemini key (required for real answers) |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Primary model id (see `GEMINI_FALLBACK_MODELS`) |
| `GEMINI_TEMPERATURE` | `0.7` | Sampling temperature |
| `GEMINI_MAX_OUTPUT_TOKENS` | `1024` | Max tokens per reply |
| `GEMINI_REQUEST_TIMEOUT` | `60` | Per-request timeout (seconds) |
| `GEMINI_MAX_RETRIES` | `1` | Retries per model for transient 429/503 errors |
| `GEMINI_RETRY_BASE_DELAY` | `1.0` | Base delay for exponential backoff (seconds) |
| `GEMINI_FALLBACK_MODELS` | _(none)_ | Comma-separated models tried when the primary is busy |
| `GEMINI_THINKING_BUDGET` | `0` | Reasoning-token budget (`0` = fast replies) |
| `BACKEND_PORT` | `8000` | FastAPI port |
| `FRONTEND_PORT` | `3000` | Express port |
| `BACKEND_URL` | `http://127.0.0.1:8000` | Backend URL used by the proxy |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed origins |
| `MAX_TRACE_HISTORY` | `50` | Traces/requests kept in memory |

## 10. Troubleshooting

**"backend offline" in the UI** &mdash; the frontend cannot reach FastAPI.
Start it with `.\.venv\Scripts\python -m uvicorn backend.main:app --port 8000`
and check `.logs\backend.err.log`.

**"no API key" in the UI** &mdash; `GEMINI_API_KEY` is empty. Edit `.env` and
restart the backend.

**Gemini returns 429/403** &mdash; the key is invalid, rate-limited, or the
model is not enabled for that project. Try `GEMINI_MODEL=gemini-2.5-flash`.

**Gemini returns 503 "high demand"** &mdash; the model is temporarily
saturated. The client retries transient errors with exponential backoff and
then walks `GEMINI_FALLBACK_MODELS` until one responds, so the bot keeps
working. Set the chain to the models your key can actually reach:

```powershell
# list what your key can use
.\.venv\Scripts\python -c "import os;from dotenv import load_dotenv;load_dotenv();from google import genai;c=genai.Client(api_key=os.environ['GEMINI_API_KEY']);print([m.name.split('/')[-1] for m in c.models.list()])"
```

**"Quota exceeded ... limit: 20"** &mdash; the Gemini **free tier** allows
only a handful of requests:

| Limit | Value |
|---|---|
| Requests per minute, per model | 5 |
| Requests per **day**, per model | 20 |

Each chat turn spends **one** request (see section 4), so a free-tier key can
answer roughly 20 conversations per model per day. The client reads Google's
`retryDelay`, puts the exhausted model on a cooldown for exactly that long,
and serves the next request from a model that still has quota. `GET
/api/health` shows the live cooldowns:

```json
"cooldowns": {"gemini-3.8-flash": 69358, "gemini-3.6-flash": 69358}
```

While every model is cooling down the bot answers with a clear message
instead of failing, and it fails **fast** (no hanging requests).

For real usage, enable billing on the Google AI Studio project - the free
tier is fine for a demo but not for repeated testing.

**"This model models/gemini-2.0-flash is no longer available"** &mdash; Google
retired that model id. The app detects the 404, skips the model from then on
and tells you the replacement (currently `gemini-3.8-flash`). Set
`GEMINI_MODEL` in `.env` accordingly.

**Check which models your key can actually use** (uses a little quota, so
run it sparingly):

```powershell
.\.venv\Scripts\python scripts\probe_models.py
.\.venv\Scripts\python scripts\probe_quota.py
```

The model that actually served a reply is reported as `active_model` on the
chat response and in the Trace tab.

**Replies are slow** &mdash; Google is serving the model slowly right now. The
Trace tab shows exactly where the time goes (e.g. `router 22s`, `respond 44s`),
and tokens still stream to the browser as they arrive.

**SSE tokens do not appear** &mdash; check the browser Network tab: the
`/api/chat/stream` response must have `Content-Type: text/event-stream`. If
you are behind a proxy, make sure it does not buffer responses.

**Port already in use** &mdash; change `BACKEND_PORT` / `FRONTEND_PORT` in
`.env` and restart.
