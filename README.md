# MonstraBackfill

HTTP service for Monstra preview requests and direct creator-bot backfills. Deploy to [Render](https://render.com) or any Python host; point the Next.js app at it with environment variables (see below).

## Endpoints

- `GET /health` - readiness and route/module diagnostics
- `POST /preview/alpha1` | `alpha2` | `gamma1` | `aptet` - same JSON body as the Next.js `/api/bots/preview-*` routes
- `POST /preview/echo1`
- `POST /preview/echo1-pair-correlation`
- `POST /backfill/alpha1` | `alpha2` | `gamma1` | `aptet` | `echo1`

Optional auth: set `MONSTRA_PREVIEW_SECRET` and send `Authorization: Bearer <secret>`.

## Health payload

`GET /health` returns a safe readiness payload with:

- service name
- version and git commit when available
- database configured true/false
- per-algorithm preview/backfill module importability
- registered preview/backfill routes
- backfill/preview readiness booleans
- queue support note pointing to the worker service

No secrets are exposed.

## Local run

```bash
cd MonstraBackfill
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set MONSTRA_PREVIEW_SECRET=dev-secret
uvicorn app:app --reload --port 8787
```

If you want direct `/backfill/*` routes to work locally, also set `DATABASE_URL`.

## Render

1. New Web Service from this Git repo; root directory = this repo folder.
2. Build: `pip install -r requirements.txt`
   Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Add env:
   - `MONSTRA_PREVIEW_SECRET`
   - `DATABASE_URL` for direct backfill support
   - optionally `MONSTRA_PREVIEW_CORS_ORIGINS`

## Next.js app env

Set:

- `MONSTRA_PREVIEW_SERVICE_URL` = preview-capable MonstraBackfill URL
- `MONSTRA_BACKFILL_SERVICE_URL` = direct-backfill-capable MonstraBackfill URL
- `MONSTRA_PREVIEW_SECRET` = same value as on Render

## Smoke test

Run:

```bash
python scripts/smoke_backfill_service.py --base-url https://monstrabackfill.onrender.com
```

This checks `/health` plus all five direct `/backfill/*` routes and classifies failures such as missing route, missing module, missing DB config, no active bot found, or generic server failure.

## Updating strategy code

Preview and direct backfill logic must stay in sync with Monstra-Worker. After changing strategies in the main worker, copy the updated files into this repo and redeploy.

