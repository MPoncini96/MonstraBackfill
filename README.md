# MonstraBackfill

HTTP service for **bot creation previews** (alpha1, alpha2, beta1, gamma1). Deploy to [Render](https://render.com) or any Python host; point the Next.js app at it with environment variables (see below).

## Endpoints

- `GET /health` — liveness
- `POST /preview/alpha1` | `alpha2` | `beta1` | `gamma1` — same JSON body as the Next.js `/api/bots/preview-*` routes

Optional auth: set `MONSTRA_PREVIEW_SECRET` and send `Authorization: Bearer <secret>`.

## Local run

```bash
cd MonstraBackfill
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
set MONSTRA_PREVIEW_SECRET=dev-secret
uvicorn app:app --reload --port 8787
```

## Render

1. New **Web Service** from this Git repo; root directory = `MonstraBackfill` (if the repo is the monorepo, set **Root Directory** to `MonstraBackfill`).
2. **Build:** `pip install -r requirements.txt`  
   **Start:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. Add env: `MONSTRA_PREVIEW_SECRET` (random string), optionally `MONSTRA_PREVIEW_CORS_ORIGINS` with your Vercel URL(s).

## Next.js (Vercel)

Set:

- `MONSTRA_PREVIEW_SERVICE_URL` = `https://<your-render-service>.onrender.com` (no trailing slash)
- `MONSTRA_PREVIEW_SECRET` = same value as on Render

When `MONSTRA_PREVIEW_SERVICE_URL` is set, preview API routes call this service instead of spawning Python locally.

## Updating strategy code

Preview logic is copied from `Monstra-Worker` (`bots/*.py`, `backfill_*.py`, `Preview_backfill_*.py`). After changing strategies in the main worker, copy the updated files into this folder and redeploy.
