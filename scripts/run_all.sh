#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m backend.data.precompute
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
cd frontend
npm run dev -- -p 3000 &
FRONTEND_PID=$!
trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true' EXIT
wait

