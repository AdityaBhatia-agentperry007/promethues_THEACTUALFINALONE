# Scalekit Gmail Gateway

Manual human-only setup:

1. Sign in at `app.scalekit.com`.
2. Create a Gmail connected-account app.
3. Put `SCALEKIT_ENV_URL`, `SCALEKIT_CLIENT_ID`, and `SCALEKIT_CLIENT_SECRET` in `.env`.
4. Keep `SCALEKIT_USER_IDENTIFIER=test_user`.
5. Run `python backend/scalekit_gateway/gmail_poller.py --once`.
6. If an authorization link is printed, Aditya opens it and authorizes `adityabhatia1505@gmail.com`.

Without credentials, the poller exits cleanly and the dashboard request box continues to drive the same `/agent` endpoint.

