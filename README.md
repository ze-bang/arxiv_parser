# arxiv_parser

Daily fetch of arXiv cond-mat.str-el submissions, impact ranking, LLM summaries, and email digests.

## Features
- Pulls daily submissions from arXiv category `cond-mat.str-el`
- Computes an impact score combining:
  - Author influence (OpenAlex h-index, recent publications)
  - Topic prevalence (recent highly cited works in inferred concepts)
  - Venue impact proxy (OpenAlex 2-year mean citedness for the journal, if available)
  - Exponential time decay (time constant τ = 1.5 years)
- Summarizes top-m papers with an LLM (OpenAI by default; falls back to simple summaries if no key)
- Email subscription: send HTML digest via SMTP
- SQLite persistence for dedup and subscriber management

## Quick start
1) Python env
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure environment (create `.env` or export env vars):
- Optional LLM:
  - `OPENAI_API_KEY` – your OpenAI key
  - `OPENAI_MODEL` – default `gpt-4o-mini`
- OpenAlex: no key required (public API), but respect rate limits.
- SMTP (for email):
  - `SMTP_HOST`, `SMTP_PORT` (e.g. 587), `SMTP_USER`, `SMTP_PASS`
  - `EMAIL_FROM` (display address)
  - Or use Resend: `RESEND_API_KEY`, `RESEND_FROM` (or reuse `EMAIL_FROM`)
- App config:
  - `TOP_M` (default 5)

Example `.env`:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@email.com
SMTP_PASS=app-password-or-pass
EMAIL_FROM=CondMat Digest <your@email.com>
TOP_M=5
```

3) Run once (dry run prints to console, no emails)
```
python -m src.main --dry-run --top 5
```

4) Subscribe an email and send
```
python -m src.main subscribe --email you@example.com
python -m src.main --top 5 --send-email
```

## Fetch by date
Pull any UTC date instead of today:
```
python -m src.main --dry-run --top 5 --date 2025-08-01
```

## Scheduling
Use cron to run daily (UTC morning). Example crontab:
```
0 9 * * * cd /home/pc_linux/arxiv_parser && . .venv/bin/activate && python -m src.main --top 5 --send-email >> logs/run.log 2>&1
```

## Notes
- OpenAlex is used for author h-index, recent works, venue metrics, and topic concept stats. Journal Impact Factor is proprietary; we use OpenAlex 2yr mean citedness as a proxy.
- If no LLM key is set, summaries fall back to a simple 2-3 sentence abstract-based summary.
- Time decay uses τ = 1.5 years: weight = exp(-age_days / (365*1.5)).

## Vercel front-end (subscriptions)
- A minimal Next.js app lives in `web/` for Vercel deployment with a subscription form.
- Backed by Vercel Postgres. Set DB env vars in Vercel.
- API routes:
  - `POST /api/subscribe` { email }
  - `POST /api/unsubscribe` { email }
  - `GET /api/subscribers` -> { subscribers: string[] }
- To have the Python app use Vercel’s list for emailing, set:
  - `SUBSCRIBERS_URL=https://<your-vercel-app>.vercel.app/api/subscribers`

## Development
- Code lives under `src/`
- DB is `data/app.db`
- Env vars via `.env` (optional)

## Troubleshooting
- Rate limits: add small sleeps for OpenAlex queries; this app already batches calls where possible.
- Email sending issues: check SMTP credentials, ports, and allow “less secure app” or app passwords as needed.
- If arXiv API returns fewer results, the category may have zero submissions that day.
