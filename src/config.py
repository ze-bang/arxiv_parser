import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    category: str = os.getenv("ARXIV_CATEGORY", "cond-mat.str-el")
    top_m: int = int(os.getenv("TOP_M", "5"))

    # LLM
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Email
    smtp_host: str | None = os.getenv("SMTP_HOST")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str | None = os.getenv("SMTP_USER")
    smtp_pass: str | None = os.getenv("SMTP_PASS")
    email_from: str | None = os.getenv("EMAIL_FROM")

    # Resend (optional, alternative to SMTP)
    resend_api_key: str | None = os.getenv("RESEND_API_KEY")
    resend_from: str | None = os.getenv("RESEND_FROM")

    # Paths
    db_path: str = os.getenv("DB_PATH", "data/app.db")
    subscribers_url: str | None = os.getenv("SUBSCRIBERS_URL")

    # Misc
    user_agent: str = os.getenv("USER_AGENT", "arxiv_parser/0.1 (+https://example.com)")
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "20"))


config = Config()
