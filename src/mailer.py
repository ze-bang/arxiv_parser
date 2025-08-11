from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import json
import requests

from .config import config


def _get_env() -> Environment:
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=select_autoescape(["html"]))
    return env


def render_digest(date_str: str, items: list[dict]) -> str:
    env = _get_env()
    tpl = env.get_template("digest.html")
    return tpl.render(date=date_str, items=items)


def send_email(subject: str, html_body: str, recipients: List[str]) -> None:
    # Prefer Resend if configured
    if config.resend_api_key and (config.resend_from or config.email_from):
        from_addr = config.resend_from or config.email_from  # allow reuse of EMAIL_FROM
        if not from_addr:
            raise RuntimeError("RESEND_FROM or EMAIL_FROM must be set when using Resend.")
        r = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {config.resend_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "from": from_addr,
                "to": recipients,
                "subject": subject,
                "html": html_body,
            }),
            timeout=20,
        )
        if r.status_code >= 300:
            raise RuntimeError(f"Resend API error: {r.status_code} {r.text}")
        return

    # Fallback to SMTP
    if not (config.smtp_host and config.smtp_user and config.smtp_pass and config.email_from):
        raise RuntimeError("Email is not configured. Provide RESEND_API_KEY/RESEND_FROM or SMTP_* and EMAIL_FROM.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config.email_from
    msg["To"] = ", ".join(recipients)
    part = MIMEText(html_body, "html")
    msg.attach(part)

    with smtplib.SMTP(config.smtp_host, config.smtp_port) as server:
        server.starttls()
        server.login(config.smtp_user, config.smtp_pass)
        server.sendmail(config.email_from, recipients, msg.as_string())
