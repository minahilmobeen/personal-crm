#
# email_sync.py -- pulls mail from the Gmail test account via the Gmail API
# and writes it into data/emails.json in the same shape pipeline.py already
# expects (from, to, cc, bcc, timestamp, subject, body). additive: existing
# entries are never overwritten, only messages not already in the file are
# fetched and appended -- old email ids stay valid since memory.json bullets
# reference them.
#
import base64
import html
import json
import re
from datetime import datetime
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

BASE_DIR = Path(__file__).parent
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
TOKEN_PATH = BASE_DIR / "token.json"
EMAILS_PATH = BASE_DIR / "data" / "emails.json"

# skip chats/drafts/spam/trash -- not real correspondence for the CRM
EXCLUDE_QUERY = "-in:chats -in:drafts -in:spam -in:trash"


def get_service():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def list_message_ids(service):
    ids = []
    page_token = None
    while True:
        resp = service.users().messages().list(
            userId="me", q=EXCLUDE_QUERY, pageToken=page_token, maxResults=500
        ).execute()
        ids.extend(m["id"] for m in resp.get("messages", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return ids


def addresses(header_value):
    return [addr.lower() for _, addr in getaddresses([header_value or ""]) if addr]


HTML_TAG = re.compile(r"<[^>]+>")

def html_to_text(markup):
    text = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", markup)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = HTML_TAG.sub("", text)
    return html.unescape(text).strip()


def extract_body(payload):
    if payload.get("mimeType") == "text/plain" and "data" in payload.get("body", {}):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

    html_fallback = None
    for part in payload.get("parts", []) or []:
        if part.get("mimeType") == "text/plain" and "data" in part.get("body", {}):
            return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
        if part.get("mimeType") == "text/html" and "data" in part.get("body", {}) and html_fallback is None:
            html_fallback = base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="replace")
        if part.get("mimeType", "").startswith("multipart/"):
            nested = extract_body(part)
            if nested:
                return nested

    return html_to_text(html_fallback) if html_fallback is not None else ""


def parse_timestamp(headers, raw):
    date_header = headers.get("date")
    if date_header:
        try:
            dt = parsedate_to_datetime(date_header)
            if dt.tzinfo is not None:
                dt = dt.astimezone().replace(tzinfo=None)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except (TypeError, ValueError):
            pass
    # fallback for the rare message with a missing/malformed Date header
    dt = datetime.fromtimestamp(int(raw["internalDate"]) / 1000)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def normalize_message(raw):
    headers = {h["name"].lower(): h["value"] for h in raw["payload"]["headers"]}
    from_addrs = addresses(headers.get("from"))

    return {
        "from": from_addrs[0] if from_addrs else "",
        "to": addresses(headers.get("to")),
        "cc": addresses(headers.get("cc")),
        "bcc": addresses(headers.get("bcc")),
        "timestamp": parse_timestamp(headers, raw),
        "subject": headers.get("subject", ""),
        "body": extract_body(raw["payload"]),
    }


def sync():
    EMAILS_PATH.parent.mkdir(exist_ok=True)
    existing = json.loads(EMAILS_PATH.read_text()) if EMAILS_PATH.exists() else {}

    service = get_service()
    message_ids = list_message_ids(service)
    new_ids = [mid for mid in message_ids if mid not in existing]

    print(f"{len(message_ids)} messages in mailbox, {len(new_ids)} new")

    for mid in new_ids:
        raw = service.users().messages().get(userId="me", id=mid, format="full").execute()
        existing[mid] = normalize_message(raw)
        print(f"  fetched {mid}: {existing[mid]['subject'][:60]!r}")

    EMAILS_PATH.write_text(json.dumps(existing, indent=2))
    print(f"done. {len(existing)} total emails in {EMAILS_PATH}")


if __name__ == "__main__":
    sync()
