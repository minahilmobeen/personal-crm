#
# local API server: reads data/memory.json and data/emails.json (both
# written by run.py) and serves them to the frontend. Mostly read-only,
# except merge resolution, which writes back via pipeline.py.
#
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import pipeline

DATA_DIR = Path(__file__).parent / "data"
MEMORY_PATH = DATA_DIR / "memory.json"
EMAILS_PATH = DATA_DIR / "emails.json"
CONFIG_PATH = DATA_DIR / "config.json"

app = FastAPI()

# local-only tool, but keeps the door open if the frontend is ever
# served from a different port during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


#
# split a person's bullets into Notes (transient, still active) and
# Previous Notes (durable, settled facts) -- both newest-updated first
#
def split_bullets(bullets):
    notes = [b for b in bullets if b.get("durability") == "transient"]
    previous_notes = [b for b in bullets if b.get("durability") == "durable"]
    notes.sort(key=lambda b: b["last_updated"], reverse=True)
    previous_notes.sort(key=lambda b: b["last_updated"], reverse=True)
    return notes, previous_notes


def person_result(person_id, record):
    notes, previous_notes = split_bullets(record.get("bullets", []))
    return {
        "found": True,
        "person_id": person_id,
        "display_name": record.get("display_name"),
        "emails": record.get("emails", []),
        "notes": notes,
        "previous_notes": previous_notes,
    }


#
# the professor's own address(es) -- excluded from being treated as a
# contact. editable from the frontend instead of hardcoded.
#
class ConfigUpdate(BaseModel):
    user_emails: list[str]


@app.get("/api/config")
def get_config():
    config = load_json(CONFIG_PATH)
    return {"user_emails": config.get("user_emails", [])}


@app.post("/api/config")
def update_config(update: ConfigUpdate):
    emails = [e.strip().lower() for e in update.user_emails if e.strip()]
    pipeline.save_config({"user_emails": emails})
    return {"user_emails": emails}


@app.get("/api/people")
def search_people(q: str):
    memory = load_json(MEMORY_PATH)
    q = q.strip().lower()
    if not q:
        return {"found": False}

    # exact email match, via the reverse index -- unambiguous, always wins
    person_id = memory.get("email_index", {}).get(q)
    if person_id:
        record = memory["people"].get(person_id)
        if record:
            return person_result(person_id, record)

    # fall back to a case-insensitive substring match on display_name.
    # different addresses for the same real person currently show up as
    # separate records here -- that's the known multi-address gap, not
    # a search bug, and gets resolved once merging is built.
    matches = [
        (pid, rec) for pid, rec in memory.get("people", {}).items()
        if q in (rec.get("display_name") or "").lower()
    ]

    if len(matches) == 1:
        pid, rec = matches[0]
        return person_result(pid, rec)

    if len(matches) > 1:
        return {
            "found": False,
            "multiple_matches": [
                {"person_id": pid, "display_name": rec.get("display_name"), "emails": rec.get("emails", [])}
                for pid, rec in matches
            ],
        }

    return {"found": False}


@app.get("/api/people/{person_id}")
def get_person(person_id: str):
    memory = load_json(MEMORY_PATH)
    record = memory.get("people", {}).get(person_id)
    if not record:
        raise HTTPException(status_code=404, detail="person not found")

    notes, previous_notes = split_bullets(record.get("bullets", []))
    return {
        "person_id": person_id,
        "display_name": record.get("display_name"),
        "emails": record.get("emails", []),
        "notes": notes,
        "previous_notes": previous_notes,
    }


@app.get("/api/emails/{email_id}")
def get_email(email_id: str):
    emails = load_json(EMAILS_PATH)
    email = emails.get(email_id)
    if not email:
        raise HTTPException(status_code=404, detail="email not found")
    return {"email_id": email_id, **email}


@app.get("/api/merges")
def list_merges():
    memory = load_json(MEMORY_PATH)
    result = []
    for merge_id, pending in memory.get("pending_merges", {}).items():
        if pending["status"] != "pending":
            continue
        candidate = memory.get("people", {}).get(pending["candidate_person_id"], {})
        result.append({
            "merge_id": merge_id,
            "new_person_id": pending["new_person_id"],
            "candidate_person_id": pending["candidate_person_id"],
            "candidate_display_name": candidate.get("display_name"),
            "evidence": pending["evidence"],
            "created": pending["created"],
        })
    return {"pending_merges": result}


class MergeResolution(BaseModel):
    confirmed: bool


@app.post("/api/merges/{merge_id}/resolve")
def resolve_merge(merge_id: str, resolution: MergeResolution):
    memory = pipeline.load_memory()
    if merge_id not in memory.get("pending_merges", {}):
        raise HTTPException(status_code=404, detail="merge not found")
    pipeline.resolve_pending_merge(merge_id, resolution.confirmed, memory)
    return {"status": "ok"}


class ManualMergeRequest(BaseModel):
    primary_person_id: str
    secondary_person_ids: list[str]


#
# manual merge: the professor has spotted duplicate records themselves
# (e.g. while searching by name) and wants to merge them directly --
# no detection/evidence involved, they've already made the judgment call
#
@app.post("/api/merges/manual")
def manual_merge(req: ManualMergeRequest):
    memory = pipeline.load_memory()
    if req.primary_person_id not in memory.get("people", {}):
        raise HTTPException(status_code=404, detail="primary person not found")

    merged = []
    for secondary_id in req.secondary_person_ids:
        if secondary_id == req.primary_person_id:
            continue
        if secondary_id not in memory.get("people", {}):
            continue
        pipeline.merge_people(req.primary_person_id, secondary_id, memory)
        merged.append(secondary_id)

    return {"status": "ok", "primary_person_id": req.primary_person_id, "merged": merged}


# serve the frontend itself
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
