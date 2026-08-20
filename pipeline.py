#
# imports and OpenAI client setup
#
import json
import re
import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# load_dotenv()
# client = OpenAI()
# MODEL = "gpt-4o-mini"

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)
MODEL = "google/gemma-4-e4b"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MEMORY_PATH = DATA_DIR / "memory.json"
EMAILS_PATH = DATA_DIR / "emails.json"
CONFIG_PATH = DATA_DIR / "config.json"


#
# config.json holds the professor's own address(es) -- kept separate from
# memory.json so resetting the CRM's people/bullets doesn't wipe this
#
def load_config():
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
        config.setdefault("user_emails", [])
        return config
    return {"user_emails": []}


def save_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def get_user_emails():
    return {addr.lower() for addr in load_config().get("user_emails", [])}


#
# memory.json maps a person's email address to their record
#
def load_memory():
    if MEMORY_PATH.exists():
        memory = json.loads(MEMORY_PATH.read_text())
        memory.setdefault("processed_emails", [])
        memory.setdefault("pending_merges", {})
        memory.setdefault("next_merge_id", 1)
        return memory
    return {
        "people": {},
        "email_index": {},
        "processed_emails": [],
        "pending_merges": {},
        "next_merge_id": 1,
    }


def save_memory(memory):
    MEMORY_PATH.write_text(json.dumps(memory, indent=2))


#
# emails.json is the source of truth for the inbox (real or fake)
#
def load_emails():
    if EMAILS_PATH.exists():
        return json.loads(EMAILS_PATH.read_text())
    return {}


#
# Extract emailref when self-addressed email
#
def extract_emailref(line):
    if line[:9].lower() == "emailref:":
        email = line[9:].strip().lower()
        if email != "" and email.find(" ") == -1:
            return email

    return None


#
# pull the first non-blank-safe line of a body, so callers can hand
# extract_emailref a line rather than the whole body
#
def first_line(body):
    lines = body.strip().splitlines()
    return lines[0].strip() if lines else ""


#
# who is this email about? returns a list of (address, role) pairs.
#
def identify_people(email):
    ref = extract_emailref(first_line(email["body"]))
    if ref:
        return [(ref, "primary")]

    user_emails = get_user_emails()
    people = []
    seen = set()

    def add(addr, role):
        addr = addr.lower()
        if addr not in user_emails and addr not in seen:
            people.append((addr, role))
            seen.add(addr)

    if email["from"].lower() not in user_emails:
        add(email["from"], "primary")
    for addr in email.get("to", []):
        add(addr, "primary")
    for addr in email.get("cc", []):
        add(addr, "looped_in")
    for addr in email.get("bcc", []):
        add(addr, "looped_in")

    return people


#
# when an emailref line is present, drop it from the body before the
# LLM sees it -- it's routing metadata, not content about the person
#
def strip_emailref_line(body):
    if extract_emailref(first_line(body)):
        return "\n".join(body.strip().splitlines()[1:]).strip()
    return body


#
# strip a forwarded-message block out of the body
#
FORWARD_START_PATTERNS = [
    r"-{3,}\s*Forwarded message\s*-{3,}",
    r"-{3,}\s*Original Message\s*-{3,}",
]
FORWARD_END_PATTERNS = [
    r"-{3,}\s*End forwarded message\s*-{3,}",
]

def strip_forwarded_content(body):
    start_pattern = "|".join(FORWARD_START_PATTERNS)
    end_pattern = "|".join(FORWARD_END_PATTERNS)

    start_match = re.search(start_pattern, body, flags=re.IGNORECASE)
    if not start_match:
        return body

    remainder = body[start_match.end():]
    end_match = re.search(end_pattern, remainder, flags=re.IGNORECASE)

    before = body[:start_match.start()].rstrip()
    if end_match:
        after = remainder[end_match.end():].lstrip()
        return (before + "\n\n" + after).strip() if after else before
    else:
        return before


#
# cleaning email of forwarded content
#
def clean_body(body):
    return strip_forwarded_content(strip_emailref_line(body))


#
# resolve a raw email address to a person_id via the reverse index
#
def resolve_person(address, memory):
    address = address.lower()
    if address in memory["email_index"]:
        return memory["email_index"][address]
    memory["email_index"][address] = address
    return address


#
# schema + prompt for detecting whether a brand-new address actually
# belongs to someone already in the CRM, based on explicit
# self-identifying language in the email -- not just a similar name
#
MERGE_DETECTION_SCHEMA = {
    "name": "merge_detection",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "is_same_person": {"type": "boolean"},
            "candidate_person_id": {"type": ["string", "null"]},
            "evidence": {"type": ["string", "null"]},
        },
        "required": ["is_same_person", "candidate_person_id", "evidence"],
        "additionalProperties": False,
    },
}

MERGE_DETECTION_SYSTEM = """
You maintain a CRM for a professor. A new email address has contacted the professor for the first time. Your job is to decide whether this
new address actually belongs to someone ALREADY in the CRM, based on explicit self-identifying language in the email -- not just a similar name.

Only answer true if the email itself explicitly says something that identifies the sender as one of the existing people below -- for example,
stating their own full name, or explaining they're writing from a different account than usual. A name that merely sounds similar, or an
initial-only sign-off, is NOT enough evidence on its own.

Example of what NOT to do: an email signed "Thanks, A. Chen" does not, by itself, explicitly claim to be any specific existing person -- do not
match it to "Alice Chen" just because the surname matches. Answer false.

Example of what TO do: an email that says "This is Alice Chen, emailing from my personal account since I'm locked out" explicitly identifies
the sender by full name -- if "Alice Chen" is in the list of existing people below, answer true and set candidate_person_id to her id.

Existing people in the CRM:
{people_list}

If is_same_person is true, set candidate_person_id to the exact person_id from the list above, and evidence to the specific phrase that
justifies the match. If false, set both candidate_person_id and evidence to null.
"""

def llm_detect_merge(email, new_address, memory):
    existing = memory["people"]
    if not existing:
        return {"is_same_person": False, "candidate_person_id": None, "evidence": None}

    people_list = "\n".join(
        f"- {pid}: {rec.get('display_name') or pid} ({', '.join(rec.get('emails', []))})"
        for pid, rec in existing.items()
    )
    system_prompt = MERGE_DETECTION_SYSTEM.format(people_list=people_list)
    user_prompt = f"""
New address: {new_address}

Email:
Subject: {email['subject']}

{clean_body(email['body'])}
"""
    return call_llm(system_prompt, user_prompt, MERGE_DETECTION_SCHEMA)


#
# fold secondary_id's addresses and bullets into primary_id, then
# delete the secondary record entirely. bullets are merged with the
# same similarity check used for dedup during normal processing --
# no extra LLM call needed for this part.
#
def merge_people(primary_id, secondary_id, memory):
    if primary_id == secondary_id:
        return
    primary = memory["people"].get(primary_id)
    secondary = memory["people"].pop(secondary_id, None)
    if primary is None or secondary is None:
        return

    for addr in secondary["emails"]:
        if addr not in primary["emails"]:
            primary["emails"].append(addr)
        memory["email_index"][addr] = primary_id

    for bullet in secondary["bullets"]:
        duplicate = next(
            (b for b in primary["bullets"] if bullets_are_similar(b["content"], bullet["content"])),
            None,
        )
        if duplicate:
            for ref in bullet["ref"]:
                if ref not in duplicate["ref"]:
                    duplicate["ref"].append(ref)
            if bullet["last_updated"] > duplicate["last_updated"]:
                duplicate["last_updated"] = bullet["last_updated"]
        else:
            primary["bullets"].append(bullet)

    save_memory(memory)


#
# apply (or reject) a pending merge suggestion. called directly from
# the review UI -- no email threading involved.
#
def resolve_pending_merge(merge_id, confirmed, memory):
    pending = memory["pending_merges"].get(merge_id)
    if not pending or pending["status"] != "pending":
        print(f"{merge_id}: not found or already resolved")
        return

    if confirmed:
        merge_people(pending["candidate_person_id"], pending["new_person_id"], memory)
        pending["status"] = "merged"
        print(f"{merge_id}: merged {pending['new_person_id']} into {pending['candidate_person_id']}")
    else:
        pending["status"] = "rejected"
        print(f"{merge_id}: rejected, kept as separate people")

    save_memory(memory)


#
# JSON Schemas for structured outputs
#
FIRST_SUMMARY_SCHEMA = {
    "name": "first_summary",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "display_name": {"type": ["string", "null"]},
            "bullets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "durability": {"type": "string", "enum": ["durable", "transient"]},
                    },
                    "required": ["content", "durability"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["display_name", "bullets"],
        "additionalProperties": False,
    },
}

UPDATE_SCHEMA = {
    "name": "summary_update",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "bullets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["new", "updated"]},
                        "content": {"type": "string"},
                        "durability": {"type": "string", "enum": ["durable", "transient"]},
                        "updated": {"type": ["integer", "null"]},
                    },
                    "required": ["type", "content", "durability", "updated"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["bullets"],
        "additionalProperties": False,
    },
}

RELEVANCE_SCHEMA = {
    "name": "cc_relevance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "relevant": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["relevant"],
        "additionalProperties": False,
    },
}


#
# call helper: schema goes in response_format, so the reply is
# guaranteed-valid JSON
#
def call_llm(system_prompt, user_prompt, schema):
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    return json.loads(response.choices[0].message.content)


#
# first email about a person -> fresh summary
#
FIRST_SUMMARY_SYSTEM = """
You maintain a CRM for a professor. Given an email, summarize what it reveals about the person it concerns as short, factual bullet points.
Each bullet must cover exactly ONE atomic fact -- if a sentence has multiple clauses (e.g. joined by 'and'), split it into separate bullets.

Example of what NOT to do: 'She is a PhD candidate in the NLP Lab and is defending in spring 2027' is TWO facts merged into one bullet.
Instead write two bullets: 'PhD candidate in the NLP Lab.' and 'Defending thesis in spring 2027.'

Only include facts the email explicitly states. Do NOT infer, guess, or add plausible-sounding details the email does not actually contain,
even if they seem likely. If nothing meaningful can be said, return an empty bullets list rather than inventing content.

The email may mention multiple people (sender, recipients, CC'd people, or people quoted in the body). Only extract facts specifically about the
target person named below -- ignore facts about anyone else mentioned, even in the same sentence or paragraph.

Example of what NOT to do: if the email is from Carla and says 'cc'ing Alice since she offered to help me with exam prep, and could you write
me a recommendation letter,' and the target person is Alice, do NOT include a bullet about the recommendation letter -- that fact belongs
to Carla, not Alice.

Example of what NOT to do: an email that only says 'I'm starting my PhD in the NLP Lab this fall' does not mention a meeting -- do not
add a bullet like 'Meeting scheduled' just because professors and students often meet.

For each bullet, set durability to 'durable' for lasting facts -- identity, role, ongoing research, relationships, achievements -- or
'transient' for short-lived facts -- logistics, one-off requests, pending events that will soon be irrelevant.

Set display_name to the person's full name if the email reveals it, otherwise null.
"""

def llm_first_summary(person_email, email):
    user_prompt = f"""
TARGET PERSON: {person_email}
Extract facts about this person ONLY, not about anyone else mentioned in the email below.

Date: {email["timestamp"][:10]}
Subject: {email['subject']}

{clean_body(email['body'])}
"""
    return call_llm(FIRST_SUMMARY_SYSTEM, user_prompt, FIRST_SUMMARY_SCHEMA)


#
# later emails -> a diff against the existing numbered bullets
#
UPDATE_SYSTEM = """
You maintain a CRM for a professor. You will get the existing bullet-point summary of a person (numbered from 1) and a new email about them. Decide
what the email adds or changes:
- type 'new' = a fact not covered by any existing bullet; set updated to null
- type 'updated' = the email changes/supersedes an existing bullet; set updated to that bullet's number and content to the full rewritten bullet
- Do NOT include bullets that are unchanged by this email, even if the email mentions or confirms them again
- Each returned bullet must be exactly ONE atomic fact -- never merge an existing bullet's content with a new fact into a single bullet
- Only include facts the email explicitly states. Do NOT infer, guess, or add specific details (dates, numbers, names) that the email does not
actually contain, even if a vague version was mentioned elsewhere
- The email may mention multiple people. Only include facts specifically about the target person named below -- ignore facts about anyone else
mentioned, CC'd, or quoted in the same email
- For each bullet, set durability to 'durable' for lasting facts -- identity, role, ongoing research, relationships, achievements -- or
'transient' for short-lived facts -- logistics, one-off requests, pending events that will soon be irrelevant

Example of what NOT to do: if the email says a task will be done 'sometime in the fall,' do not write a bullet with a specific date like
'October 15th' -- keep the vagueness the email actually used.

Example of what NOT to do: if bullet 2 already says 'PhD candidate in the NLP Lab' and the new email just confirms this while also mentioning
she is TAing this fall, do NOT return a bullet that repeats bullet 2's content merged with the TA fact. Only return the new TA fact as its own
bullet, and leave bullet 2 out of your response entirely.

Return an empty bullets array if the email adds nothing about the person.
"""

def llm_update_summary(person_email, email, bullets):
    numbered = "\n".join(f"{i+1}. {b['content']}" for i, b in enumerate(bullets))
    user_prompt = f"""
TARGET PERSON: {person_email}
Extract facts about this person ONLY, not about anyone else mentioned in the email below.

Existing summary of {person_email}:
{numbered}

New email, dated {email["timestamp"][:10]}:
Subject: {email['subject']}

{clean_body(email['body'])}
"""
    return call_llm(UPDATE_SYSTEM, user_prompt, UPDATE_SCHEMA)


RELEVANCE_SYSTEM = """
You maintain a CRM for a professor. An email has been CC'd or BCC'd to one or more people who are not the main recipient. Your job is to decide
which of them the email substantively says something about, versus which are just included for visibility with no real content concerning them.

Example: if the email says 'cc'ing Alice since she offered to help me with exam prep,' that IS substantive -- Alice is doing something specific.
Example: if Alice is CC'd on a routine logistics email that never mentions her, that is NOT substantive -- she's just included for awareness.

Return only the addresses (exact strings from the provided list) that meet the substantive bar. Return an empty list if none do.
"""

def llm_classify_cc_relevance(email, looped_in_addresses):
    if not looped_in_addresses:
        return []
    addr_list = "\n".join(looped_in_addresses)
    user_prompt = f"""
CC/BCC'd addresses:
{addr_list}

Email:
Subject: {email['subject']}

{clean_body(email['body'])}
"""
    result = call_llm(RELEVANCE_SYSTEM, user_prompt, RELEVANCE_SCHEMA)
    return [a.lower() for a in result["relevant"]]


#
# crude but cheap similarity check: fraction of the shorter bullet's
# words that also appear in the longer one. good enough to catch
# near-verbatim restatement without needing another LLM call
#
def bullets_are_similar(a, b, threshold=0.75):
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    smaller = min(len(words_a), len(words_b))
    return (overlap / smaller) >= threshold


#
# turn an LLM diff into changes on the stored record. the LLM's
# 1-based 'updated' index maps to our 0-based list. before overwriting
# an existing bullet, its current version is snapshotted into history.
# 'new' bullets that are near-duplicates of an existing bullet get
# folded in as a ref addition instead of appended as a fresh bullet --
# a safety net for models that restate known facts instead of omitting them
#
def apply_update(record, diff, email_id, email_date):
    for item in diff.get("bullets", []):
        if not item["content"].strip():
            continue
        if item["type"] == "updated" and item.get("updated"):
            idx = item["updated"] - 1
            if 0 <= idx < len(record["bullets"]):
                bullet = record["bullets"][idx]
                bullet["history"].append({
                    "content": bullet["content"],
                    "ref": list(bullet["ref"]),
                    "created": bullet["created"],
                    "last_updated": bullet["last_updated"],
                })
                bullet["content"] = item["content"]
                bullet["durability"] = item["durability"]
                bullet["last_updated"] = email_date
                if email_id not in bullet["ref"]:
                    bullet["ref"].append(email_id)
                continue
            else:
                print(f"  [warn] {email_id}: 'updated' index {item['updated']} out of range "
                      f"(record has {len(record['bullets'])} bullets) -- falling back to new/dup check")

        # check for near-duplicate before treating as a genuinely new bullet
        duplicate = next(
            (b for b in record["bullets"]
             if bullets_are_similar(b["content"], item["content"])),
            None,
        )
        if duplicate:
            duplicate["last_updated"] = email_date
            if email_id not in duplicate["ref"]:
                duplicate["ref"].append(email_id)
            continue

        # genuinely new, or an 'updated' item pointing at a bad index
        record["bullets"].append({
            "content": item["content"],
            "durability": item["durability"],
            "ref": [email_id],
            "created": email_date,
            "last_updated": email_date,
            "history": [],
        })


#
# quick, deterministic check for auto-generated / no-reply mail
#
NOISE_SENDER_PATTERNS = ["noreply", "no-reply", "donotreply", "do-not-reply", "calendar-"]
NOISE_SUBJECT_PATTERNS = ["invitation:", "automatic reply", "out of office"]

def is_auto_generated(email):
    sender = email["from"].lower()
    subject = email.get("subject", "").lower()
    if any(p in sender for p in NOISE_SENDER_PATTERNS):
        return True
    if any(p in subject for p in NOISE_SUBJECT_PATTERNS):
        return True
    return False


#
# full pipeline for one email: identify the person, resolve to a
# person_id, branch on first-contact vs update, apply, persist
#
def process_email(email_id, email, memory):

    if email_id in memory["processed_emails"]:
        print(f"{email_id}: already processed, skipped")
        return

    if is_auto_generated(email):
        print(f"{email_id}: auto-generated, skipped")
        memory["processed_emails"].append(email_id)
        return

    if len(strip_emailref_line(email["body"]).strip()) < 10:
        print(f"{email_id}: body empty or too short, skipped")
        memory["processed_emails"].append(email_id)
        return

    people = identify_people(email)
    if not people:
        print(f"{email_id}: could not identify anyone, skipped")
        memory["processed_emails"].append(email_id)
        return

    primaries = [addr for addr, role in people if role == "primary"]
    looped_in = [addr for addr, role in people if role == "looped_in"]

    relevant_looped_in = set(llm_classify_cc_relevance(email, looped_in))

    to_process = primaries + [addr for addr in looped_in if addr in relevant_looped_in]

    if not to_process:
        print(f"{email_id}: no relevant people to process, skipped")
        memory["processed_emails"].append(email_id)
        return

    for address in to_process:
        is_new_address = address not in memory["email_index"]

        if is_new_address and memory["people"]:
            detection = llm_detect_merge(email, address, memory)
            if detection["is_same_person"] and detection["candidate_person_id"] in memory["people"]:
                merge_id = f"merge_{memory['next_merge_id']:04d}"
                memory["next_merge_id"] += 1
                memory["pending_merges"][merge_id] = {
                    "new_person_id": address,
                    "candidate_person_id": detection["candidate_person_id"],
                    "evidence": detection["evidence"],
                    "source_email_id": email_id,
                    "created": email["timestamp"][:10],
                    "status": "pending",
                }
                print(f"{email_id}: [merge] suggested {merge_id}: {address} -> "
                      f"{detection['candidate_person_id']} ({detection['evidence']})")

        person_id = resolve_person(address, memory)

        if person_id not in memory["people"]:
            result = llm_first_summary(address, email)
            memory["people"][person_id] = {
                "display_name": result.get("display_name") if result.get("display_name") not in (None, "", "null") else address,
                "emails": [address],
                "bullets": [
                    {
                        "content": b["content"],
                        "durability": b["durability"],
                        "ref": [email_id],
                        "created": email["timestamp"][:10],
                        "last_updated": email["timestamp"][:10],
                        "history": [],
                    }
                    for b in result["bullets"] if b["content"].strip()
                ],
            }
            print(f"{email_id}: created {person_id}")
        else:
            record = memory["people"][person_id]
            diff = llm_update_summary(address, email, record["bullets"])
            apply_update(record, diff, email_id, email["timestamp"][:10])
            print(f"{email_id}: updated {person_id} ({len(diff.get('bullets', []))} changes)")

    memory["processed_emails"].append(email_id)
    save_memory(memory)
