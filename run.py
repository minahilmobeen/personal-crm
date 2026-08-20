#
# entry point: process the inbox (data/emails.json) through the
# pipeline in chronological order, then persist to data/memory.json
#
import pipeline

def main():
    emails = pipeline.load_emails()
    if not emails:
        print(f"No emails found at {pipeline.EMAILS_PATH} -- nothing to process.")
        return

    memory = pipeline.load_memory()

    for email_id, email in sorted(emails.items(), key=lambda kv: kv[1]["timestamp"]):
        pipeline.process_email(email_id, email, memory)

    print(f"\nDone. {len(memory['people'])} people tracked, "
          f"{len(memory['processed_emails'])} emails processed.")


if __name__ == "__main__":
    main()
