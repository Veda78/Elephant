"""Promise/decision extraction for Elephant.

Reads a single chat message (with a little surrounding context) and asks Claude
to pull out concrete promises and group decisions. The model is told to answer
with strict JSON; anything that doesn't parse cleanly yields an empty list, so a
flaky response never injects junk into memory.
"""

import datetime
import json
import os
import zoneinfo

import anthropic
from dotenv import load_dotenv

load_dotenv()

PHOENIX = zoneinfo.ZoneInfo("America/Phoenix")
DEFAULT_MODEL = "claude-haiku-5-5"

# Cached client, built on first use so importing the module needs no API key
# (and so tests can inject their own client instead).
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


SYSTEM_PROMPT = """\
You are Elephant, a quiet observer of a group chat. Your only job is to notice \
when someone makes a *promise* or the group reaches a *decision*, and to report \
them as strict JSON. You are deliberately conservative: a missed item is fine, a \
made-up item is not.

Definitions:
- "promise": one person commits to doing a specific thing. \
  ("I'll send the notes tonight", "I can pay you back Friday")
- "decision": the group agrees on a concrete plan. \
  ("ok let's meet Thursday at 6 at Hayden Library")

Return an EMPTY array for casual chatter, questions, jokes, venting, and \
tentative maybes ("maybe we should...", "we could grab food sometime"). Being \
wrong about silence is better than extracting junk.

A short reply like "yes, sounds good" or "works for me" can confirm a decision \
proposed in the recent messages. In that case, record the decision it agrees \
with, attributing `who` to the group.

Resolve relative times ("tonight", "Friday", "tomorrow at 5") against the \
provided current time, which is in the America/Phoenix time zone. Express `due` \
as an integer Unix timestamp (seconds). Use null when there is no clear time.

Output format: a JSON array (possibly empty) of objects, each with exactly these \
keys:
  "kind": "promise" or "decision"
  "who":  who is responsible (a name, or "group")
  "what": a short description of the promise or decision
  "due":  integer Unix timestamp, or null

Output ONLY the JSON array. No prose, no markdown fences, no explanation.\
"""


def _build_user_prompt(message_text, author, now, recent_messages):
    now_dt = datetime.datetime.fromtimestamp(now, tz=PHOENIX)
    now_str = now_dt.strftime("%A, %Y-%m-%d %H:%M %Z")

    lines = [
        f"Current time: {now_str} (Unix {int(now)})",
        "",
    ]
    if recent_messages:
        lines.append("Recent messages (oldest first), for context:")
        for msg in recent_messages:
            # Accept either {"author", "text"} dicts or plain strings.
            if isinstance(msg, dict):
                lines.append(f"  {msg.get('author', '?')}: {msg.get('text', '')}")
            else:
                lines.append(f"  {msg}")
        lines.append("")
    lines.append("New message to analyze:")
    lines.append(f"  {author}: {message_text}")
    return "\n".join(lines)


def _parse_items(raw_text):
    """Parse the model's reply into a validated list of item dicts."""
    text = raw_text.strip()
    # Tolerate a stray ```json ... ``` fence even though we asked for none.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return []

    if not isinstance(data, list):
        return []

    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        who = entry.get("who")
        what = entry.get("what")
        due = entry.get("due")
        if kind not in ("promise", "decision"):
            continue
        if not who or not what:
            continue
        if due is not None:
            try:
                due = int(due)
            except (ValueError, TypeError):
                due = None
        items.append({"kind": kind, "who": who, "what": what, "due": due})
    return items


def extract(message_text, author, now, recent_messages, client=None):
    """Extract promises and decisions from one message.

    Returns a list of dicts, each with keys: kind, who, what, due. Returns an
    empty list for chatter or if the model's reply cannot be parsed safely.
    `client` may be injected for testing; otherwise a shared client is used.
    """
    client = client or _get_client()
    model = os.environ.get("ELEPHANT_MODEL", DEFAULT_MODEL)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            output_config={"effort": "low"},
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        message_text, author, now, recent_messages
                    ),
                }
            ],
        )
    except anthropic.APIError:
        return []

    raw_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return _parse_items(raw_text)
