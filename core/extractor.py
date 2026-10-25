"""Promise/decision extraction for Elephant.

Reads a single chat message (with a little surrounding context) and asks Claude
to pull out concrete promises and group decisions. The model is told to answer
with strict JSON; anything that doesn't parse cleanly yields an empty list, so a
flaky response never injects junk into memory.
"""

import datetime
import json
import os
import re
import zoneinfo

import anthropic
import dateparser
from dotenv import load_dotenv

load_dotenv()

PHOENIX = zoneinfo.ZoneInfo("America/Phoenix")
DEFAULT_MODEL = "claude-haiku-5-5"

# ============================================================================
# Time-resolution defaults  (tweak these to change how phrases are resolved)
# ============================================================================
# Weekday names and abbreviations -> Python weekday() index (Monday = 0).
WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

# Part-of-day word -> hour (24h). "tonight" uses the "night" entry.
PART_OF_DAY_HOURS = {
    "morning": 9,     # 9am
    "afternoon": 14,  # 2pm
    "evening": 18,    # 6pm
    "night": 21,      # 9pm
}

# Default time when a weekday is named with no time or part of day.
DEFAULT_WEEKDAY_HOUR = 17  # 5pm

# Bare-hour (no am/pm) interpretation: 1-7 -> pm, 8-11 -> am, 12 -> noon.
PM_BARE_HOURS = {1, 2, 3, 4, 5, 6, 7}  # add 12 to reach the afternoon/evening
AM_BARE_HOURS = {8, 9, 10, 11}         # kept as-is (morning)
# 12 with no am/pm is treated as noon (12:00).
# ============================================================================

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
- "promise": the SPEAKER commits THEMSELVES, in their own words, to doing a \
  specific thing ("I'll send the notes tonight", "I can pay you back Friday"). \
  It must be a first-person commitment the speaker is making right now. These \
  are NOT promises: a question ("can you send the notes?"), a suggestion or \
  assignment aimed at someone else ("you should book the room", "maya can do \
  the slides"), a wish or maybe ("I should really start studying"), or someone \
  else's promise being quoted or recalled. If in doubt, it is not a promise.
- "decision": the group agrees on a concrete plan. \
  ("ok let's meet Thursday at 6 at Hayden Library")

Examples:
- "let's do Saturday night on Mill Ave" -> a decision (the group is settling a \
  plan): {"kind": "decision", "who": "group", "what": "go out on mill ave", \
  "due_phrase": "Saturday night"}
- "I can book the room Wednesday" -> a promise (a first-person commitment): \
  {"kind": "promise", "who": "<speaker>", "what": "book the room", \
  "due_phrase": "Wednesday"}

Only extract promises and decisions introduced by the NEW message being \
analyzed. The recent messages are there ONLY as context, to help you resolve \
what the new message refers to (for instance a confirmation like "that works"). \
NEVER extract an item from a recent message -- if the new message itself \
introduces nothing, return an empty array.

Return an EMPTY array for casual chatter, questions, jokes, venting, and \
tentative maybes ("maybe we should...", "we could grab food sometime"). Being \
wrong about silence is better than extracting junk.

Never create a second item for a decision the group is only confirming. A bare \
agreement to an existing plan ("sounds good", "works for me", "bet", "I'm in") \
is NOT a new decision -- return an empty array for it.

Keep "what" SHORT and in plain lowercase, phrased the way the person said it \
(for a promise, "send the slides outline", not "Maya will put together and \
send the slides outline"). Do not add detail they did not say.

Do NOT compute dates or timestamps yourself. If the message mentions when \
something is due, copy the time phrase verbatim into `due_phrase` ("tonight", \
"Friday", "tomorrow at 5", "by next week"). Use null when there is no time \
mentioned. The phrase is resolved to an actual date later, outside of you.

Output format: a JSON array (possibly empty) of objects, each with exactly these \
keys:
  "kind":       "promise" or "decision"
  "who":        who is responsible (a name, or "group")
  "what":       a short description of the promise or decision
  "due_phrase": the time phrase as written, or null

Output ONLY the JSON array. No prose, no markdown fences, no explanation.\
"""


# Colloquial today-phrases dateparser doesn't understand, mapped to forms it
# does. (Weekday phrases, "tomorrow" and "tonight" are handled in Python below.)
_NORMALIZE = {
    "this evening": "today 18:00",
    "this afternoon": "today 15:00",
    "this morning": "today 09:00",
    "noon": "today 12:00",
    "midnight": "today 23:59",
}
_LEADING = re.compile(r"^(by|before|around|about|due)\s+")

# A clock time like "6", "at 5pm", "3:30", "at 17:00".
_TIME_RE = re.compile(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b")


def _normalize_phrase(phrase):
    q = phrase.strip().lower()
    q = _LEADING.sub("", q)
    return _NORMALIZE.get(q, q)


def _bare_hour_to_24(hour):
    """Apply the no-am/pm convention: 1-7 pm, 8-11 am, 12 noon."""
    if hour in PM_BARE_HOURS:
        return hour + 12
    if hour == 12:
        return 12
    return hour  # 8-11 stay as morning; 0 or >12 pass through untouched


def _parse_clock_or_part(text):
    """Return (hour, minute) from an explicit time or a part-of-day word.

    Explicit clock times win over part-of-day words. Returns None when the
    text carries neither.
    """
    m = _TIME_RE.search(text)
    if m and (m.group(2) or m.group(3) or m.group(1)):
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3)
        if ampm == "pm":
            hour = 12 if hour == 12 else hour + 12
        elif ampm == "am":
            hour = 0 if hour == 12 else hour
        else:
            hour = _bare_hour_to_24(hour)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
    for word, hour in PART_OF_DAY_HOURS.items():
        if re.search(r"\b" + word + r"\b", text):
            return hour, 0
    return None


def _combine(base, target_date, hour, minute):
    dt = datetime.datetime(
        target_date.year, target_date.month, target_date.day,
        hour, minute, tzinfo=PHOENIX,
    )
    return int(dt.timestamp())


def _resolve_weekday_phrase(phrase, now):
    """Resolve weekday / tomorrow / tonight phrases in plain Python.

    Returns a Unix timestamp, or None if the phrase isn't one of these forms
    (so the caller can fall back to dateparser).
    """
    q = phrase.strip().lower()
    base = datetime.datetime.fromtimestamp(now, PHOENIX)

    # "tonight" -> today at the night hour.
    if q == "tonight":
        return _combine(base, base.date(), PART_OF_DAY_HOURS["night"], 0)

    # "tomorrow [time/part]" -> the next calendar day.
    if q == "tomorrow" or q.startswith("tomorrow "):
        target_date = (base + datetime.timedelta(days=1)).date()
        clock = _parse_clock_or_part(q[len("tomorrow"):])
        hour, minute = clock if clock else (DEFAULT_WEEKDAY_HOUR, 0)
        return _combine(base, target_date, hour, minute)

    # Optional "this"/"next" prefix, then a weekday somewhere in the phrase.
    is_next = False
    pref = re.match(r"(this|next)\s+", q)
    rest = q
    if pref:
        is_next = pref.group(1) == "next"
        rest = q[pref.end():]

    wd_index = None
    for word in re.findall(r"[a-z]+", rest):
        if word in WEEKDAYS:
            wd_index = WEEKDAYS[word]
            break
    if wd_index is None:
        return None  # not a weekday phrase -> let dateparser try

    clock = _parse_clock_or_part(rest)
    hour, minute = clock if clock else (DEFAULT_WEEKDAY_HOUR, 0)

    base_days = (wd_index - base.weekday()) % 7  # 0..6, today = 0
    if is_next:
        days_ahead = base_days + 7  # the weekday of the following week
    else:
        days_ahead = base_days
        if days_ahead == 0:
            # Today is that weekday: use today only if the time is still ahead.
            candidate = base.replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            if candidate <= base:
                days_ahead = 7

    target_date = (base + datetime.timedelta(days=days_ahead)).date()
    return _combine(base, target_date, hour, minute)


def _resolve_due(phrase, now):
    """Resolve a relative time phrase to a Unix timestamp, or None.

    Weekday / tomorrow / tonight forms are resolved in Python (see
    `_resolve_weekday_phrase`); anything else falls back to dateparser,
    relative to `now`, in America/Phoenix, preferring future dates. Whatever
    neither can make sense of yields None.
    """
    if not phrase or not str(phrase).strip():
        return None

    weekday = _resolve_weekday_phrase(str(phrase), now)
    if weekday is not None:
        return weekday

    base = datetime.datetime.fromtimestamp(now, PHOENIX).replace(tzinfo=None)
    dt = dateparser.parse(
        _normalize_phrase(str(phrase)),
        settings={
            "RELATIVE_BASE": base,
            "TIMEZONE": "America/Phoenix",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "PREFER_DATES_FROM": "future",
        },
    )
    if dt is None:
        return None
    return int(dt.timestamp())


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


def _parse_items(raw_text, now):
    """Parse the model's reply into a validated list of item dicts.

    `due_phrase` from the model is resolved to a `due` Unix timestamp (or None)
    in Python, relative to `now`.
    """
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
        if kind not in ("promise", "decision"):
            continue
        if not who or not what:
            continue
        due = _resolve_due(entry.get("due_phrase"), now)
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
    return _parse_items(raw_text, now)
