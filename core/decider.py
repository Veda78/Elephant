"""Decide whether Elephant should speak, and why.

The guiding principle is restraint: Elephant stays quiet unless it has a good
reason to speak. Cheap, deterministic checks (mentions, keywords, memory
lookups) run first; the Anthropic API is only consulted for the genuinely
ambiguous cases -- composing an answer from memory, or judging whether a message
contradicts a standing decision.

Every decision carries a human-readable `reason`, because the demo surfaces why
Elephant chose to stay quiet.
"""

import datetime
import json
import os
import re

import anthropic

from core import extractor


# ---------------------------------------------------------------------------
# Unprompted-speech cooldown (cases 3 and 4 only; mentions/questions bypass it)
# ---------------------------------------------------------------------------
COOLDOWN_SECONDS = 600   # at most one unprompted message every 10 minutes...
MAX_PER_HOUR = 4         # ...and at most 4 in any rolling hour, per channel.
_HOUR = 3600

# channel -> list of timestamps when Elephant last spoke unprompted
_speak_log = {}

DEFAULT_CHANNEL = "general"


def reset_cooldowns():
    """Clear cooldown state (used by tests)."""
    _speak_log.clear()


def _recent_unprompted(channel, now):
    log = [t for t in _speak_log.get(channel, []) if now - t < _HOUR]
    _speak_log[channel] = log
    return log


def _cooldown_ok(channel, now):
    log = _recent_unprompted(channel, now)
    if any(now - t < COOLDOWN_SECONDS for t in log):
        return False
    return len(log) < MAX_PER_HOUR


def _register_unprompted(channel, now):
    _speak_log.setdefault(channel, []).append(now)


# ---------------------------------------------------------------------------
# Cheap text heuristics
# ---------------------------------------------------------------------------
_QUESTION_WORDS = {
    "what", "who", "when", "where", "why", "how",
    "did", "do", "does", "is", "are", "was", "were", "should", "can", "will",
}
_PAST_CUES = ("what did we", "who said", "when are we", "when do we",
              "what's the plan", "whats the plan", "did we decide",
              "what did i", "when did")
_CONFLICT_CUES = ("actually", "instead", "change", "changed", "move", "moved",
                  "reschedule", "let's do", "lets do", "can we do", "make it")
_STOPWORDS = {
    "the", "and", "that", "this", "with", "about", "they", "them", "then",
    "were", "what", "when", "where", "which", "would", "could", "should",
    "your", "yours", "have", "will", "just", "like", "from", "into", "over",
    "elephant", "going", "gonna", "wanna",
}
_TIME_RE = re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b")


def _looks_like_question(text):
    t = text.strip().lower()
    if t.endswith("?"):
        return True
    if any(cue in t for cue in _PAST_CUES):
        return True
    first = t.split(" ", 1)[0] if t else ""
    return first in _QUESTION_WORDS


def _mentions_time(text):
    t = text.lower()
    if _TIME_RE.search(t):
        return True
    return any(wd in re.findall(r"[a-z]+", t) for wd in extractor.WEEKDAYS)


def _keywords(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 4 and w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# API helpers (only reached for ambiguous cases)
# ---------------------------------------------------------------------------
def _complete(client, system, user, max_tokens=200):
    """Single short completion. Returns stripped text, or None on failure."""
    client = client or extractor._get_client()
    model = os.environ.get("ELEPHANT_MODEL", extractor.DEFAULT_MODEL)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            output_config={"effort": "low"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIError:
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


ANSWER_SYSTEM = """\
You are Elephant, a warm, low-key group-chat helper. Using ONLY the context \
provided (open items the group is tracking, and matching past messages), decide \
whether you can answer the user's question.

Respond with STRICT JSON only, one of:
  {"answerable": false}
  {"answerable": true, "reply": "<at most two short, friendly sentences>"}

Answer only from the context. If the context does not actually contain the \
answer, respond {"answerable": false} -- never guess or make something up. \
Output ONLY the JSON object. No prose, no markdown fences.\
"""

CONFLICT_SYSTEM = """\
You check whether a new message conflicts with a decision the group already \
made. You are given the open decisions and the new message.

A conflict means the new message changes the SAME event or SAME time slot as \
one of the open decisions -- same plan, now at a different time or place (the \
group decided Thursday at 6, the message now says Thursday at 7).

A different plan, on a different day, or in a different place is NOT a \
conflict. Adding a new, separate activity is NOT a conflict. For example, if \
the open decision is "meet Thursday at 6 to study" and the message is "let's \
go out Saturday night", that is a separate plan -- not a conflict.

Respond with STRICT JSON only, one of:
  {"conflict": false}
  {"conflict": true, "line": "<ONE short, friendly sentence pointing out the clash>"}

Never invent a conflict; when in doubt, respond {"conflict": false}. Output \
ONLY the JSON object. No prose, no markdown fences.\
"""


def _due_label(due):
    """Render a stored epoch `due` as a readable local time for the model."""
    if not due:
        return ""
    dt = datetime.datetime.fromtimestamp(due, extractor.PHOENIX)
    return " (due " + dt.strftime("%A %-I:%M%p").lower() + ")"


def _format_context(open_items, hits):
    lines = []
    if open_items:
        lines.append("Open items the group is tracking:")
        for it in open_items:
            lines.append(f"  - [{it['kind']}] {it['who']}: "
                         f"{it['what']}{_due_label(it.get('due'))}")
    if hits:
        lines.append("Matching past messages:")
        for m in hits:
            lines.append(f"  - {m['author']}: {m['text']}")
    if not lines:
        lines.append("(no relevant memory)")
    return "\n".join(lines)


def _gather_context(message_text, memory, channel):
    open_items = memory.get_open_items(channel)
    hits, seen = [], set()
    for word in _keywords(message_text):
        for m in memory.search_messages(channel, word, limit=3):
            if m["id"] not in seen:
                seen.add(m["id"])
                hits.append(m)
    return open_items, hits[:5]


def _parse_json_object(raw_text):
    """Parse a strict-JSON object reply, tolerating a stray fence. None on fail."""
    if not raw_text:
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _answer_from_memory(message_text, memory, channel, client):
    """Compose a reply from memory, or None if memory can't answer.

    The model returns strict JSON: {"answerable": false} or
    {"answerable": true, "reply": "..."}. Parsing failures stay quiet.
    """
    open_items, hits = _gather_context(message_text, memory, channel)
    context = _format_context(open_items, hits)
    user = f"{context}\n\nQuestion from the chat:\n  {message_text}"
    data = _parse_json_object(_complete(client, ANSWER_SYSTEM, user))
    if not data or data.get("answerable") is not True:
        return None
    reply = data.get("reply")
    return reply if isinstance(reply, str) and reply.strip() else None


def _detect_conflict(message_text, memory, channel, client):
    """Return a one-line conflict callout, or None.

    The model returns strict JSON: {"conflict": false} or
    {"conflict": true, "line": "..."}. Prose, or any parse failure, stays
    quiet -- Elephant speaks only when conflict is true with a usable line.
    """
    decisions = [it for it in memory.get_open_items(channel)
                 if it["kind"] == "decision"]
    # Cheap pre-filter: no standing decisions, or nothing time-like to clash.
    if not decisions or not _mentions_time(message_text):
        return None
    lines = ["Open decisions:"]
    for it in decisions:
        lines.append(f"  - {it['who']}: {it['what']}{_due_label(it.get('due'))}")
    user = "\n".join(lines) + f"\n\nNew message:\n  {message_text}"
    data = _parse_json_object(_complete(client, CONFLICT_SYSTEM, user))
    if not data or data.get("conflict") is not True:
        return None
    line = data.get("line")
    return line if isinstance(line, str) and line.strip() else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _result(speak, reason, kind, reply):
    return {"speak": speak, "reason": reason, "kind": kind, "reply": reply}


def decide(message_text, author, now, memory, mentioned, *,
           channel=DEFAULT_CHANNEL, client=None):
    """Decide whether Elephant should speak about one incoming message.

    Returns a dict: speak (bool), reason (str), kind ("answer"/"followup"/
    "conflict"/"none"), reply (str or None). Records the message as seen,
    records a reply whenever Elephant speaks, and logs the decision (spoke or
    not) to the decision log.
    """
    res = _decide(message_text, author, now, memory, mentioned,
                  channel=channel, client=client)
    memory.log_decision(channel, message_text, res["speak"], res["kind"],
                        res["reason"])
    return res


def _decide(message_text, author, now, memory, mentioned, *,
            channel=DEFAULT_CHANNEL, client=None):
    memory.record_seen()
    text_lower = message_text.lower()

    # Case 1: directly addressed -> always answer (never cooldown-limited).
    if mentioned or "elephant" in text_lower:
        reply = _answer_from_memory(message_text, memory, channel, client)
        if reply is None:
            reply = "I don't have anything noted on that yet, but I'm listening!"
        memory.record_reply()
        return _result(True, "directly addressed", "answer", reply)

    # Case 3: the message contradicts a standing decision. Checked before the
    # question case, since a conflict is often phrased as one ("can we do 7
    # instead?"); the pre-filter + API keep genuine questions from matching.
    conflict = _detect_conflict(message_text, memory, channel, client)
    if conflict is not None:
        if not _cooldown_ok(channel, now):
            return _result(False, "conflict found but on cooldown", "none", None)
        _register_unprompted(channel, now)
        memory.record_reply()
        return _result(True, "contradicts an open decision", "conflict",
                       conflict)

    # Case 2: a question about the past that memory can answer.
    if _looks_like_question(message_text):
        reply = _answer_from_memory(message_text, memory, channel, client)
        if reply is not None:
            memory.record_reply()
            return _result(True, "question answerable from memory", "answer",
                           reply)
        # A question memory can't answer -> stay quiet rather than guess.
        return _result(False, "question, but memory has no answer", "none", None)

    # Default: nothing worth saying.
    return _result(False, "chatter / side conversation", "none", None)


def check_due(memory, now, *, channel=DEFAULT_CHANNEL):
    """Return friendly follow-up lines for past-due promises.

    One short line per overdue promise. Marks nothing done. Respects the same
    unprompted cooldown as conflicts; returns [] when on cooldown or nothing is
    due.

    Each follow-up is logged to the decision log with kind "followup", but
    record_seen/record_reply are deliberately NOT called: follow-ups are
    proactive, not responses to a message that was "seen", so counting them
    would distort the silence score.
    """
    due = memory.get_due_items(now)
    if not due:
        return []
    if not _cooldown_ok(channel, now):
        return []
    _register_unprompted(channel, now)
    lines = []
    for it in due:
        line = f"hey {it['who']}, did you get a chance to {it['what']}? no rush!"
        memory.log_decision(channel, line, True, "followup",
                            "promise past due")
        lines.append(line)
    return lines
