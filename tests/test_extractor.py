"""Tests for core.extractor.

The logic tests use a fake Anthropic client that returns canned JSON, so they run
offline and deterministically. One live test actually calls the API and is
skipped when ANTHROPIC_API_KEY is not set.

The model now returns a relative time phrase as `due_phrase`; the extractor
resolves it to a `due` Unix timestamp in Python (dateparser, America/Phoenix,
preferring future dates). All expected timestamps below were computed against
the fixed NOW anchor.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import extractor


# A fixed "now": Friday, 2026-10-23 07:00 in America/Phoenix (MST, no DST).
NOW = 1792764000.0

# Resolved timestamps for the phrases used below, relative to NOW.
TONIGHT = 1792814400         # Fri 2026-10-23 21:00  (night = 9pm)
TOMORROW_AT_5 = 1792886400   # Sat 2026-10-24 17:00  ("5" -> pm)
FRIDAY_TODAY = 1792800000    # Fri 2026-10-23 17:00  (today, default 5pm)
THURSDAY_AT_6 = 1793322000   # Thu 2026-10-29 18:00  (next Thu, "6" -> pm)
NEXT_FRIDAY = 1793404800     # Fri 2026-10-30 17:00  (following week, 5pm)
SATURDAY_MORNING = 1792857600  # Sat 2026-10-24 09:00  (morning = 9am)
NEXT_WEEK = 1793368800       # Fri 2026-10-30 07:00  (dateparser fallback)


# -- fake Anthropic client -----------------------------------------------

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text):
        self.content = [
            # A thinking block with empty text, as Haiku 5.5 returns by
            # default, to prove extract() skips non-text blocks.
            type("Thinking", (), {"type": "thinking", "thinking": ""})(),
            _Block(text),
        ]


class _Messages:
    def __init__(self, replies):
        self._replies = replies

    def create(self, *, model, messages, **kwargs):
        user_text = messages[0]["content"]
        # Match only against the new message, not the recent-context block
        # (earlier messages reappear there and would collide).
        new_part = user_text.split("New message to analyze:", 1)[-1]
        for needle, reply in self._replies.items():
            if needle in new_part:
                return _Response(reply)
        return _Response("[]")


class FakeClient:
    """Stands in for anthropic.Anthropic; returns canned JSON per message."""

    def __init__(self, replies):
        self.messages = _Messages(replies)


def _reply(kind, who, what, due_phrase):
    return json.dumps([{"kind": kind, "who": who, "what": what,
                        "due_phrase": due_phrase}])


# -- 10 realistic ASU student messages -----------------------------------
# Each: (author, text, canned_model_reply, expected_items). The model emits a
# due_phrase; the extractor resolves it to the `due` timestamp we expect.

MESSAGES = [
    (
        "maya",
        "I'll send out the study guide for ECON 211 tonight",
        _reply("promise", "maya", "send out the ECON 211 study guide", "tonight"),
        [{"kind": "promise", "who": "maya",
          "what": "send out the ECON 211 study guide", "due": TONIGHT}],
    ),
    (
        "deshawn",
        "ok let's meet at Hayden Library, I'll be there tomorrow at 5",
        _reply("decision", "group", "meet at Hayden Library", "tomorrow at 5"),
        [{"kind": "decision", "who": "group",
          "what": "meet at Hayden Library", "due": TOMORROW_AT_5}],
    ),
    (
        "priya",
        "anyone else dying in CSE 310 rn 😭",
        "[]",
        [],
    ),
    (
        "sam",
        "I can Venmo you back for the pizza on Friday",
        _reply("promise", "sam", "Venmo back for the pizza", "Friday"),
        [{"kind": "promise", "who": "sam",
          "what": "Venmo back for the pizza", "due": FRIDAY_TODAY}],
    ),
    (
        "maya",
        "maybe we should start the group project earlier this time",
        "[]",
        [],
    ),
    (
        "priya",
        "yeah tomorrow at 5 works for me",
        _reply("decision", "group", "meet at Hayden Library", "tomorrow at 5"),
        [{"kind": "decision", "who": "group",
          "what": "meet at Hayden Library", "due": TOMORROW_AT_5}],
    ),
    (
        "sam",
        "is the library open till midnight during finals?",
        "[]",
        [],
    ),
    (
        "deshawn",
        "I'll book the study room, should be done by next week",
        _reply("promise", "deshawn", "book the study room", "by next week"),
        [{"kind": "promise", "who": "deshawn",
          "what": "book the study room", "due": NEXT_WEEK}],
    ),
    (
        "sam",
        "lol did you see the sun devils game last night",
        "[]",
        [],
    ),
    (
        "maya",
        "decided, we're using Overleaf for the BIO 181 lab report",
        _reply("decision", "group", "use Overleaf for the BIO 181 lab report",
               None),
        [{"kind": "decision", "who": "group",
          "what": "use Overleaf for the BIO 181 lab report", "due": None}],
    ),
]


def test_each_message():
    replies = {text: reply for (_, text, reply, _) in MESSAGES}
    client = FakeClient(replies)
    recent = []
    for author, text, _, expected in MESSAGES:
        got = extractor.extract(text, author, NOW, recent, client=client)
        assert got == expected, f"\n  msg: {text!r}\n  got: {got}\n  exp: {expected}"
        recent.append({"author": author, "text": text})
        label = expected[0]["kind"] if expected else "silence"
        print(f"  [{label:8}] {text[:48]}")
    promises = sum(1 for _, _, _, e in MESSAGES if e and e[0]["kind"] == "promise")
    decisions = sum(1 for _, _, _, e in MESSAGES if e and e[0]["kind"] == "decision")
    silent = sum(1 for _, _, _, e in MESSAGES if not e)
    print(f"  -> {promises} promises, {decisions} decisions, {silent} silent")


def test_due_resolution():
    """Relative phrases resolve to the expected timestamps; junk -> None.

    NOW is Friday 07:00, so "Friday" is a weekday that is *today* (its default
    5pm is still ahead, so it stays today).
    """
    cases = [
        ("Thursday at 6", THURSDAY_AT_6),     # weekday + bare hour -> next Thu 6pm
        ("next Friday", NEXT_FRIDAY),         # "next" -> following week, 5pm
        ("Saturday morning", SATURDAY_MORNING),  # weekday + part of day
        ("Friday at 5pm", FRIDAY_TODAY),      # weekday today, explicit pm
        ("tomorrow at 5", TOMORROW_AT_5),     # tomorrow, bare "5" -> pm
        ("tonight", TONIGHT),                 # today at night (9pm)
        ("Friday", FRIDAY_TODAY),             # weekday that is today -> today 5pm
        ("by next week", NEXT_WEEK),          # not a weekday form -> dateparser
        (None, None),
        ("", None),
        ("sometime soon", None),              # neither path resolves it -> None
    ]
    for phrase, expected in cases:
        got = extractor._resolve_due(phrase, NOW)
        assert got == expected, f"{phrase!r}: got {got}, expected {expected}"
        print(f"  {str(phrase)!r:16} -> {got}")


def test_bad_json_is_silent():
    """A reply that isn't valid JSON must yield an empty list, not an error."""
    client = FakeClient({"anything": "sorry, I couldn't find anything useful!"})
    got = extractor.extract("anything goes here", "maya", NOW, [], client=client)
    assert got == [], got
    print("  malformed reply -> []  OK")


def test_fenced_json_is_tolerated():
    """Stray markdown fences around the JSON are stripped before parsing."""
    fenced = "```json\n[{\"kind\": \"promise\", \"who\": \"sam\", " \
             "\"what\": \"return the charger\", \"due_phrase\": null}]\n```"
    client = FakeClient({"charger": fenced})
    got = extractor.extract("bring the charger back", "sam", NOW, [], client=client)
    assert got == [{"kind": "promise", "who": "sam",
                    "what": "return the charger", "due": None}], got
    print("  fenced reply   -> parsed  OK")


def test_drops_malformed_entries():
    """Entries with a bad kind or missing fields are dropped, valid ones kept."""
    reply = json.dumps([
        {"kind": "promise", "who": "maya", "what": "email the TA",
         "due_phrase": "tonight"},
        {"kind": "rumor", "who": "sam", "what": "something", "due_phrase": None},
        {"kind": "decision", "who": "", "what": "no owner", "due_phrase": None},
        {"kind": "promise", "who": "deshawn", "what": "", "due_phrase": None},
    ])
    client = FakeClient({"mixed": reply})
    got = extractor.extract("mixed bag", "maya", NOW, [], client=client)
    assert got == [{"kind": "promise", "who": "maya",
                    "what": "email the TA", "due": TONIGHT}], got
    print("  malformed entries dropped  OK")


def test_live_api():
    """Hits the real API. Skipped when ANTHROPIC_API_KEY is unset."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  SKIPPED (ANTHROPIC_API_KEY not set)")
        return
    got = extractor.extract(
        "I'll drop off the lab notebook at your dorm tomorrow at 5pm",
        "maya", NOW, [],
    )
    assert isinstance(got, list), got
    assert all(i["kind"] in ("promise", "decision") for i in got), got
    assert all(i["due"] is None or isinstance(i["due"], int) for i in got), got
    assert any(i["kind"] == "promise" for i in got), f"expected a promise: {got}"
    print(f"  live extract -> {got}")


def main():
    tests = [
        ("extract each of 10 messages", test_each_message),
        ("relative time resolution", test_due_resolution),
        ("malformed JSON is silent", test_bad_json_is_silent),
        ("fenced JSON tolerated", test_fenced_json_is_tolerated),
        ("malformed entries dropped", test_drops_malformed_entries),
        ("live API call", test_live_api),
    ]
    for name, fn in tests:
        print(f"\n{name}")
        fn()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
