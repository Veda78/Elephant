"""Tests for core.extractor.

The logic tests use a fake Anthropic client that returns canned JSON, so they run
offline and deterministically. One live test actually calls the API and is
skipped when ANTHROPIC_API_KEY is not set.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import extractor


# A fixed "now": Monday, 2026-10-26 09:00 in America/Phoenix.
NOW = 1792764000.0


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


# -- 10 realistic ASU student messages + the JSON we expect the model to --
# -- return for each. A mix of real promises, real decisions, chatter.    --

MESSAGES = [
    # (author, text, canned_model_reply, expected_items)
    (
        "maya",
        "I'll send out the study guide for ECON 211 tonight",
        json.dumps([{"kind": "promise", "who": "maya",
                     "what": "send out the ECON 211 study guide", "due": None}]),
        [{"kind": "promise", "who": "maya",
          "what": "send out the ECON 211 study guide", "due": None}],
    ),
    (
        "deshawn",
        "ok let's meet Thursday at 6 at Hayden Library",
        json.dumps([{"kind": "decision", "who": "group",
                     "what": "meet at Hayden Library", "due": 1793059200}]),
        [{"kind": "decision", "who": "group",
          "what": "meet at Hayden Library", "due": 1793059200}],
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
        json.dumps([{"kind": "promise", "who": "sam",
                     "what": "Venmo back for the pizza", "due": 1793145600}]),
        [{"kind": "promise", "who": "sam",
          "what": "Venmo back for the pizza", "due": 1793145600}],
    ),
    (
        "maya",
        "maybe we should start the group project earlier this time",
        "[]",
        [],
    ),
    (
        "priya",
        "yeah Thursday at 6 works for me",
        json.dumps([{"kind": "decision", "who": "group",
                     "what": "meet at Hayden Library Thursday at 6",
                     "due": 1793059200}]),
        [{"kind": "decision", "who": "group",
          "what": "meet at Hayden Library Thursday at 6", "due": 1793059200}],
    ),
    (
        "sam",
        "is the library open till midnight during finals?",
        "[]",
        [],
    ),
    (
        "deshawn",
        "I'll book the study room for Saturday morning",
        json.dumps([{"kind": "promise", "who": "deshawn",
                     "what": "book the study room", "due": 1793311200}]),
        [{"kind": "promise", "who": "deshawn",
          "what": "book the study room", "due": 1793311200}],
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
        json.dumps([{"kind": "decision", "who": "group",
                     "what": "use Overleaf for the BIO 181 lab report",
                     "due": None}]),
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


def test_bad_json_is_silent():
    """A reply that isn't valid JSON must yield an empty list, not an error."""
    client = FakeClient({"anything": "sorry, I couldn't find anything useful!"})
    got = extractor.extract("anything goes here", "maya", NOW, [], client=client)
    assert got == [], got
    print("  malformed reply -> []  OK")


def test_fenced_json_is_tolerated():
    """Stray markdown fences around the JSON are stripped before parsing."""
    fenced = "```json\n[{\"kind\": \"promise\", \"who\": \"sam\", " \
             "\"what\": \"return the charger\", \"due\": null}]\n```"
    client = FakeClient({"charger": fenced})
    got = extractor.extract("bring the charger back", "sam", NOW, [], client=client)
    assert got == [{"kind": "promise", "who": "sam",
                    "what": "return the charger", "due": None}], got
    print("  fenced reply   -> parsed  OK")


def test_drops_malformed_entries():
    """Entries with a bad kind or missing fields are dropped, valid ones kept."""
    reply = json.dumps([
        {"kind": "promise", "who": "maya", "what": "email the TA", "due": None},
        {"kind": "rumor", "who": "sam", "what": "something", "due": None},
        {"kind": "decision", "who": "", "what": "no owner", "due": None},
        {"kind": "promise", "who": "deshawn", "what": "", "due": None},
    ])
    client = FakeClient({"mixed": reply})
    got = extractor.extract("mixed bag", "maya", NOW, [], client=client)
    assert got == [{"kind": "promise", "who": "maya",
                    "what": "email the TA", "due": None}], got
    print("  malformed entries dropped  OK")


def test_live_api():
    """Hits the real API. Skipped when ANTHROPIC_API_KEY is unset."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  SKIPPED (ANTHROPIC_API_KEY not set)")
        return
    got = extractor.extract(
        "I'll drop off the lab notebook at your dorm tomorrow at 5",
        "maya", NOW, [],
    )
    assert isinstance(got, list), got
    assert all(i["kind"] in ("promise", "decision") for i in got), got
    assert any(i["kind"] == "promise" for i in got), f"expected a promise: {got}"
    print(f"  live extract -> {got}")


def main():
    tests = [
        ("extract each of 10 messages", test_each_message),
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
