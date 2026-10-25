"""Tests for core.stats.format_stats.

Runs a handful of messages through decider.decide() against a temporary memory
store and a fake API client, then checks the rendered card.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryStore
from core import decider, stats


NOW = 1792764000.0
CH = "general"


# -- fake Anthropic client (same shape as test_decider) ------------------

class _Resp:
    def __init__(self, text):
        self.content = [
            type("Thinking", (), {"type": "thinking", "thinking": ""})(),
            type("Text", (), {"type": "text", "text": text})(),
        ]


class FakeClient:
    """Emits the strict JSON the decider expects.

    `answer` is a reply string (None -> {"answerable": false}); `conflict` is a
    callout line (None -> {"conflict": false}).
    """

    def __init__(self, answer=None, conflict=None):
        self.answer = answer
        self.conflict = conflict
        outer = self

        class _Messages:
            def create(self, *, system, messages, **kwargs):
                if "conflicts with a decision" in system:
                    if outer.conflict is None:
                        return _Resp('{"conflict": false}')
                    return _Resp(json.dumps(
                        {"conflict": True, "line": outer.conflict}))
                if outer.answer is None:
                    return _Resp('{"answerable": false}')
                return _Resp(json.dumps(
                    {"answerable": True, "reply": outer.answer}))

        self.messages = _Messages()


def _fresh_memory():
    tmp = tempfile.mkdtemp()
    return MemoryStore(os.path.join(tmp, "stats.db"))


def test_stats_card():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    # A standing decision gives the conflict/answer paths something to work on.
    mid = mem.add_message(CH, "deshawn", "ok let's meet Thursday at 6 at Hayden")
    mem.add_item("decision", "group", "meet at Hayden Thursday at 6", None, mid)

    client = FakeClient(
        answer="We're meeting Thursday at 6 at Hayden.",
        conflict="heads up - you'd set Thursday at 6, this says 7.",
    )

    # Five messages: quiet, answer, answer, conflict, quiet.
    script = [
        ("sam", "lol that lecture was wild", False),       # quiet  (chatter)
        ("maya", "elephant what's the plan?", True),        # spoke  (answer)
        ("priya", "what did we decide about the meeting?", False),  # spoke
        ("maya", "actually let's do Thursday at 7", False), # spoke  (conflict)
        ("sam", "brb grabbing coffee", False),              # quiet  (chatter)
    ]
    for author, text, mentioned in script:
        decider.decide(text, author, NOW, mem, mentioned, channel=CH,
                       client=client)

    card = stats.format_stats(mem, CH)
    print("\n" + card + "\n")

    # Counts: 5 seen, 3 replies (mention + question + conflict).
    assert "messages seen: 5" in card, card
    assert "replies sent: 3" in card, card
    # Silence score: (5 - 3) / 5 = 40%.
    assert "silence score: 40%" in card, card
    # Kind breakdown: 2 answers, 1 conflict (followups shown separately).
    assert "answer 2" in card, card
    assert "conflict 1" in card, card
    assert "follow ups sent (not counted in the score): 0" in card, card

    # Recent decisions: exactly 5 lines, newest first.
    decision_lines = [ln for ln in card.splitlines()
                      if ln.startswith("  spoke") or ln.startswith("  quiet")]
    assert len(decision_lines) == 5, decision_lines
    assert decision_lines[0] == "  quiet - chatter / side conversation", \
        decision_lines[0]
    assert decision_lines[1] == "  spoke - contradicts an open decision", \
        decision_lines[1]
    assert decision_lines[2] == "  spoke - question answerable from memory", \
        decision_lines[2]
    assert decision_lines[3] == "  spoke - directly addressed", decision_lines[3]
    assert decision_lines[4] == "  quiet - chatter / side conversation", \
        decision_lines[4]
    print("  counts, score, kind breakdown, and recent decisions all OK")


def test_followup_logged_but_not_scored():
    """A follow-up appears on the card but leaves seen/replied/score untouched."""
    decider.reset_cooldowns()
    mem = _fresh_memory()
    # A promise that came due an hour ago.
    mid = mem.add_message(CH, "maya", "I'll email the TA")
    mem.add_item("promise", "maya", "email the TA", NOW - 3600, mid)

    # One chatter message establishes a baseline: 1 seen, 0 replied, 100%.
    decider.decide("lol anyway", "sam", NOW, mem, mentioned=False,
                   channel=CH, client=FakeClient())
    before = mem.get_seen_replied()
    score_before = mem.get_silence_score()
    assert before == {"seen": 1, "replied": 0} and score_before == 100.0, before

    # The follow-up is emitted and logged...
    lines = decider.check_due(mem, NOW, channel=CH)
    assert len(lines) == 1 and "maya" in lines[0], lines

    # ...but seen, replied, and the silence score are unchanged.
    assert mem.get_seen_replied() == before, mem.get_seen_replied()
    assert mem.get_silence_score() == score_before, mem.get_silence_score()

    card = stats.format_stats(mem, CH)
    print("\n" + card + "\n")
    assert "messages seen: 1" in card, card
    assert "replies sent: 0" in card, card
    assert "silence score: 100%" in card, card
    assert "follow ups sent (not counted in the score): 1" in card, card
    # The follow-up is NOT in the recent-decisions list (that list is for
    # message-driven decisions only); it lives on the counter line instead.
    assert "promise past due" not in card.split("last 5 decisions:")[1], card
    decision_lines = [ln for ln in card.splitlines()
                      if ln.startswith("  spoke") or ln.startswith("  quiet")]
    assert decision_lines == ["  quiet - chatter / side conversation"], \
        decision_lines
    print("  follow-up counted but kept out of recent decisions OK")


def test_recent_list_skips_followups_to_find_messages():
    """With follow-ups interleaved, the list still shows 5 message decisions."""
    mem = _fresh_memory()
    # 6 message decisions with a couple of follow-ups mixed in between.
    mem.log_decision(CH, "m1", False, "none", "chatter / side conversation")
    mem.log_decision(CH, "m2", True, "answer", "directly addressed")
    mem.log_decision(CH, "f1", True, "followup", "promise past due")
    mem.log_decision(CH, "m3", False, "none", "chatter / side conversation")
    mem.log_decision(CH, "m4", True, "conflict", "contradicts an open decision")
    mem.log_decision(CH, "f2", True, "followup", "promise past due")
    mem.log_decision(CH, "m5", False, "none", "question, but memory has no answer")
    mem.log_decision(CH, "m6", True, "answer", "directly addressed")

    card = stats.format_stats(mem, CH)
    body = card.split("last 5 decisions:")[1]
    assert "promise past due" not in body, body
    lines = [ln for ln in body.splitlines()
             if ln.startswith("  spoke") or ln.startswith("  quiet")]
    # The 5 newest message decisions (m6..m2), newest first.
    assert lines == [
        "  spoke - directly addressed",                 # m6
        "  quiet - question, but memory has no answer",  # m5
        "  spoke - contradicts an open decision",        # m4
        "  quiet - chatter / side conversation",         # m3
        "  spoke - directly addressed",                  # m2
    ], lines
    print("  recent list skips follow-ups to show 5 message decisions OK")


def test_empty_card():
    """A fresh store renders a sane, empty card."""
    mem = _fresh_memory()
    card = stats.format_stats(mem, CH)
    print("\n" + card + "\n")
    assert "messages seen: 0" in card
    assert "replies sent: 0" in card
    assert "silence score: 0%" in card
    assert "(none yet)" in card
    print("  empty card OK")


def test_message_text_truncated_to_120():
    """log_decision stores only the first 120 characters of the message."""
    mem = _fresh_memory()
    long_text = "x" * 200
    mem.log_decision(CH, long_text, False, "none", "chatter / side conversation")
    stored = mem.get_decisions(CH, limit=1)[0]["message_text"]
    assert len(stored) == 120, len(stored)
    print(f"  message_text truncated to {len(stored)} chars OK")


def main():
    for name, fn in [
        ("stats card", test_stats_card),
        ("follow-up logged but not scored", test_followup_logged_but_not_scored),
        ("recent list skips follow-ups", test_recent_list_skips_followups_to_find_messages),
        ("empty card", test_empty_card),
        ("message truncation", test_message_text_truncated_to_120),
    ]:
        print(f"\n{name}")
        fn()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
