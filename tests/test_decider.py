"""Tests for core.decider.

Uses a fake Anthropic client (so no network, deterministic) and a temporary
MemoryStore per test. Covers the six required behaviors plus check_due.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryStore
from core import decider


NOW = 1792764000.0  # Fri 2026-10-23 07:00 America/Phoenix
CH = "general"


# -- fake Anthropic client -----------------------------------------------

class _Resp:
    def __init__(self, text):
        self.content = [
            type("Thinking", (), {"type": "thinking", "thinking": ""})(),
            type("Text", (), {"type": "text", "text": text})(),
        ]


class FakeClient:
    """Returns canned text, branching on which system prompt is used.

    `answer_text` is returned for answer prompts, `conflict_text` for conflict
    prompts. "NONE" means "memory can't answer" / "no conflict".
    """

    def __init__(self, answer_text="NONE", conflict_text="NONE"):
        self.answer_text = answer_text
        self.conflict_text = conflict_text
        self.n_answer = 0
        self.n_conflict = 0
        outer = self

        class _Messages:
            def create(self, *, system, messages, **kwargs):
                if "conflicts with a decision" in system:
                    outer.n_conflict += 1
                    return _Resp(outer.conflict_text)
                outer.n_answer += 1
                return _Resp(outer.answer_text)

        self.messages = _Messages()


# -- helpers --------------------------------------------------------------

def _fresh_memory():
    tmp = tempfile.mkdtemp()
    return MemoryStore(os.path.join(tmp, "decider.db"))


def _seed_meeting_decision(mem):
    """A standing decision: meet at Hayden Library Thursday at 6."""
    mid = mem.add_message(CH, "deshawn", "ok let's meet Thursday at 6 at Hayden")
    return mem.add_item("decision", "group",
                        "meet at Hayden Library Thursday at 6", None, mid)


# -- tests ----------------------------------------------------------------

def test_mention_gets_answer():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient(answer_text="We're meeting Thursday at 6 at Hayden 🙂")

    res = decider.decide("hey elephant, where are we meeting?", "maya", NOW,
                         mem, mentioned=True, channel=CH, client=client)
    assert res["speak"] is True, res
    assert res["kind"] == "answer", res
    assert res["reply"] == "We're meeting Thursday at 6 at Hayden 🙂", res
    assert res["reason"] == "directly addressed", res
    print(f"  mention  -> speak={res['speak']} kind={res['kind']} "
          f"reply={res['reply']!r}")


def test_mention_without_keyword_via_flag():
    """`mentioned=True` triggers case 1 even without the word 'elephant'."""
    decider.reset_cooldowns()
    mem = _fresh_memory()
    client = FakeClient(answer_text="NONE")  # memory empty -> fallback line
    res = decider.decide("you around?", "sam", NOW, mem, mentioned=True,
                         channel=CH, client=client)
    assert res["speak"] is True and res["kind"] == "answer", res
    assert res["reply"] is not None, res  # fallback, not None
    print(f"  @mention -> fallback reply={res['reply']!r}")


def test_question_answered_from_memory():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient(answer_text="You decided to meet Thursday at 6.")

    res = decider.decide("what did we decide about the meeting?", "priya", NOW,
                         mem, mentioned=False, channel=CH, client=client)
    assert res["speak"] is True and res["kind"] == "answer", res
    assert res["reason"] == "question answerable from memory", res
    assert client.n_answer == 1, "should have called the answer prompt once"
    print(f"  question -> {res['reply']!r}")


def test_question_memory_cannot_answer_stays_quiet():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    client = FakeClient(answer_text="NONE")  # model says memory has no answer
    res = decider.decide("who's bringing snacks?", "sam", NOW, mem,
                         mentioned=False, channel=CH, client=client)
    assert res["speak"] is False and res["kind"] == "none", res
    assert res["reason"] == "question, but memory has no answer", res
    print(f"  unanswerable question -> quiet ({res['reason']})")


def test_conflict_flagged():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient(
        conflict_text="heads up — you'd set Thursday at 6, this says 7.")

    res = decider.decide("actually can we do Thursday at 7 instead?", "maya",
                         NOW, mem, mentioned=False, channel=CH, client=client)
    assert res["speak"] is True and res["kind"] == "conflict", res
    assert res["reason"] == "contradicts an open decision", res
    assert client.n_conflict == 1, res
    print(f"  conflict -> {res['reply']!r}")


def test_chatter_stays_quiet():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient()  # all NONE
    res = decider.decide("lol that lecture was brutal today", "sam", NOW,
                         mem, mentioned=False, channel=CH, client=client)
    assert res["speak"] is False and res["kind"] == "none", res
    assert res["reason"] == "chatter / side conversation", res
    # Chatter must not reach the API at all (cheap path only).
    assert client.n_answer == 0 and client.n_conflict == 0, "API called on chatter"
    print(f"  chatter  -> quiet, no API call ({res['reason']})")


def test_cooldown_blocks_second_unprompted():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient(conflict_text="that clashes with the Thursday 6 plan.")

    first = decider.decide("actually let's do Thursday at 7", "maya", NOW, mem,
                           mentioned=False, channel=CH, client=client)
    assert first["speak"] is True and first["kind"] == "conflict", first

    # 60s later, another conflict -> blocked by the 10-minute cooldown.
    second = decider.decide("no wait, make it Thursday at 8", "maya", NOW + 60,
                            mem, mentioned=False, channel=CH, client=client)
    assert second["speak"] is False, second
    assert second["reason"] == "conflict found but on cooldown", second
    print(f"  cooldown -> 1st speaks, 2nd blocked ({second['reason']})")

    # A direct mention is never limited, even while on cooldown.
    addressed = decider.decide("elephant what's the plan?", "maya", NOW + 120,
                               mem, mentioned=True, channel=CH, client=client)
    assert addressed["speak"] is True, addressed
    print("  cooldown -> direct mention still answered")


def test_silence_score_changes():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    client = FakeClient(answer_text="We're set for Thursday at 6.")

    assert mem.get_silence_score() == 0.0

    # 1) chatter -> seen, no reply
    decider.decide("anyone free this weekend", "sam", NOW, mem,
                   mentioned=False, channel=CH, client=client)
    assert mem.get_silence_score() == 100.0, "1 seen, 0 replied"

    # 2) direct mention -> seen + reply
    decider.decide("elephant when are we meeting?", "maya", NOW, mem,
                   mentioned=True, channel=CH, client=client)
    # 2 seen, 1 replied -> 50% silent
    assert abs(mem.get_silence_score() - 50.0) < 1e-9, mem.get_silence_score()

    # 3) more chatter -> seen, no reply
    decider.decide("brb getting coffee", "sam", NOW, mem,
                   mentioned=False, channel=CH, client=client)
    # 3 seen, 1 replied -> 66.67% silent
    assert abs(mem.get_silence_score() - (2 / 3 * 100)) < 1e-9, \
        mem.get_silence_score()
    print(f"  silence score -> {mem.get_silence_score():.1f}% (3 seen, 1 reply)")


def test_check_due_followups():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    mid = mem.add_message(CH, "maya", "I'll email the TA by yesterday")
    item_id = mem.add_item("promise", "maya", "email the TA",
                           NOW - 3600, mid)  # due an hour ago

    lines = decider.check_due(mem, NOW, channel=CH)
    assert len(lines) == 1, lines
    assert "maya" in lines[0] and "email the TA" in lines[0], lines
    # Marks nothing done: the promise is still open.
    assert any(it["id"] == item_id for it in mem.get_open_items(CH)), "still open"
    print(f"  check_due -> {lines[0]!r}")

    # Second call within the cooldown window returns nothing.
    assert decider.check_due(mem, NOW + 60, channel=CH) == [], "cooldown"
    print("  check_due -> second call suppressed by cooldown")


def main():
    tests = [
        ("mention gets an answer", test_mention_gets_answer),
        ("mention via flag (no keyword)", test_mention_without_keyword_via_flag),
        ("question answered from memory", test_question_answered_from_memory),
        ("unanswerable question stays quiet",
         test_question_memory_cannot_answer_stays_quiet),
        ("conflict is flagged", test_conflict_flagged),
        ("chatter stays quiet", test_chatter_stays_quiet),
        ("cooldown blocks 2nd unprompted", test_cooldown_blocks_second_unprompted),
        ("silence score changes", test_silence_score_changes),
        ("check_due follow-ups", test_check_due_followups),
    ]
    for name, fn in tests:
        print(f"\n{name}")
        fn()
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
