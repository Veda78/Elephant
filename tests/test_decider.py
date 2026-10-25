"""Tests for core.decider.

Uses a fake Anthropic client (so no network, deterministic) and a temporary
MemoryStore per test. Covers the six required behaviors plus check_due.
"""

import json
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
    """Fake client emitting the strict JSON the decider now expects.

    `answer` is the reply string (None -> {"answerable": false}); `conflict`
    is the callout line (None -> {"conflict": false}). `raw_answer`/
    `raw_conflict` inject arbitrary raw text instead, to test non-JSON replies.
    """

    def __init__(self, answer=None, conflict=None, *,
                 raw_answer=None, raw_conflict=None):
        self.answer = answer
        self.conflict = conflict
        self.raw_answer = raw_answer
        self.raw_conflict = raw_conflict
        self.n_answer = 0
        self.n_conflict = 0
        outer = self

        class _Messages:
            def create(self, *, system, messages, **kwargs):
                if "conflicts with a decision" in system:
                    outer.n_conflict += 1
                    if outer.raw_conflict is not None:
                        return _Resp(outer.raw_conflict)
                    if outer.conflict is None:
                        return _Resp('{"conflict": false}')
                    return _Resp(json.dumps(
                        {"conflict": True, "line": outer.conflict}))
                outer.n_answer += 1
                if outer.raw_answer is not None:
                    return _Resp(outer.raw_answer)
                if outer.answer is None:
                    return _Resp('{"answerable": false}')
                return _Resp(json.dumps(
                    {"answerable": True, "reply": outer.answer}))

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
    client = FakeClient(answer="We're meeting Thursday at 6 at Hayden 🙂")

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
    client = FakeClient()  # not answerable -> case 1 fallback line
    res = decider.decide("you around?", "sam", NOW, mem, mentioned=True,
                         channel=CH, client=client)
    assert res["speak"] is True and res["kind"] == "answer", res
    assert res["reply"] is not None, res  # fallback, not None
    print(f"  @mention -> fallback reply={res['reply']!r}")


def test_question_answered_from_memory():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient(answer="You decided to meet Thursday at 6.")

    res = decider.decide("what did we decide about the meeting?", "priya", NOW,
                         mem, mentioned=False, channel=CH, client=client)
    assert res["speak"] is True and res["kind"] == "answer", res
    assert res["reason"] == "question answerable from memory", res
    assert client.n_answer == 1, "should have called the answer prompt once"
    print(f"  question -> {res['reply']!r}")


def test_question_memory_cannot_answer_stays_quiet():
    decider.reset_cooldowns()
    mem = _fresh_memory()
    client = FakeClient()  # model says memory has no answer
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
        conflict="heads up — you'd set Thursday at 6, this says 7.")

    res = decider.decide("actually can we do Thursday at 7 instead?", "maya",
                         NOW, mem, mentioned=False, channel=CH, client=client)
    assert res["speak"] is True and res["kind"] == "conflict", res
    assert res["reason"] == "contradicts an open decision", res
    assert client.n_conflict == 1, res
    print(f"  conflict -> {res['reply']!r}")


def test_separate_plan_not_flagged():
    """A separate plan on a different day/place is not a conflict.

    The message mentions a time (so the cheap pre-filter sends it to the
    conflict check), but the model returns {"conflict": false}, and the decider
    must then stay quiet rather than fabricate a conflict.
    """
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)  # standing: meet Thursday at 6 to study
    client = FakeClient()  # conflict=None -> {"conflict": false}

    res = decider.decide("separately, let's go out Saturday night on Mill Ave",
                         "jordan", NOW, mem, mentioned=False, channel=CH,
                         client=client)
    assert res["speak"] is False and res["kind"] == "none", res
    assert res["reason"] == "chatter / side conversation", res
    assert client.n_conflict == 1, "conflict check ran but returned false"
    print(f"  separate plan -> quiet (conflict check ran, said false)")


def test_conflict_prose_stays_quiet():
    """If the model writes prose instead of JSON, Elephant stays quiet.

    This is the exact failure we saw live: the model 'explained' that the
    message was not a conflict instead of returning {"conflict": false}.
    """
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    prose = ("The new message suggests Wednesday as a backup spot, which "
             "doesn't change the Thursday 6pm meetup, so it's a separate "
             "suggestion rather than a conflict.")
    client = FakeClient(raw_conflict=prose)

    res = decider.decide("I can book the study room for Wednesday as a backup",
                         "deshawn", NOW, mem, mentioned=False, channel=CH,
                         client=client)
    assert res["speak"] is False and res["kind"] == "none", res
    assert client.n_conflict == 1, "conflict check ran"
    print("  conflict prose (non-JSON) -> stayed quiet OK")


def test_malformed_answer_stays_quiet():
    """A non-JSON answer reply leaves a (non-addressed) question quiet."""
    decider.reset_cooldowns()
    mem = _fresh_memory()
    _seed_meeting_decision(mem)
    client = FakeClient(raw_answer="Sure! I think it's probably Thursday?")
    res = decider.decide("when are we meeting again?", "priya", NOW, mem,
                         mentioned=False, channel=CH, client=client)
    assert res["speak"] is False and res["kind"] == "none", res
    assert res["reason"] == "question, but memory has no answer", res
    print("  malformed answer JSON -> stayed quiet OK")


def test_conflict_prompt_tightening_present():
    """Regression guard: the conflict prompt scopes conflicts to same event."""
    cp = decider.CONFLICT_SYSTEM
    assert "SAME event or SAME time slot" in cp, "same-event rule"
    assert "NOT a conflict" in cp, "separate-plan exclusion"
    print("  tightened conflict rule present in prompt  OK")


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
    client = FakeClient(conflict="that clashes with the Thursday 6 plan.")

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
    client = FakeClient(answer="We're set for Thursday at 6.")

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
        ("separate plan not flagged", test_separate_plan_not_flagged),
        ("conflict prose stays quiet", test_conflict_prose_stays_quiet),
        ("malformed answer stays quiet", test_malformed_answer_stays_quiet),
        ("conflict prompt tightening present",
         test_conflict_prompt_tightening_present),
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
