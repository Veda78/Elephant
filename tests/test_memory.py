"""Smoke tests for core.memory.MemoryStore using a temporary database."""

import os
import sys
import tempfile

# Make the project root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryStore


def main():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_elephant.db")
    store = MemoryStore(db_path)

    # -- add_message ------------------------------------------------------
    m1 = store.add_message("general", "alice", "I'll send the report tomorrow")
    m2 = store.add_message("general", "bob", "Let's go with Postgres then")
    m3 = store.add_message("random", "alice", "anyone up for lunch?")
    assert m1 and m2 and m3, "add_message should return truthy ids"
    assert m1 != m2 != m3, "message ids should be distinct"
    print("add_message            OK  ->", m1, m2, m3)

    # -- search_messages --------------------------------------------------
    hits = store.search_messages("general", "report", limit=5)
    assert len(hits) == 1 and hits[0]["id"] == m1, hits
    assert store.search_messages("general", "lunch", 5) == [], "channel filter"
    print("search_messages        OK  ->", hits[0]["text"])

    # -- add_item ---------------------------------------------------------
    past = 1000.0  # a due time well in the past
    future = 10_000_000_000.0  # year ~2286
    promise = store.add_item("promise", "alice", "send the report", past, m1)
    future_promise = store.add_item(
        "promise", "alice", "review PR", future, m1
    )
    decision = store.add_item(
        "decision", "bob", "use Postgres", None, m2
    )
    assert promise and future_promise and decision
    print("add_item               OK  ->", promise, future_promise, decision)

    # -- get_open_items ---------------------------------------------------
    open_general = store.get_open_items("general")
    assert len(open_general) == 3, open_general
    assert store.get_open_items("random") == [], "no items sourced in random"
    print("get_open_items         OK  ->", len(open_general), "open in general")

    # -- get_due_items ----------------------------------------------------
    due = store.get_due_items(now=2000.0)
    assert len(due) == 1 and due[0]["id"] == promise, due
    # The decision has no due time and must never surface as due.
    assert all(d["kind"] == "promise" for d in due)
    print("get_due_items          OK  ->", due[0]["what"])

    # -- mark_done --------------------------------------------------------
    store.mark_done(promise)
    assert store.get_due_items(now=2000.0) == [], "done item not due"
    assert len(store.get_open_items("general")) == 2, "one fewer open item"
    print("mark_done              OK")

    # -- stats / silence score -------------------------------------------
    assert store.get_silence_score() == 0.0, "no messages seen yet -> 0"
    for _ in range(4):
        store.record_seen()
    store.record_reply()
    # seen=4, replied=1 -> 3/4 unanswered = 75%
    score = store.get_silence_score()
    assert abs(score - 75.0) < 1e-9, score
    print("get_silence_score      OK  ->", f"{score:.1f}%")

    store.close()

    _test_dedup()

    print("\nAll tests passed.")


def _test_dedup():
    """add_item_if_new skips near-duplicate open promises and decisions."""
    store = MemoryStore(os.path.join(tempfile.mkdtemp(), "dedup.db"))
    mid = store.add_message("general", "priya", "I'll push the writeup")

    # First promise is created.
    id1, created1 = store.add_item_if_new(
        "promise", "priya", "push the writeup", 1000.0, mid, "general")
    assert created1 is True, "first promise should be created"

    # A near-duplicate from the same person (extra words, different due) is
    # skipped and maps back to the first item's id.
    id2, created2 = store.add_item_if_new(
        "promise", "priya", "push my part of the writeup", 9999.0, mid, "general")
    assert created2 is False, "similar promise should be skipped"
    assert id2 == id1, "duplicate maps to the existing item"

    # Same text but a DIFFERENT person is not a duplicate.
    id3, created3 = store.add_item_if_new(
        "promise", "jordan", "push the writeup", 1000.0, mid, "general")
    assert created3 is True, "different person is a separate promise"

    # A genuinely different promise from priya is kept.
    id4, created4 = store.add_item_if_new(
        "promise", "priya", "book the study room", 1000.0, mid, "general")
    assert created4 is True, "different task is a separate promise"

    assert len(store.get_open_items("general")) == 3, store.get_open_items("general")
    print("add_item_if_new (promise)  OK  -> 1 kept, 1 skipped, 2 distinct")

    # Decisions: similar "what" is a duplicate (who is ignored).
    d1, c1 = store.add_item_if_new(
        "decision", "group", "go out on mill ave", None, mid, "general")
    d2, c2 = store.add_item_if_new(
        "decision", "group", "go out on mill ave saturday", None, mid, "general")
    assert c1 is True and c2 is False, (c1, c2)
    assert d2 == d1, "duplicate decision maps to existing"
    d3, c3 = store.add_item_if_new(
        "decision", "group", "use overleaf for the report", None, mid, "general")
    assert c3 is True, "different decision is kept"
    decisions = [it for it in store.get_open_items("general")
                 if it["kind"] == "decision"]
    assert len(decisions) == 2, decisions
    print("add_item_if_new (decision) OK  -> 1 kept, 1 skipped, 1 distinct")
    store.close()


if __name__ == "__main__":
    main()
