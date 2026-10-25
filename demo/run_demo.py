"""Replay the scripted chat through Elephant's full pipeline -- no Discord.

For each message we: store it, run the extractor (saving promises/decisions),
run decide(), and print the chat with either Elephant's reply or a dim
"stayed quiet" line and the reason. Then time jumps, check_due() fires for
overdue promises, and the stats card prints.

Offline mode (DEMO_OFFLINE=1) uses a canned fake client -- no key, no network.
Otherwise the real Anthropic API is used (ANTHROPIC_API_KEY required).

Run:
    DEMO_OFFLINE=1 python demo/run_demo.py      # offline
    python demo/run_demo.py                      # live API
"""

import datetime
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.memory import MemoryStore
from core import decider, extractor, stats
from demo import chat_script
from demo.chat_script import PHOENIX, SCRIPT, CHANNEL


# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------
_USE_COLOR = (
    (os.environ.get("FORCE_COLOR") or sys.stdout.isatty())
    and not os.environ.get("NO_COLOR")
)


def _c(code):
    return code if _USE_COLOR else ""


RESET = _c("\033[0m")
DIM = _c("\033[2m")
BOLD = _c("\033[1m")
ELEPHANT = _c("\033[1m\033[38;5;110m")   # bold slate-blue
TIME = _c("\033[2m\033[38;5;245m")       # dim gray
NOTE = _c("\033[38;5;108m")              # soft green
_AUTHOR_COLORS = {
    "maya": _c("\033[38;5;211m"),     # pink
    "deshawn": _c("\033[38;5;75m"),   # blue
    "priya": _c("\033[38;5;179m"),    # gold
    "jordan": _c("\033[38;5;114m"),   # green
}


def _author_color(name):
    return _AUTHOR_COLORS.get(name, _c("\033[37m"))


def _clock(ts):
    dt = datetime.datetime.fromtimestamp(ts, PHOENIX)
    return dt.strftime("%a %I:%M%p").replace(" 0", " ").lower()


def _due_str(due):
    if not due:
        return "no due date"
    dt = datetime.datetime.fromtimestamp(due, PHOENIX)
    return "due " + dt.strftime("%a %I:%M%p").replace(" 0", " ").lower()


# ---------------------------------------------------------------------------
# Offline fake Anthropic client
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]


class FakeClient:
    """Canned client driven by the current message's annotations."""

    def __init__(self):
        self.current = {}
        outer = self

        class _Messages:
            def create(self, *, system, messages, **kwargs):
                if "quiet observer" in system:          # extractor (JSON array)
                    return _Resp(json.dumps(outer.current.get("extract", [])))
                if "conflicts with a decision" in system:  # conflict check
                    line = outer.current.get("conflict")
                    payload = ({"conflict": True, "line": line} if line
                               else {"conflict": False})
                    return _Resp(json.dumps(payload))
                reply = outer.current.get("answer")       # answer check
                payload = ({"answerable": True, "reply": reply} if reply
                           else {"answerable": False})
                return _Resp(json.dumps(payload))

        self.messages = _Messages()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main():
    offline = os.environ.get("DEMO_OFFLINE") == "1"
    client = FakeClient() if offline else None

    decider.reset_cooldowns()
    tmp = tempfile.mkdtemp()
    mem = MemoryStore(os.path.join(tmp, "demo.db"))

    mode = "OFFLINE (canned)" if offline else "LIVE API"
    print(f"\n{BOLD}🐘  Elephant demo{RESET}  {DIM}· #{CHANNEL} · {mode}{RESET}")
    print(f"{DIM}{'-' * 64}{RESET}\n")

    recent = []
    current_day = None
    for msg in SCRIPT:
        if offline:
            client.current = msg

        # Day banner.
        if msg["day"] != current_day:
            current_day = msg["day"]
            day_name = datetime.datetime.fromtimestamp(
                msg["ts"], PHOENIX).strftime("%A, %b %-d")
            print(f"{DIM}=== {day_name} ==={RESET}")

        author, text, now = msg["author"], msg["text"], msg["ts"]

        # 1) store, 2) extract + save items (skipping near-duplicates).
        mid = mem.add_message(CHANNEL, author, text)
        items = extractor.extract(text, author, now, recent[-6:], client=client)
        saved = []
        for it in items:
            _id, created = mem.add_item_if_new(
                it["kind"], it["who"], it["what"], it["due"], mid, CHANNEL)
            saved.append((it, created))

        # 3) decide.
        res = decider.decide(text, author, now, mem,
                             mentioned=msg.get("addressed", False),
                             channel=CHANNEL, client=client)

        # Print the chat line.
        acolor = _author_color(author)
        print(f"{TIME}{_clock(now):>14}{RESET}  "
              f"{acolor}{author:>7}{RESET}  {text}")

        # Dim note for anything Elephant quietly filed away (or skipped as dup).
        for it, created in saved:
            if created:
                print(f"{'':>14}  {NOTE}  · noted {it['kind']}: {it['who']} — "
                      f"{it['what']} ({_due_str(it['due'])}){RESET}")
            else:
                print(f"{'':>14}  {DIM}  · skipped duplicate {it['kind']}: "
                      f"{it['who']} — {it['what']}{RESET}")

        # Elephant's reaction.
        if res["speak"]:
            print(f"{'':>14}  {ELEPHANT}🐘 elephant{RESET}  {res['reply']}")
            print(f"{'':>14}  {DIM}({res['kind']}: {res['reason']}){RESET}")
        else:
            print(f"{'':>14}  {DIM}🐘 Elephant stayed quiet — "
                  f"{res['reason']}{RESET}")

        recent.append({"author": author, "text": text})

    # ---- time jump: overdue follow-ups ------------------------------------
    print(f"\n{DIM}{'-' * 64}{RESET}")
    print(f"{DIM}⏭  time passes… it's now {chat_script.CHECK_DUE_CLOCK}, "
          f"and some promises never landed.{RESET}\n")

    followups = decider.check_due(mem, chat_script.CHECK_DUE_TS, channel=CHANNEL)
    if followups:
        for line in followups:
            print(f"{ELEPHANT}🐘 elephant{RESET}  {line}")
    else:
        print(f"{DIM}(nothing overdue — or Elephant's on cooldown){RESET}")

    # ---- stats card -------------------------------------------------------
    print(f"\n{DIM}{'-' * 64}{RESET}")
    print(f"{BOLD}/stats{RESET}\n")
    card = stats.format_stats(mem, CHANNEL)
    for line in card.splitlines():
        if line.startswith("silence score:"):
            print(f"{BOLD}{NOTE}{line}{RESET}")
        else:
            print(line)
    print()

    mem.close()


if __name__ == "__main__":
    main()
