"""A scripted ASU group chat for the Elephant demo (no Discord, no network).

Four CSE 310 students -- maya, deshawn, priya, jordan -- juggle a graph project
and weekend plans over three days (Mon-Wed), then time jumps to Thursday
afternoon so overdue promises surface.

Each message carries a simulated timestamp plus optional annotations used only
by the OFFLINE fake client:
  - "extract":  what the extractor should return (a list of
                {kind, who, what, due_phrase}); omitted means nothing.
  - "answer":   canned reply text for a question/mention (default "NONE").
  - "conflict": canned reply text for a conflict check (default "NONE").
  - "addressed": True when Elephant is spoken to by name.
In online mode these are ignored and the real model decides.
"""

import datetime
import zoneinfo

PHOENIX = zoneinfo.ZoneInfo("America/Phoenix")
CHANNEL = "cse310-squad"

# Day 0 is Monday, 2026-10-19 (Tempe). Day 2 is Wednesday, day 3 Thursday.
BASE_DATE = datetime.date(2026, 10, 19)


def _ts(day, hhmm):
    hour, minute = (int(x) for x in hhmm.split(":"))
    d = BASE_DATE + datetime.timedelta(days=day)
    return datetime.datetime(
        d.year, d.month, d.day, hour, minute, tzinfo=PHOENIX
    ).timestamp()


# Raw script: (day, time, author, text, annotations)
_RAW = [
    # ---- Day 0: Monday -----------------------------------------------------
    (0, "08:14", "maya", "ugh CSE 310 project drops today and I haven't even opened Canvas", {}),
    (0, "08:16", "jordan", "same lol I slept through my 7:30 alarm AGAIN", {}),
    (0, "08:20", "deshawn", "ok so it's the graph assignment, due next week", {}),
    (0, "08:22", "priya", "wait graphs already?? we JUST did heaps 😭", {}),
    (0, "08:25", "maya", "we should meet and split this up before it snowballs", {}),
    (0, "08:31", "deshawn", "ok let's meet Thursday at 6 at Hayden Library",
     {"extract": [{"kind": "decision", "who": "group",
                   "what": "meet at Hayden Library", "due_phrase": "Thursday at 6"}]}),
    (0, "08:32", "jordan", "bet. top floor by the windows", {}),
    (0, "08:33", "priya", "works for me", {}),
    (0, "08:40", "maya", "I'll put together the slides outline and send it tonight",
     {"extract": [{"kind": "promise", "who": "maya",
                   "what": "send the slides outline", "due_phrase": "tonight"}]}),
    (0, "08:41", "jordan", "mvp 🙌", {}),
    (0, "09:02", "deshawn", "I can book the study room for Wednesday as a backup spot",
     {"extract": [{"kind": "promise", "who": "deshawn",
                   "what": "book the study room", "due_phrase": "Wednesday"}]}),
    (0, "09:10", "priya", "someone explain why Canvas logged me out AGAIN", {}),
    (0, "12:30", "maya", "elephant when are we meeting?",
     {"addressed": True,
      "answer": "You're meeting Thursday at 6 at Hayden Library 🙂"}),
    (0, "12:45", "jordan", "lunch at the MU? the chick-fil-a line better be short", {}),
    (0, "12:47", "priya", "it never is 💀", {}),

    # ---- Day 1: Tuesday ----------------------------------------------------
    (1, "09:05", "deshawn", "pushed the skeleton code to the repo, take a look", {}),
    (1, "09:30", "maya", "nice. I'll take the BFS/DFS part", {}),
    (1, "10:00", "priya", "I'll push my part of the writeup by tomorrow",
     {"extract": [{"kind": "promise", "who": "priya",
                   "what": "push the writeup", "due_phrase": "tomorrow"}]}),
    (1, "10:05", "jordan", "has anyone seen the new Sparky meme on the class discord 💀💀", {}),
    (1, "10:06", "maya", "the one with the dining hall food?? dead 😂", {}),
    (1, "10:30", "deshawn", "wait who was bringing the poster?",
     {"answer": "I don't see anyone who's claimed the poster yet -- still open."}),
    (1, "10:45", "jordan", "not me, I did the slides last time", {}),
    (1, "10:46", "jordan", "fine fine I'll grab the poster board tomorrow after my lab",
     {"extract": [{"kind": "promise", "who": "jordan",
                   "what": "grab the poster board", "due_phrase": "tomorrow"}]}),
    (1, "11:00", "maya", "legend", {}),
    (1, "13:00", "priya", "ok non-project: what are we doing this weekend", {}),
    (1, "13:05", "deshawn", "Mill Ave? I haven't been since the semester started", {}),
    (1, "13:06", "jordan", "let's do Saturday night on Mill Ave",
     {"extract": [{"kind": "decision", "who": "group",
                   "what": "go out on Mill Ave", "due_phrase": "Saturday night"}]}),
    (1, "13:10", "maya", "down. Sun Devil Fitness first so I can pretend I'm balanced 😂", {}),
    (1, "13:12", "priya", "lol the gym at 5pm is a zoo", {}),
    (1, "13:15", "deshawn", "facts", {}),

    # ---- Day 2: Wednesday --------------------------------------------------
    (2, "08:00", "maya", "morning. reminder we meet Thursday at 6", {}),
    (2, "08:30", "jordan", "my brain is not awake for graphs rn", {}),
    (2, "09:00", "priya", "did the study room get booked? just wondering", {}),
    (2, "09:05", "deshawn", "yep booked it, Hayden's still primary tho", {}),
    (2, "09:40", "jordan", "the MU smoothie place saved my life this morning", {}),
    (2, "11:00", "maya", "actually can we push Thursday to 7 instead of 6? I have a lab till 6",
     {"conflict": "heads up -- you'd agreed on Thursday at 6, and this moves it to 7. "
                  "want me to update it?"}),
    (2, "11:02", "jordan", "7 works for me honestly", {}),
    (2, "11:03", "priya", "same, 6 is tight", {}),
    (2, "11:05", "deshawn", "ok 7 it is", {}),
    (2, "12:20", "jordan", "sparky showed up at my 12pm lecture lol peak Tempe", {}),
    (2, "15:00", "maya", "elephant what did we decide for the weekend?",
     {"addressed": True,
      "answer": "You decided to go out on Mill Ave on Saturday night 🎉"}),
    (2, "15:30", "jordan", "poster board update: acquired. it's huge lol", {}),
    (2, "16:00", "priya", "iconic", {}),
    (2, "20:00", "maya", "I'll finish the README tonight so we can submit early",
     {"extract": [{"kind": "promise", "who": "maya",
                   "what": "finish the README", "due_phrase": "tonight"}]}),
    (2, "20:30", "deshawn", "you're carrying fr", {}),
    (2, "21:00", "jordan", "ok sleeping, graphs gave me a headache 😮‍💨", {}),
    (2, "21:05", "priya", "night nerds 🫡", {}),
]

SCRIPT = [
    {"ts": _ts(day, hhmm), "day": day, "clock": hhmm,
     "author": author, "text": text, **ann}
    for (day, hhmm, author, text, ann) in _RAW
]

# Time jump: Thursday afternoon, before the 7pm meeting -- several promises
# (slides, study room, writeup, poster, README) are now overdue.
CHECK_DUE_TS = _ts(3, "15:00")
CHECK_DUE_CLOCK = "Thursday 3:00pm"
