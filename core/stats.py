"""Render Elephant's self-report card for a /stats command.

A small, plain-text summary of how talkative Elephant has been and why: how
much it saw, how often it spoke, its silence score, a breakdown of what kind of
replies it made, and the last few decisions with their reasons.
"""

# Reply kinds shown in the inline breakdown (in display order). "none" is a
# quiet decision, and "followup" is proactive (shown separately, since it is
# not counted in the silence score), so neither is listed here.
_REPLY_KINDS = ("answer", "conflict")


def format_stats(memory, channel):
    """Return a short plain-text stats card for `channel`."""
    counts = memory.get_seen_replied()
    seen, replied = counts["seen"], counts["replied"]
    score = memory.get_silence_score()
    kinds = memory.count_reply_kinds(channel)
    decisions = memory.get_decisions(channel, limit=5)

    breakdown = ", ".join(f"{k} {kinds.get(k, 0)}" for k in _REPLY_KINDS)

    lines = [
        f"Elephant stats - #{channel}",
        f"messages seen: {seen}",
        f"replies sent: {replied}",
        f"silence score: {score:.0f}% (stayed quiet)",
        f"replies by kind: {breakdown}",
        f"follow ups sent (not counted in the score): "
        f"{kinds.get('followup', 0)}",
        "",
        "last 5 decisions:",
    ]
    if decisions:
        for d in decisions:
            tag = "spoke" if d["speak"] else "quiet"
            lines.append(f"  {tag} - {d['reason']}")
    else:
        lines.append("  (none yet)")
    return "\n".join(lines)
