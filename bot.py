"""Discord adapter for Elephant.

Wires the Discord gateway to the existing core modules: every incoming message
is stored, mined for promises/decisions, and run through the decider, which
decides whether Elephant should speak. A !stats command posts the stats card,
and a background loop nudges about overdue promises.

Requires, in .env:
    DISCORD_TOKEN       the bot token
    ANTHROPIC_API_KEY   used by the extractor/decider (core modules)

The Message Content Intent must also be enabled for the bot in the Discord
Developer Portal (it is requested in code below, but Discord gates it there).
"""

import os
import time
from collections import defaultdict, deque

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from core.memory import MemoryStore
from core import decider, extractor, stats

load_dotenv()

DB_PATH = "elephant.db"
DUE_INTERVAL_SECONDS = 60
RECENT_WINDOW = 6  # messages of context passed to the extractor

memory = MemoryStore(DB_PATH)

intents = discord.Intents.default()
intents.message_content = True  # required to read message text
bot = commands.Bot(command_prefix="!", intents=intents)

# Per-channel recent-message context and the channel objects we've seen (so the
# background loop knows where to post follow-ups).
_recent = defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
_channels = {}


def _channel_key(channel):
    """A stable, readable key for a channel (its name, falling back to id)."""
    return getattr(channel, "name", None) or str(channel.id)


def _log(tag, text):
    print(f"[{tag}] {text}", flush=True)


@bot.event
async def on_ready():
    _log("ready", f"logged in as {bot.user} (id {bot.user.id})")
    if not due_loop.is_running():
        due_loop.start()


@bot.event
async def on_message(message):
    # Ignore anything from a bot, including ourselves (prevents loops).
    if message.author.bot:
        return

    # Let commands (e.g. !stats) run through the command system, and don't
    # treat them as ordinary chat.
    if message.content.startswith(bot.command_prefix):
        await bot.process_commands(message)
        return

    channel = message.channel
    ch = _channel_key(channel)
    _channels[ch] = channel

    author = message.author.display_name
    text = message.content
    now = message.created_at.timestamp()
    mentioned = bot.user in message.mentions

    # 1) store the message.
    mid = memory.add_message(ch, author, text)

    # 2) extract promises/decisions; save new ones (dedup against open items).
    recent = list(_recent[ch])
    items = extractor.extract(text, author, now, recent)
    for it in items:
        _id, created = memory.add_item_if_new(
            it["kind"], it["who"], it["what"], it["due"], mid, ch)
        if created:
            _log("note", f"{ch} | {it['kind']} {it['who']}: {it['what']}")

    # 3) decide whether to speak.
    res = decider.decide(text, author, now, memory, mentioned, channel=ch)
    _log("decide", f"{ch} | {author}: {text[:60]!r} -> "
                   f"{'SPEAK' if res['speak'] else 'quiet'} "
                   f"({res['kind']}: {res['reason']})")

    _recent[ch].append({"author": author, "text": text})

    # 4) reply in the channel if Elephant chose to speak.
    if res["speak"] and res["reply"]:
        await channel.send(res["reply"])


@bot.command(name="stats")
async def stats_command(ctx):
    """Post the stats card for this channel."""
    ch = _channel_key(ctx.channel)
    _channels[ch] = ctx.channel
    card = stats.format_stats(memory, ch)
    _log("stats", f"{ch} requested stats")
    await ctx.send(f"```\n{card}\n```")


@tasks.loop(seconds=DUE_INTERVAL_SECONDS)
async def due_loop():
    """Every minute, nudge about overdue promises in each channel we've seen."""
    now = time.time()
    for ch, channel in list(_channels.items()):
        for line in decider.check_due(memory, now, channel=ch):
            _log("check_due", f"{ch} -> {line}")
            await channel.send(line)


@due_loop.before_loop
async def _before_due_loop():
    await bot.wait_until_ready()


def main():
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_TOKEN is not set. Add it to .env (see .env.example).")
    bot.run(token)


if __name__ == "__main__":
    main()
