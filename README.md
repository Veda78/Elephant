# Elephant

A Discord bot that remembers.

## Run the bot

1. **Install dependencies** (into a virtual environment):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure secrets.** Copy `.env.example` to `.env` and fill in both values:

   ```
   DISCORD_TOKEN=your-discord-bot-token
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

3. **Enable the Message Content Intent.** In the
   [Discord Developer Portal](https://discord.com/developers/applications) →
   your app → **Bot** → enable **Message Content Intent**. (The bot also
   requests it in code, but Discord gates it here.) Invite the bot to your
   server with permission to read and send messages.

4. **Start the bot:**

   ```bash
   python bot.py
   ```

Elephant stores everything in a local `elephant.db` (created on first run).
It stays quiet by default, speaking only when directly @mentioned, when it can
answer a question from memory, or when a message conflicts with a past
decision. Use `!stats` in any channel to see its activity card.

## Try it without Discord

Replay a scripted group chat through the full pipeline:

```bash
DEMO_OFFLINE=1 python demo/run_demo.py   # canned responses, no API key needed
python demo/run_demo.py                   # uses the live Anthropic API
```
