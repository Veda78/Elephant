"""SQLite-backed memory store for Elephant.

Uses only the Python standard library (sqlite3). A single MemoryStore owns one
database file and creates its schema on construction.
"""

import sqlite3
import time


class MemoryStore:
    """Persistent store for messages, tracked items, and bot stats."""

    def __init__(self, db_path):
        self.db_path = db_path
        # check_same_thread=False so the store can be shared across the bot's
        # async callbacks; sqlite3 serializes writes internally.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                channel   TEXT NOT NULL,
                author    TEXT NOT NULL,
                text      TEXT NOT NULL,
                timestamp REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS items (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                kind              TEXT NOT NULL,
                who               TEXT NOT NULL,
                what              TEXT NOT NULL,
                due               REAL,
                status            TEXT NOT NULL DEFAULT 'open',
                source_message_id INTEGER,
                created_at        REAL NOT NULL,
                FOREIGN KEY (source_message_id) REFERENCES messages (id)
            );

            CREATE TABLE IF NOT EXISTS stats (
                id      INTEGER PRIMARY KEY CHECK (id = 1),
                seen    INTEGER NOT NULL DEFAULT 0,
                replied INTEGER NOT NULL DEFAULT 0
            );

            INSERT OR IGNORE INTO stats (id, seen, replied) VALUES (1, 0, 0);
            """
        )
        self.conn.commit()

    # -- messages ---------------------------------------------------------

    def add_message(self, channel, author, text):
        """Store a message and return its id."""
        cur = self.conn.execute(
            "INSERT INTO messages (channel, author, text, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (channel, author, text, time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def search_messages(self, channel, keyword, limit=10):
        """Return recent messages in a channel whose text contains keyword."""
        rows = self.conn.execute(
            "SELECT * FROM messages "
            "WHERE channel = ? AND text LIKE ? "
            "ORDER BY timestamp DESC LIMIT ?",
            (channel, "%" + keyword + "%", limit),
        ).fetchall()
        return [dict(row) for row in rows]

    # -- items ------------------------------------------------------------

    def add_item(self, kind, who, what, due, source_message_id):
        """Store a promise or decision and return its id."""
        cur = self.conn.execute(
            "INSERT INTO items (kind, who, what, due, status, "
            "source_message_id, created_at) VALUES (?, ?, ?, ?, 'open', ?, ?)",
            (kind, who, what, due, source_message_id, time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_open_items(self, channel):
        """Return all open items whose source message is in the channel."""
        rows = self.conn.execute(
            "SELECT items.* FROM items "
            "JOIN messages ON items.source_message_id = messages.id "
            "WHERE items.status = 'open' AND messages.channel = ? "
            "ORDER BY items.created_at",
            (channel,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_due_items(self, now):
        """Return open promises with a due time at or before `now`."""
        rows = self.conn.execute(
            "SELECT * FROM items "
            "WHERE status = 'open' AND kind = 'promise' "
            "AND due IS NOT NULL AND due <= ? "
            "ORDER BY due",
            (now,),
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_done(self, item_id):
        """Mark an item as done."""
        self.conn.execute(
            "UPDATE items SET status = 'done' WHERE id = ?", (item_id,)
        )
        self.conn.commit()

    # -- stats ------------------------------------------------------------

    def record_seen(self):
        """Count one message observed by Elephant."""
        self.conn.execute("UPDATE stats SET seen = seen + 1 WHERE id = 1")
        self.conn.commit()

    def record_reply(self):
        """Count one message Elephant replied to."""
        self.conn.execute("UPDATE stats SET replied = replied + 1 WHERE id = 1")
        self.conn.commit()

    def get_silence_score(self):
        """Share of seen messages Elephant did not reply to, as a percentage."""
        row = self.conn.execute(
            "SELECT seen, replied FROM stats WHERE id = 1"
        ).fetchone()
        if not row or row["seen"] == 0:
            return 0.0
        return (row["seen"] - row["replied"]) / row["seen"] * 100

    def close(self):
        self.conn.close()
