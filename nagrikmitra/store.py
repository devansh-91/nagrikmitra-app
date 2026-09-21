"""
SQLite persistence: officer inbox + override audit trail.

Responsibilities:
- data/nagrikmitra.db (gitignored, created on first run).
- Tables: tickets (id, text, language, domain, department, priority,
  confidence, needs_human, sla_hours, due_at, source, channel, created_at,
  status), overrides (id, ticket_id, old_domain, old_priority, new_domain,
  new_priority, reason, actor, created_at).
- save_ticket(...) -> ticket_id
- list_tickets(filters) -> list[dict]
- override_ticket(ticket_id, new_domain, new_priority, reason, actor) ->
  writes an audit row AND recomputes/updates due_at from the new priority's SLA.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

from nagrikmitra.config import DB_PATH
from nagrikmitra.taxonomy import get_department, get_sla_hours

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    language TEXT,
    domain TEXT,
    department TEXT,
    priority TEXT,
    confidence REAL,
    needs_human INTEGER,
    sla_hours INTEGER,
    due_at TEXT,
    source TEXT,
    channel TEXT,
    created_at TEXT,
    status TEXT DEFAULT 'New',
    name TEXT,
    location TEXT
);

CREATE TABLE IF NOT EXISTS overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id INTEGER NOT NULL,
    old_domain TEXT,
    old_priority TEXT,
    new_domain TEXT,
    new_priority TEXT,
    reason TEXT,
    actor TEXT,
    created_at TEXT,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);
"""

# Columns added after the initial v1 schema. Added via ALTER TABLE for
# existing databases created before this column existed; CREATE TABLE above
# already includes them for fresh databases.
_MIGRATIONS = [
    "ALTER TABLE tickets ADD COLUMN name TEXT",
    "ALTER TABLE tickets ADD COLUMN location TEXT",
]

STATUSES = ["New", "In Progress", "Resolved"]


def _utcnow() -> datetime:
    """Naive UTC "now", matching the format already stored in created_at/
    due_at (and what the frontend's parseUTC expects). datetime.utcnow() is
    deprecated in 3.13+; this is the timezone-aware replacement stripped
    back to naive so on-disk/API format doesn't change."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't already exist, and apply column
    migrations for databases created before those columns existed."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        for migration in _MIGRATIONS:
            try:
                conn.execute(migration)
            except sqlite3.OperationalError:
                pass  # column already exists
        conn.execute("UPDATE tickets SET status = 'New' WHERE status = 'open'")
        conn.commit()
    finally:
        conn.close()


def save_ticket(
    text: str,
    language: str,
    domain: str,
    department: str,
    priority: str,
    confidence: float,
    needs_human: bool,
    sla_hours: int,
    source: str = "api",
    channel: str = "web",
    name: str = "",
    location: str = "",
) -> int:
    """Insert a new ticket and return its id. due_at is computed from
    created_at + sla_hours."""
    init_db()
    created_at = _utcnow()
    due_at = created_at + timedelta(hours=sla_hours)

    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO tickets
                (text, language, domain, department, priority, confidence,
                 needs_human, sla_hours, due_at, source, channel, created_at,
                 status, name, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', ?, ?)
            """,
            (
                text,
                language,
                domain,
                department,
                priority,
                confidence,
                int(bool(needs_human)),
                sla_hours,
                due_at.isoformat(),
                source,
                channel,
                created_at.isoformat(),
                name or None,
                location or None,
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def list_tickets(filters: dict = None) -> list:
    """List tickets, optionally filtered by domain/priority/language/
    needs_human/status."""
    init_db()
    filters = filters or {}
    allowed_keys = {"domain", "priority", "language", "needs_human", "status"}

    clauses = []
    params = []
    for key, value in filters.items():
        if key not in allowed_keys or value is None:
            continue
        if key == "needs_human":
            value = int(bool(value))
        clauses.append(f"{key} = ?")
        params.append(value)

    query = "SELECT * FROM tickets"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC"

    conn = _connect()
    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_ticket(ticket_id: int) -> dict:
    """Fetch a single ticket by id, or None if not found."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def override_ticket(
    ticket_id: int,
    new_domain: str,
    new_priority: str,
    reason: str,
    actor: str,
) -> dict:
    """Officer override: writes an audit row and recomputes/updates due_at
    from the new priority's SLA. Returns the updated ticket, or None if the
    ticket doesn't exist."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            return None

        old_domain = row["domain"]
        old_priority = row["priority"]

        new_sla_hours = get_sla_hours(new_priority)
        new_department = get_department(new_domain)
        created_at = datetime.fromisoformat(row["created_at"])
        new_due_at = created_at + timedelta(hours=new_sla_hours)
        now = _utcnow().isoformat()

        conn.execute(
            """
            UPDATE tickets
            SET domain = ?, department = ?, priority = ?, sla_hours = ?, due_at = ?, needs_human = 0
            WHERE id = ?
            """,
            (
                new_domain,
                new_department,
                new_priority,
                new_sla_hours,
                new_due_at.isoformat(),
                ticket_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO overrides
                (ticket_id, old_domain, old_priority, new_domain, new_priority,
                 reason, actor, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, old_domain, old_priority, new_domain, new_priority, reason, actor, now),
        )
        conn.commit()

        updated = conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        return dict(updated)
    finally:
        conn.close()


def update_status(ticket_id: int, status: str) -> dict:
    """Officer/citizen-facing status change (New / In Progress / Resolved).
    Does not touch domain/priority/due_at. Returns the updated ticket, or
    None if the ticket doesn't exist."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id)
        )
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        return dict(updated)
    finally:
        conn.close()


def list_overrides(ticket_id: int = None) -> list:
    """List override audit rows, optionally filtered by ticket_id."""
    init_db()
    conn = _connect()
    try:
        if ticket_id is not None:
            rows = conn.execute(
                "SELECT * FROM overrides WHERE ticket_id = ? ORDER BY created_at DESC",
                (ticket_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM overrides ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
