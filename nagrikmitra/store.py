"""
SQLite persistence: officer inbox + override audit trail.

Responsibilities:
- data/nagrikmitra.db (gitignored, created on first run).
- Tables: tickets (id, text, language, domain, department, priority,
  confidence, needs_human, sla_hours, due_at, source, channel, created_at,
  status, name, location, portal_name, portal_reference, portal_status),
  overrides (id, ticket_id, old_domain, old_priority, new_domain,
  new_priority, reason, actor, created_at).
- save_ticket(...) -> ticket_id
- list_tickets(filters) -> list[dict]
- override_ticket(ticket_id, new_domain, new_priority, reason, actor) ->
  writes an audit row AND recomputes/updates due_at from the new priority's SLA.
- set_portal_submission(ticket_id, portal_name, portal_reference, portal_status)
  -> records the outcome of forwarding a ticket to a (simulated) gov portal
  adapter (see gov_portal.py — no real portal is ever contacted).
- users (id, role, username, email, password_hash, google_sub, full_name,
  phone, department, avatar_url, created_at, last_login_at) — citizen/officer
  accounts (see nagrikmitra/auth.py, auth_routes.py). tickets.citizen_id is a
  nullable FK onto this table; guest (unauthenticated) tickets leave it NULL.
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

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT NOT NULL CHECK(role IN ('citizen','officer')),
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    password_hash TEXT,
    google_sub TEXT UNIQUE,
    full_name TEXT,
    phone TEXT,
    department TEXT,
    avatar_url TEXT,
    created_at TEXT,
    last_login_at TEXT
);
"""

# Columns added after the initial v1 schema. Added via ALTER TABLE for
# existing databases created before this column existed; CREATE TABLE above
# already includes them for fresh databases.
_MIGRATIONS = [
    "ALTER TABLE tickets ADD COLUMN name TEXT",
    "ALTER TABLE tickets ADD COLUMN location TEXT",
    "ALTER TABLE tickets ADD COLUMN portal_name TEXT",
    "ALTER TABLE tickets ADD COLUMN portal_reference TEXT",
    "ALTER TABLE tickets ADD COLUMN portal_status TEXT",
    "ALTER TABLE tickets ADD COLUMN citizen_id INTEGER",
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
    citizen_id: int = None,
) -> int:
    """Insert a new ticket and return its id. due_at is computed from
    created_at + sla_hours. citizen_id is None for guest (unauthenticated)
    submissions."""
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
                 status, name, location, citizen_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', ?, ?, ?)
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
                citizen_id,
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


def list_tickets_for_citizen(citizen_id: int) -> list:
    """List tickets filed by a specific logged-in citizen."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM tickets WHERE citizen_id = ? ORDER BY created_at DESC",
            (citizen_id,),
        ).fetchall()
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


def set_portal_submission(
    ticket_id: int, portal_name: str, portal_reference: str, portal_status: str
) -> dict:
    """Record the outcome of forwarding a ticket to a (simulated) government
    portal adapter. Returns the updated ticket, or None if it doesn't exist."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT id FROM tickets WHERE id = ?", (ticket_id,)
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            "UPDATE tickets SET portal_name = ?, portal_reference = ?, portal_status = ? WHERE id = ?",
            (portal_name, portal_reference, portal_status, ticket_id),
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


# ---------------------------------------------------------------- users


def create_user(
    role: str,
    full_name: str,
    username: str = None,
    email: str = None,
    password_hash: str = None,
    google_sub: str = None,
    phone: str = "",
    department: str = "",
    avatar_url: str = "",
) -> dict:
    """Create a citizen or officer account. Raises sqlite3.IntegrityError if
    username/email/google_sub already exists (UNIQUE constraints). Returns
    the created user row."""
    if role not in ("citizen", "officer"):
        raise ValueError("role must be 'citizen' or 'officer'")
    init_db()
    created_at = _utcnow().isoformat()
    conn = _connect()
    try:
        cur = conn.execute(
            """
            INSERT INTO users
                (role, username, email, password_hash, google_sub, full_name,
                 phone, department, avatar_url, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                role,
                username or None,
                email or None,
                password_hash,
                google_sub,
                full_name,
                phone or None,
                department or None,
                avatar_url or None,
                created_at,
                created_at,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (cur.lastrowid,)
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def get_user_by_id(user_id: int) -> dict:
    """Fetch a user by id, or None if not found."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_identifier(identifier: str) -> dict:
    """Fetch a user by username OR email (for login), or None if not found."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_google_sub(google_sub: str) -> dict:
    """Fetch a user by their linked Google account id, or None if not found."""
    init_db()
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE google_sub = ?", (google_sub,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def link_google_account(user_id: int, google_sub: str, avatar_url: str = "") -> dict:
    """Attach a Google account id to an existing (password-based) user, e.g.
    when the same email signs in with Google for the first time."""
    init_db()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE users SET google_sub = ?, avatar_url = COALESCE(?, avatar_url) WHERE id = ?",
            (google_sub, avatar_url or None, user_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def touch_last_login(user_id: int) -> None:
    """Update a user's last_login_at to now."""
    init_db()
    conn = _connect()
    try:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (_utcnow().isoformat(), user_id),
        )
        conn.commit()
    finally:
        conn.close()
