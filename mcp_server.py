"""
Nagarik Mitra — MCP server.

Exposes the same routing engine api.py serves over HTTP (nagrikmitra.predict,
nagrikmitra.store, nagrikmitra.reply) as MCP tools, so an MCP client (Claude
Desktop, etc.) can classify complaints and manage the officer inbox directly.
Imports the package in-process rather than proxying the FastAPI server, so it
works standalone; it shares data/nagrikmitra.db with api.py if both run at
once (SQLite handles that fine at this scale).

Government portal integration (gov_portal.py):
  There is no public submission API for CPGRAMS (pgportal.gov.in) or the
  state/municipal grievance portals as far as this project could confirm —
  their citizen-facing forms are CAPTCHA/OTP-gated specifically to prevent
  automated filing. This server does NOT scrape or automate any live portal:
  gov_portal.py defines the adapter interface a real integration would need
  and ships a `MockGovPortalAdapter` that simulates one for demos. Swap in a
  real adapter only once you hold an actual, documented API credential from
  that portal's operator (see gov_portal.py's module docstring).

Run:
  .venv/bin/python mcp_server.py                     # stdio transport
Configure in an MCP client (e.g. Claude Desktop's claude_desktop_config.json):
  {"mcpServers": {"nagrikmitra": {"command": "/absolute/path/to/.venv/bin/python",
                                   "args": ["/absolute/path/to/mcp_server.py"]}}}
"""
from typing import Optional

from mcp.server.fastmcp import FastMCP

from gov_portal import MockGovPortalAdapter, PortalSubmissionError
from nagrikmitra import store
from nagrikmitra.llm import polish
from nagrikmitra.predict import predict
from nagrikmitra.reply import render_template
from nagrikmitra.similar import similar_tickets
from nagrikmitra.store import STATUSES
from nagrikmitra.taxonomy import DOMAINS, PRIORITIES

mcp = FastMCP("nagrikmitra")
_gov_portal = MockGovPortalAdapter()


@mcp.tool()
def classify_complaint(
    text: str,
    name: str = "",
    location: str = "",
    channel: str = "web",
    persist: bool = False,
    use_llm: bool = False,
) -> dict:
    """Run a civic grievance through the routing pipeline: detect language,
    classify its domain (Roads/Lights/Water/Waste/Sanitation/Electricity/
    Health/Other), score priority, assign an SLA, and draft an acknowledgment
    reply. sklearn (TF-IDF + Logistic Regression) owns domain/priority; set
    persist=True to also save it to the local officer inbox (data/nagrikmitra.db).
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    result = predict(text, channel=channel)
    template_reply = render_template(
        language=result["language"],
        domain=result["domain"],
        priority=result["priority"],
        sla_hours=result["sla_hours"],
        department=result["department"],
    )

    suggested_reply = template_reply
    source = "template"
    llm_backend = None
    llm_error = None
    if use_llm:
        polished_text, used, backend, error = polish(
            template_reply, result["sla_hours"], result["language"]
        )
        llm_backend, llm_error = backend, error
        if used:
            suggested_reply, source = polished_text, "llm"

    ticket_id = None
    if persist:
        ticket_id = store.save_ticket(
            text=text,
            language=result["language"],
            domain=result["domain"],
            department=result["department"],
            priority=result["priority"],
            confidence=result["confidence"],
            needs_human=result["needs_human"],
            sla_hours=result["sla_hours"],
            source="mcp",
            channel=channel,
            name=name,
            location=location,
        )

    return {
        "ticket_id": ticket_id,
        "language": result["language"],
        "normalized_text": result["clean_text"],
        "domain": result["domain"],
        "department": result["department"],
        "priority": result["priority"],
        "confidence": result["confidence"],
        "needs_human": result["needs_human"],
        "sla_hours": result["sla_hours"],
        "suggested_reply": suggested_reply,
        "source": source,
        "llm_backend": llm_backend,
        "llm_error": llm_error,
        "domain_top3": result["domain_top3"],
        "priority_top3": result["priority_top3"],
        "similar": similar_tickets(text, top_k=5),
    }


@mcp.tool()
def list_tickets(
    domain: Optional[str] = None,
    priority: Optional[str] = None,
    language: Optional[str] = None,
    needs_human: Optional[bool] = None,
    status: Optional[str] = None,
) -> list:
    """List tickets from the officer inbox, optionally filtered by domain,
    priority, language, needs_human, or status (New/In Progress/Resolved)."""
    return store.list_tickets(
        {
            "domain": domain,
            "priority": priority,
            "language": language,
            "needs_human": needs_human,
            "status": status,
        }
    )


@mcp.tool()
def get_ticket(ticket_id: int) -> dict:
    """Fetch a single ticket by id from the officer inbox."""
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"ticket {ticket_id} not found")
    return ticket


@mcp.tool()
def update_ticket_status(ticket_id: int, status: str) -> dict:
    """Change a ticket's workflow status (New / In Progress / Resolved).
    Does not touch its domain, priority, or SLA due date."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    updated = store.update_status(ticket_id, status)
    if updated is None:
        raise ValueError(f"ticket {ticket_id} not found")
    return updated


@mcp.tool()
def override_ticket(
    ticket_id: int,
    new_domain: str,
    new_priority: str,
    reason: str = "",
    actor: str = "officer",
) -> dict:
    """Officer override of a ticket's classified domain/priority. Writes an
    audit row (nagrikmitra.store.list_overrides) and recomputes the SLA due date."""
    if new_domain not in DOMAINS:
        raise ValueError(f"new_domain must be one of {DOMAINS}")
    if new_priority not in PRIORITIES:
        raise ValueError(f"new_priority must be one of {PRIORITIES}")
    updated = store.override_ticket(
        ticket_id=ticket_id,
        new_domain=new_domain,
        new_priority=new_priority,
        reason=reason,
        actor=actor,
    )
    if updated is None:
        raise ValueError(f"ticket {ticket_id} not found")
    return updated


@mcp.tool()
def get_analytics() -> dict:
    """Volume of tickets by domain/priority/language and count of SLA breaches
    among open (non-Resolved) tickets in the officer inbox."""
    from datetime import datetime, timezone

    tickets = store.list_tickets({})
    volume_by_domain, volume_by_priority, volume_by_language = {}, {}, {}
    sla_breaches = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for t in tickets:
        volume_by_domain[t["domain"]] = volume_by_domain.get(t["domain"], 0) + 1
        volume_by_priority[t["priority"]] = volume_by_priority.get(t["priority"], 0) + 1
        volume_by_language[t["language"]] = volume_by_language.get(t["language"], 0) + 1
        if t["status"] != "Resolved" and t["due_at"]:
            if now > datetime.fromisoformat(t["due_at"]):
                sla_breaches += 1
    return {
        "total_tickets": len(tickets),
        "volume_by_domain": volume_by_domain,
        "volume_by_priority": volume_by_priority,
        "volume_by_language": volume_by_language,
        "sla_breaches": sla_breaches,
    }


@mcp.tool()
def submit_to_gov_portal(ticket_id: int, portal: str = "CPGRAMS") -> dict:
    """Forward a locally logged ticket to an external government grievance
    portal. NOT connected to a real portal: no public submission API for
    CPGRAMS or state/municipal portals was found, and their citizen forms are
    CAPTCHA/OTP-gated specifically against automated filing. This calls a
    MockGovPortalAdapter that simulates the round trip for demo purposes and
    never contacts any live government system. See gov_portal.py to wire in a
    real adapter once you hold an actual API credential for that portal."""
    ticket = store.get_ticket(ticket_id)
    if ticket is None:
        raise ValueError(f"ticket {ticket_id} not found")
    try:
        return _gov_portal.submit(ticket, portal=portal)
    except PortalSubmissionError as e:
        return {"submitted": False, "portal": portal, "error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
