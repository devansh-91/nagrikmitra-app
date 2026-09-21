"""
Nagarik Mitra — FastAPI service.

Endpoints (per spec section 7):
  GET  /health
       -> { ok, service, llm: { groq, ollama, active, model } }

  POST /predict
       body: { text, use_llm, channel, persist, name, location }
       -> { language, normalized_text, domain, department, priority,
            confidence, needs_human, sla_hours, suggested_reply,
            template_reply, source, llm_backend, llm_error, similar,
            explain, domain_top3, priority_top3, ticket_id,
            portal_name, portal_reference, portal_status, portal_note }

  GET  /tickets            (filterable: domain, priority, language, needs_human, status)
       -> officer-only (401 without an officer session)
  GET  /my-tickets         -> citizen-only: tickets filed by the logged-in citizen
  POST /tickets/{id}/override
       body: { new_domain, new_priority, reason }
       -> officer-only; writes audit row (actor = the officer's account),
          recomputes due_at
  PATCH /tickets/{id}/status
       body: { status }   status in ["New", "In Progress", "Resolved"]
       -> officer-only

  GET  /analytics
       -> officer-only; volume by domain/priority/language, SLA breach counts

sklearn always runs and owns domain/priority decisions; the LLM (Groq/Ollama,
optional) only polishes the template reply text and can never change routing.

When a complaint is persisted, it is also forwarded to gov_portal's
MockGovPortalAdapter — a SIMULATED per-domain government portal (see
gov_portal.py). No real CPGRAMS or department portal is ever contacted; the
portal_* response fields and portal_note make that explicit.

Auth (see nagrikmitra/auth.py, auth_routes.py) is cookie-session based, with
separate citizen/officer accounts. POST /predict never requires login — guest
submission still works — but attaches the caller's account when a citizen
session is present. /tickets, /tickets/{id}/override, /tickets/{id}/status,
and /analytics require an officer session.

GET "/" and any unmatched path serve frontend/index.html (the citizen/officer
web UI), which talks to the API above via same-origin fetch calls.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth_routes
from gov_portal import MockGovPortalAdapter, PortalSubmissionError
from nagrikmitra import auth, store
from nagrikmitra.config import BASE_DIR, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL
from nagrikmitra.explain import explain_prediction
from nagrikmitra.llm import active_backend, polish
from nagrikmitra.predict import load_models, predict
from nagrikmitra.reply import render_template
from nagrikmitra.similar import similar_tickets
from nagrikmitra.store import STATUSES

_gov_portal = MockGovPortalAdapter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    # Warm the sklearn models so the first /predict call isn't slow.
    load_models()
    yield


app = FastAPI(title="Nagarik Mitra", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)


class PredictRequest(BaseModel):
    text: str
    use_llm: bool = False
    channel: str = "web"
    persist: bool = False
    name: str = ""
    location: str = ""


class OverrideRequest(BaseModel):
    new_domain: str
    new_priority: str
    reason: str = ""


class StatusRequest(BaseModel):
    status: str


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "nagrikmitra",
        "llm": {
            "groq": bool(GROQ_API_KEY),
            "ollama": True,
            "active": active_backend(),
            "model": GROQ_MODEL if GROQ_API_KEY else OLLAMA_MODEL,
        },
    }


@app.post("/predict")
def do_predict(req: PredictRequest, request: Request):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Guest submission always works — no login required. If a citizen IS
    # logged in, attach their account and use its name instead of whatever
    # the client sent for "name".
    current_user = auth.get_current_user(request)
    citizen_id = current_user["id"] if current_user and current_user["role"] == "citizen" else None
    ticket_name = current_user["full_name"] if citizen_id else req.name

    result = predict(req.text, channel=req.channel)

    template_reply = render_template(
        language=result["language"],
        domain=result["domain"],
        priority=result["priority"],
        sla_hours=result["sla_hours"],
        department=result["department"],
    )

    suggested_reply = template_reply
    llm_backend = None
    llm_error = None
    source = "template"

    if req.use_llm:
        polished_text, used, backend, error = polish(
            template_reply, result["sla_hours"], result["language"]
        )
        llm_backend = backend
        llm_error = error
        if used:
            suggested_reply = polished_text
            source = "llm"

    domain_model, priority_model = load_models()
    explain = explain_prediction(domain_model, priority_model, result["clean_text"])

    similar = similar_tickets(req.text, top_k=5)

    ticket_id = None
    portal_name = portal_reference = portal_status = portal_note = None
    if req.persist:
        ticket_id = store.save_ticket(
            text=req.text,
            language=result["language"],
            domain=result["domain"],
            department=result["department"],
            priority=result["priority"],
            confidence=result["confidence"],
            needs_human=result["needs_human"],
            sla_hours=result["sla_hours"],
            source="api",
            channel=req.channel,
            name=ticket_name,
            location=req.location,
            citizen_id=citizen_id,
        )

        # Forward to the SIMULATED per-domain government portal adapter (see
        # gov_portal.py) — no real portal is contacted, ever.
        try:
            portal_result = _gov_portal.submit(store.get_ticket(ticket_id))
            portal_name = portal_result["portal"]
            portal_reference = portal_result["portal_reference"]
            portal_status = "Submitted (Simulated)"
            portal_note = portal_result["note"]
            store.set_portal_submission(ticket_id, portal_name, portal_reference, portal_status)
        except PortalSubmissionError as e:
            portal_status = f"Not submitted (Simulated): {e}"

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
        "template_reply": template_reply,
        "source": source,
        "llm_backend": llm_backend,
        "llm_error": llm_error,
        "similar": similar,
        "explain": explain,
        "domain_top3": result["domain_top3"],
        "priority_top3": result["priority_top3"],
        "portal_name": portal_name,
        "portal_reference": portal_reference,
        "portal_status": portal_status,
        "portal_note": portal_note,
    }


@app.get("/tickets")
def get_tickets(
    domain: Optional[str] = None,
    priority: Optional[str] = None,
    language: Optional[str] = None,
    needs_human: Optional[bool] = None,
    status: Optional[str] = None,
    officer: dict = Depends(auth.require_officer),
):
    filters = {
        "domain": domain,
        "priority": priority,
        "language": language,
        "needs_human": needs_human,
        "status": status,
    }
    return store.list_tickets(filters)


@app.get("/my-tickets")
def my_tickets(citizen: dict = Depends(auth.require_citizen)):
    return store.list_tickets_for_citizen(citizen["id"])


@app.post("/tickets/{ticket_id}/override")
def override_ticket(ticket_id: int, req: OverrideRequest, officer: dict = Depends(auth.require_officer)):
    updated = store.override_ticket(
        ticket_id=ticket_id,
        new_domain=req.new_domain,
        new_priority=req.new_priority,
        reason=req.reason,
        actor=officer["full_name"],
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return updated


@app.patch("/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: int, req: StatusRequest, officer: dict = Depends(auth.require_officer)):
    if req.status not in STATUSES:
        raise HTTPException(
            status_code=400, detail=f"status must be one of {STATUSES}"
        )
    updated = store.update_status(ticket_id, req.status)
    if updated is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return updated


@app.get("/analytics")
def analytics(officer: dict = Depends(auth.require_officer)):
    tickets = store.list_tickets({})

    volume_by_domain = {}
    volume_by_priority = {}
    volume_by_language = {}
    sla_breaches = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for t in tickets:
        volume_by_domain[t["domain"]] = volume_by_domain.get(t["domain"], 0) + 1
        volume_by_priority[t["priority"]] = volume_by_priority.get(t["priority"], 0) + 1
        volume_by_language[t["language"]] = volume_by_language.get(t["language"], 0) + 1

        if t["status"] != "Resolved" and t["due_at"]:
            due_at = datetime.fromisoformat(t["due_at"])
            if now > due_at:
                sla_breaches += 1

    return {
        "total_tickets": len(tickets),
        "volume_by_domain": volume_by_domain,
        "volume_by_priority": volume_by_priority,
        "volume_by_language": volume_by_language,
        "sla_breaches": sla_breaches,
    }


# Serves frontend/index.html (the citizen/officer web UI) at "/" and its
# static assets alongside it. Mounted last so it never shadows the API
# routes defined above — Starlette matches routes in registration order.
app.mount("/", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="frontend")
