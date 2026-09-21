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
            explain, domain_top3, priority_top3, ticket_id }

  GET  /tickets            (filterable: domain, priority, language, needs_human, status)
  POST /tickets/{id}/override
       body: { new_domain, new_priority, reason, actor }
       -> writes audit row, recomputes due_at
  PATCH /tickets/{id}/status
       body: { status }   status in ["New", "In Progress", "Resolved"]

  GET  /analytics
       -> volume by domain/priority/language, SLA breach counts

sklearn always runs and owns domain/priority decisions; the LLM (Groq/Ollama,
optional) only polishes the template reply text and can never change routing.

GET "/" and any unmatched path serve frontend/index.html (the citizen/officer
web UI), which talks to the API above via same-origin fetch calls.
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from nagrikmitra import store
from nagrikmitra.config import BASE_DIR, GROQ_API_KEY, GROQ_MODEL, OLLAMA_MODEL
from nagrikmitra.explain import explain_prediction
from nagrikmitra.llm import active_backend, polish
from nagrikmitra.predict import load_models, predict
from nagrikmitra.reply import render_template
from nagrikmitra.similar import similar_tickets
from nagrikmitra.store import STATUSES


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
    actor: str = "officer"


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
def do_predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

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
            name=req.name,
            location=req.location,
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
        "template_reply": template_reply,
        "source": source,
        "llm_backend": llm_backend,
        "llm_error": llm_error,
        "similar": similar,
        "explain": explain,
        "domain_top3": result["domain_top3"],
        "priority_top3": result["priority_top3"],
    }


@app.get("/tickets")
def get_tickets(
    domain: Optional[str] = None,
    priority: Optional[str] = None,
    language: Optional[str] = None,
    needs_human: Optional[bool] = None,
    status: Optional[str] = None,
):
    filters = {
        "domain": domain,
        "priority": priority,
        "language": language,
        "needs_human": needs_human,
        "status": status,
    }
    return store.list_tickets(filters)


@app.post("/tickets/{ticket_id}/override")
def override_ticket(ticket_id: int, req: OverrideRequest):
    updated = store.override_ticket(
        ticket_id=ticket_id,
        new_domain=req.new_domain,
        new_priority=req.new_priority,
        reason=req.reason,
        actor=req.actor,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return updated


@app.patch("/tickets/{ticket_id}/status")
def update_ticket_status(ticket_id: int, req: StatusRequest):
    if req.status not in STATUSES:
        raise HTTPException(
            status_code=400, detail=f"status must be one of {STATUSES}"
        )
    updated = store.update_status(ticket_id, req.status)
    if updated is None:
        raise HTTPException(status_code=404, detail="ticket not found")
    return updated


@app.get("/analytics")
def analytics():
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
