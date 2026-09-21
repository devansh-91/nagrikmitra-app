"""
Citizen/officer auth tests (nagrikmitra/auth.py, auth_routes.py, api.py).

Cover:
- Password hashing roundtrip (bcrypt).
- Citizen registration: success, then duplicate username/email rejected.
- Officer registration: rejected with no/wrong signup code, succeeds with
  the right one (auth_routes.OFFICER_SIGNUP_CODE is monkeypatched per-test
  so this doesn't depend on the real environment).
- Login: success (by username AND by email), wrong password, unknown
  identifier.
- Session cookie roundtrip via FastAPI's TestClient: register -> cookie set
  -> GET /auth/me -> logout -> GET /auth/me now 401.
- Officer-only endpoints (/tickets, /tickets/{id}/status) 401 for anonymous
  and for a citizen session, succeed for an officer session.
- GET /my-tickets returns only the logged-in citizen's own tickets.

Each test gets an isolated on-disk SQLite DB (tmp_path) via monkeypatching
nagrikmitra.store.DB_PATH, so this never touches data/nagrikmitra.db.
"""
import pytest
from fastapi.testclient import TestClient

import api
import auth_routes
from nagrikmitra import store
from nagrikmitra.auth import hash_password, verify_password


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test_nagrikmitra.db")
    monkeypatch.setattr(auth_routes, "OFFICER_SIGNUP_CODE", "test-signup-code")
    return TestClient(api.app)


def register_citizen(client, username="ramesh", email="ramesh@example.com"):
    return client.post(
        "/auth/register",
        json={
            "role": "citizen",
            "username": username,
            "email": email,
            "password": "secret123",
            "full_name": "Ramesh Kumar",
        },
    )


def register_officer(client, code="test-signup-code", username="officer1"):
    return client.post(
        "/auth/register",
        json={
            "role": "officer",
            "username": username,
            "email": f"{username}@example.com",
            "password": "secret123",
            "full_name": "Priya Officer",
            "department": "PWD",
            "officer_code": code,
        },
    )


def test_password_hash_roundtrip():
    h = hash_password("secret123")
    assert h != "secret123"
    assert verify_password("secret123", h) is True
    assert verify_password("wrong", h) is False


def test_register_citizen_success(client):
    r = register_citizen(client)
    assert r.status_code == 200
    body = r.json()
    assert body["role"] == "citizen"
    assert body["username"] == "ramesh"
    assert "password_hash" not in body


def test_register_citizen_duplicate_username_rejected(client):
    register_citizen(client)
    r = client.post(
        "/auth/register",
        json={
            "role": "citizen",
            "username": "ramesh",
            "email": "someone.else@example.com",
            "password": "secret123",
            "full_name": "Someone Else",
        },
    )
    assert r.status_code == 409


def test_register_citizen_duplicate_email_rejected(client):
    register_citizen(client)
    r = client.post(
        "/auth/register",
        json={
            "role": "citizen",
            "username": "someone_else",
            "email": "ramesh@example.com",
            "password": "secret123",
            "full_name": "Someone Else",
        },
    )
    assert r.status_code == 409


def test_register_officer_no_code_provided_rejected(client):
    """A code IS configured (via the fixture) but the request omits it —
    treated the same as a wrong code: 403."""
    r = client.post(
        "/auth/register",
        json={
            "role": "officer",
            "username": "officer1",
            "email": "officer1@example.com",
            "password": "secret123",
            "full_name": "Priya Officer",
        },
    )
    assert r.status_code == 403


def test_register_officer_wrong_code_rejected(client):
    r = register_officer(client, code="wrong-code")
    assert r.status_code == 403


def test_register_officer_rejected_when_signup_code_unset_on_server(client, monkeypatch):
    """When OFFICER_SIGNUP_CODE isn't configured on the server at all,
    officer registration is rejected outright (400), regardless of what
    code the client sends — never silently open."""
    monkeypatch.setattr(auth_routes, "OFFICER_SIGNUP_CODE", "")
    r = register_officer(client, code="anything")
    assert r.status_code == 400


def test_register_officer_correct_code_succeeds(client):
    r = register_officer(client)
    assert r.status_code == 200
    assert r.json()["role"] == "officer"


def test_login_success_by_username(client):
    register_citizen(client)
    fresh = TestClient(api.app)
    r = fresh.post("/auth/login", json={"identifier": "ramesh", "password": "secret123"})
    assert r.status_code == 200
    assert r.json()["role"] == "citizen"


def test_login_success_by_email(client):
    register_citizen(client)
    fresh = TestClient(api.app)
    r = fresh.post("/auth/login", json={"identifier": "ramesh@example.com", "password": "secret123"})
    assert r.status_code == 200


def test_login_wrong_password_rejected(client):
    register_citizen(client)
    fresh = TestClient(api.app)
    r = fresh.post("/auth/login", json={"identifier": "ramesh", "password": "wrongpass"})
    assert r.status_code == 401


def test_login_unknown_identifier_rejected(client):
    r = client.post("/auth/login", json={"identifier": "nobody", "password": "whatever"})
    assert r.status_code == 401


def test_session_roundtrip_me_and_logout(client):
    register_citizen(client)
    r = client.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["username"] == "ramesh"

    client.post("/auth/logout")
    r = client.get("/auth/me")
    assert r.status_code == 401


def test_officer_only_endpoints_reject_anonymous(client):
    r = client.get("/tickets")
    assert r.status_code == 401
    r = client.patch("/tickets/1/status", json={"status": "New"})
    assert r.status_code == 401


def test_officer_only_endpoints_reject_citizen(client):
    register_citizen(client)
    r = client.get("/tickets")
    assert r.status_code == 401


def test_officer_only_endpoints_succeed_for_officer(client):
    register_officer(client)
    r = client.get("/tickets")
    assert r.status_code == 200
    r = client.get("/analytics")
    assert r.status_code == 200


def test_predict_attaches_logged_in_citizen(client):
    citizen = register_citizen(client).json()
    r = client.post(
        "/predict",
        json={"text": "sadak par gadda hai bahut khatarnak hai", "persist": True, "name": "ignored"},
    )
    assert r.status_code == 200
    ticket_id = r.json()["ticket_id"]

    ticket = store.get_ticket(ticket_id)
    assert ticket["citizen_id"] == citizen["id"]
    assert ticket["name"] == "Ramesh Kumar"  # account's full_name, not the "ignored" client value


def test_predict_guest_submission_still_works(client):
    r = client.post("/predict", json={"text": "kachra nahi utha bahut badbu hai", "persist": True})
    assert r.status_code == 200
    assert r.json()["ticket_id"] is not None


def test_my_tickets_scoped_to_logged_in_citizen(client):
    r1 = register_citizen(client, username="ramesh", email="ramesh@example.com")
    citizen1_id = r1.json()["id"]
    client.post("/predict", json={"text": "sadak par gadda hai", "persist": True})
    client.post("/auth/logout")

    other_client = TestClient(api.app)  # separate cookie jar, same DB
    r2 = register_citizen(other_client, username="sunita", email="sunita@example.com")
    other_client.post("/predict", json={"text": "bijli chali gayi hai", "persist": True})

    r = client.post("/auth/login", json={"identifier": "ramesh", "password": "secret123"})
    assert r.status_code == 200
    r = client.get("/my-tickets")
    assert r.status_code == 200
    tickets = r.json()
    assert len(tickets) == 1
    assert tickets[0]["citizen_id"] == citizen1_id

    r = other_client.get("/my-tickets")
    assert len(r.json()) == 1
    assert r.json()[0]["citizen_id"] == r2.json()["id"]


def test_override_uses_officer_account_name_as_actor(client):
    register_citizen(client)
    r = client.post("/predict", json={"text": "sadak par gadda hai", "persist": True})
    ticket_id = r.json()["ticket_id"]
    client.post("/auth/logout")

    register_officer(client)
    r = client.post(
        f"/tickets/{ticket_id}/override",
        json={"new_domain": "Roads", "new_priority": "High", "reason": "test"},
    )
    assert r.status_code == 200

    overrides = store.list_overrides(ticket_id)
    assert overrides[0]["actor"] == "Priya Officer"
