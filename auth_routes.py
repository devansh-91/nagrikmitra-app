"""
Nagarik Mitra — auth endpoints (citizen + officer accounts).

  POST /auth/register   body: { role, username, email, password, full_name,
                                 phone?, department?, officer_code? }
                         -> public user profile; sets session cookie.
                         role="officer" requires officer_code to match
                         OFFICER_SIGNUP_CODE (env) — rejected outright if
                         that env var isn't set on the server at all.

  POST /auth/login       body: { identifier, password }  (identifier =
                          username OR email) -> public user profile; sets
                          session cookie.

  POST /auth/logout      -> clears the session cookie.

  GET  /auth/me          -> current session's public user profile, or 401.

  GET  /auth/config      -> { google_enabled } so the frontend knows whether
                          to render "Continue with Google".

  GET  /auth/google/login?role=citizen|officer[&officer_code=...]
                          -> redirects to Google's consent screen. 503 if
                          Google isn't configured (GOOGLE_CLIENT_ID/SECRET
                          unset); officer role re-checks officer_code here
                          too, since there's no form step in the OAuth
                          redirect flow to collect it later.

  GET  /auth/google/callback?code=...&state=...
                          -> exchanges the code, creates the account if this
                          Google identity/email is new (else logs the
                          existing one in, or links Google onto a matching
                          password account), sets the session cookie,
                          redirects to "/" (or "/?auth_error=..." on failure).

See nagrikmitra/auth.py for the hashing/session/OAuth mechanics this calls,
and README.md / .env.example for how to obtain real Google OAuth credentials
(nothing here contacts any government portal — unrelated to gov_portal.py).
"""
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from nagrikmitra import auth, store
from nagrikmitra.config import OFFICER_SIGNUP_CODE

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    role: Literal["citizen", "officer"]
    username: str = Field(min_length=3, max_length=40)
    email: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=6, max_length=200)
    full_name: str = Field(min_length=1, max_length=120)
    phone: str = ""
    department: str = ""
    officer_code: str = ""


class LoginRequest(BaseModel):
    identifier: str
    password: str


def _check_officer_code(officer_code: str) -> None:
    if not OFFICER_SIGNUP_CODE:
        raise HTTPException(
            status_code=400,
            detail="officer registration is not enabled on this server (OFFICER_SIGNUP_CODE unset)",
        )
    if officer_code != OFFICER_SIGNUP_CODE:
        raise HTTPException(status_code=403, detail="invalid officer signup code")


@router.post("/register")
def register(req: RegisterRequest, response: Response):
    if req.role == "officer":
        _check_officer_code(req.officer_code)

    if store.get_user_by_identifier(req.username) is not None:
        raise HTTPException(status_code=409, detail="username already taken")
    if store.get_user_by_identifier(req.email) is not None:
        raise HTTPException(status_code=409, detail="email already registered")

    user = store.create_user(
        role=req.role,
        full_name=req.full_name,
        username=req.username,
        email=req.email,
        password_hash=auth.hash_password(req.password),
        phone=req.phone,
        department=req.department,
    )
    auth.create_session_cookie(response, user)
    return auth.to_public_user(user)


@router.post("/login")
def login(req: LoginRequest, response: Response):
    user = store.get_user_by_identifier(req.identifier)
    if user is None or not auth.verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="invalid username/email or password")
    store.touch_last_login(user["id"])
    auth.create_session_cookie(response, user)
    return auth.to_public_user(user)


@router.post("/logout")
def logout(response: Response):
    auth.clear_session_cookie(response)
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    user = auth.get_current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="not logged in")
    return user


@router.get("/config")
def config():
    return {"google_enabled": auth.google_configured()}


@router.get("/google/login")
def google_login(role: Literal["citizen", "officer"], officer_code: str = ""):
    if not auth.google_configured():
        raise HTTPException(status_code=503, detail="Google login is not configured on this server")
    if role == "officer":
        _check_officer_code(officer_code)
    state = auth.make_state(role)
    return RedirectResponse(auth.google_authorize_url(state))


@router.get("/google/callback")
def google_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        return RedirectResponse(f"/?auth_error={quote(error)}")
    if not auth.google_configured():
        raise HTTPException(status_code=503, detail="Google login is not configured on this server")
    if not code or not state:
        return RedirectResponse("/?auth_error=missing+code+or+state")

    state_data = auth.read_state(state)

    try:
        profile = auth.google_exchange_code(code)
    except RuntimeError as e:
        return RedirectResponse(f"/?auth_error={quote(str(e))}")

    user = store.get_user_by_google_sub(profile["sub"])
    if user is None:
        user = store.get_user_by_identifier(profile["email"])
        if user is not None:
            user = store.link_google_account(user["id"], profile["sub"], profile["picture"])
        else:
            user = store.create_user(
                role=state_data["role"],
                full_name=profile["name"],
                email=profile["email"],
                google_sub=profile["sub"],
                avatar_url=profile["picture"],
            )

    store.touch_last_login(user["id"])
    redirect = RedirectResponse("/")
    auth.create_session_cookie(redirect, user)
    return redirect
