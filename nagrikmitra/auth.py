"""
Citizen/officer authentication: password hashing, signed session cookies,
and optional Google OAuth. See auth_routes.py for the HTTP endpoints that
use these.

Responsibilities:
- hash_password / verify_password (bcrypt).
- Session cookie: create_session_cookie / clear_session_cookie / get_current_user.
  Cookie ("nm_session") is signed (itsdangerous), httpOnly, SameSite=Lax, and
  holds only {uid, role} — the full profile is re-read from the DB on every
  request so it's always current and password_hash never leaves the server.
- require_citizen / require_officer: FastAPI dependencies that 401 if there's
  no session, or the session's role doesn't match.
- Google OAuth (google_configured/google_authorize_url/google_exchange_code):
  plain `requests` calls, same style as nagrikmitra/llm.py's Ollama client —
  no OAuth SDK dependency. Fully inert (google_configured() is False) unless
  GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET are set; see .env.example.
"""
import secrets
from urllib.parse import urlencode

import requests
from fastapi import HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from nagrikmitra import store
from nagrikmitra.config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    IS_RENDER,
    SESSION_SECRET,
    SESSION_SECRET_FILE,
)

SESSION_COOKIE = "nm_session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 days

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _resolve_session_secret() -> str:
    """SESSION_SECRET from env if set; otherwise generate one and persist it
    to data/.session_secret (gitignored) so cookies survive restarts without
    forcing manual setup. A real deployment should set SESSION_SECRET
    explicitly rather than relying on this generated file."""
    if SESSION_SECRET:
        return SESSION_SECRET
    if SESSION_SECRET_FILE.exists():
        return SESSION_SECRET_FILE.read_text(encoding="utf-8").strip()
    SESSION_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_urlsafe(32)
    SESSION_SECRET_FILE.write_text(generated, encoding="utf-8")
    return generated


_serializer = URLSafeTimedSerializer(_resolve_session_secret(), salt="nm-session")


# ---------------------------------------------------------------- passwords

def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    import bcrypt

    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------- sessions

PUBLIC_USER_FIELDS = (
    "id", "role", "username", "email", "full_name", "phone",
    "department", "avatar_url", "created_at",
)


def to_public_user(user_row: dict) -> dict:
    """Strip password_hash/google_sub before a user row leaves the server."""
    return {k: user_row.get(k) for k in PUBLIC_USER_FIELDS}


def create_session_cookie(response: Response, user: dict) -> None:
    token = _serializer.dumps({"uid": user["id"], "role": user["role"]})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=IS_RENDER,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def get_current_user(request: Request) -> dict:
    """Current session's user (password_hash stripped), or None if there's
    no session, it's invalid/expired, or the account no longer exists."""
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        payload = _serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    user = store.get_user_by_id(payload.get("uid"))
    if user is None:
        return None
    return to_public_user(user)


def require_citizen(request: Request) -> dict:
    user = get_current_user(request)
    if user is None or user["role"] != "citizen":
        raise HTTPException(status_code=401, detail="citizen login required")
    return user


def require_officer(request: Request) -> dict:
    user = get_current_user(request)
    if user is None or user["role"] != "officer":
        raise HTTPException(status_code=401, detail="officer login required")
    return user


def make_state(role: str) -> str:
    """Signed 'state' param for the Google OAuth redirect, carrying which
    role (citizen/officer) is signing up/in so the callback knows what to
    create if this is a brand-new account."""
    return _serializer.dumps({"role": role, "nonce": secrets.token_urlsafe(8)})


def read_state(state: str) -> dict:
    try:
        return _serializer.loads(state, max_age=600)  # 10 min to complete the flow
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=400, detail="invalid or expired OAuth state")


# ---------------------------------------------------------------- google oauth

def google_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def google_authorize_url(state: str) -> str:
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def google_exchange_code(code: str) -> dict:
    """Exchange an OAuth authorization code for the signed-in Google
    account's profile: {sub, email, name, picture}. Raises RuntimeError on
    any failure (network, bad code, Google-side error)."""
    try:
        token_resp = requests.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": GOOGLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            },
            timeout=10,
        )
        token_resp.raise_for_status()
        access_token = token_resp.json()["access_token"]

        userinfo_resp = requests.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        userinfo_resp.raise_for_status()
        info = userinfo_resp.json()
    except Exception as exc:
        raise RuntimeError(f"Google OAuth exchange failed: {exc}")

    if not info.get("sub") or not info.get("email"):
        raise RuntimeError("Google did not return a usable profile (sub/email)")

    return {
        "sub": info["sub"],
        "email": info["email"],
        "name": info.get("name") or info["email"].split("@")[0],
        "picture": info.get("picture", ""),
    }
