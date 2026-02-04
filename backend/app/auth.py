from __future__ import annotations

import hmac
import json
import base64
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from fastapi import Depends, HTTPException, Request, status


# Role Base access control

ROLE_HOSPITAL = "Hospital"
ROLE_RESEARCHER = "Researcher"
ROLE_BIOBANK = "Biobank"
ROLE_ADMIN = "Admin"

ALL_ROLES = {ROLE_HOSPITAL, ROLE_RESEARCHER, ROLE_BIOBANK, ROLE_ADMIN}

AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me")

TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "86400"))  # 24h default


@dataclass(frozen=True)
class Actor:
    username: str
    role: str
    org: str
    is_active: bool = True


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sign(payload_b64: str) -> str:
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(sig)


def issue_token(actor: Actor) -> str:
    now = int(time.time())
    payload = {
        "sub": actor.username,
        "role": actor.role,
        "org": actor.org,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
        "active": bool(actor.is_active),
    }
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    sig = _sign(payload_b64)
    return f"{payload_b64}.{sig}"


def verify_token(token: str) -> Actor:  # Επαληθεύει token και επιστρέφει Actor

    try:
        payload_b64, sig = token.split(".", 1)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    expected = _sign(payload_b64)
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    now = int(time.time())
    if int(payload.get("exp", 0)) < now:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")

    username = str(payload.get("sub", "")).strip()
    role = str(payload.get("role", "")).strip()
    org = str(payload.get("org", "")).strip()
    active = bool(payload.get("active", True))

    if not username or not org or role not in ALL_ROLES:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token claims")
    if not active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User inactive")

    return Actor(username=username, role=role, org=org, is_active=active)


def get_actor(request: Request) -> Actor:
    auth = request.headers.get("Authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    return verify_token(token)


def require_roles(*roles: str):
    allowed: Sequence[str] = roles

    def _dep(actor: Actor = Depends(get_actor)) -> Actor:
        # Admin = superuser
        if actor.role == ROLE_ADMIN:
            return actor

        if actor.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden for role '{actor.role}'",
            )
        return actor

    return _dep



def hash_password(password: str) -> str:
    if not password or len(password) < 4:
        raise ValueError("Password too short")

    iterations = int(os.getenv("PWD_ITERATIONS", "210000"))
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8").rstrip("=")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("utf-8").rstrip("=")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, it_s, salt_b64, dk_b64 = stored.split("$", 3)
        if algo != "pbkdf2_sha256":
            return False

        iterations = int(it_s)

        # restore padding for base64 strings
        pad1 = "=" * ((4 - len(salt_b64) % 4) % 4)
        pad2 = "=" * ((4 - len(dk_b64) % 4) % 4)
        salt = base64.urlsafe_b64decode((salt_b64 + pad1).encode("utf-8"))
        expected = base64.urlsafe_b64decode((dk_b64 + pad2).encode("utf-8"))

        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=len(expected))
        return hashlib.compare_digest(dk, expected)
    except Exception:
        return False
