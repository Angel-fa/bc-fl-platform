# backend/app/auth.py
from __future__ import annotations

"""
Auth module της πλατφόρμας BC-FL.

Σκοπός:
- Υλοποιεί μηχανισμό authentication/authorization με:
  1) lightweight “JWT-like” token (δεν είναι πλήρες JWT library, αλλά ίδια λογική)
  2) RBAC (Role-Based Access Control) με FastAPI dependencies
  3) password hashing helpers (PBKDF2) για Admin-managed users (admin_routes.py)

Πού χρησιμοποιείται:
- Το Streamlit UI παίρνει token από /api/v1/auth/login (auth_routes.py)
- Μετά στέλνει Authorization: Bearer <token> σε όλα τα protected endpoints στο routes.py
- Στο routes.py χρησιμοποιούμε Depends(require_roles(...)) για να προστατεύουμε endpoints

Σημείωση:
- Υπάρχουν 2 “σχολές” hashing στο project:
  A) auth_routes.py: uses _hash_password (sha256 με salt env PASSWORD_SALT) -> απλό/PoC
  B) admin_routes.py: uses hash_password() (PBKDF2) -> πιο “σωστό” security-wise
  Αυτό δεν το αλλάζουμε τώρα (όπως ζήτησες), απλώς το τεκμηριώνουμε.
"""

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from fastapi import Depends, HTTPException, Request, status

# -------------------------
# Roles (RBAC)
# -------------------------
# Οι ρόλοι αυτοί πρέπει να ταιριάζουν με:
# - Streamlit UI επιλογές ρόλων
# - auth_routes.py register/login logic
# - require_roles(...) checks σε routes.py
ROLE_HOSPITAL = "Hospital"
ROLE_RESEARCHER = "Researcher"
ROLE_BIOBANK = "Biobank"
ROLE_ADMIN = "Admin"

# Set με όλους τους επιτρεπόμενους ρόλους (χρήσιμο σε validation token claims)
ALL_ROLES = {ROLE_HOSPITAL, ROLE_RESEARCHER, ROLE_BIOBANK, ROLE_ADMIN}

# -------------------------
# Token config
# -------------------------
# AUTH_SECRET: shared secret για υπογραφή token (HMAC-SHA256).
# Αν αλλάξει, όλα τα tokens που έχουν εκδοθεί παύουν να είναι έγκυρα.
AUTH_SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me")

# Token TTL σε seconds (default 24h).
# Χρησιμοποιείται για exp (expiration claim).
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "86400"))  # 24h default


@dataclass(frozen=True)
class Actor:
    """
    Actor = ο “ταυτοποιημένος χρήστης” που προκύπτει από το token.

    Περιέχει μόνο ό,τι χρειάζεται για authorization/scoping:
    - username: μοναδικό identifier (στην SQLite users table)
    - role: RBAC role
    - org: οργανισμός (hospital/biobank/research group)
    - is_active: flag (αν user είναι απενεργοποιημένος, απορρίπτεται στο verify_token)
    """
    username: str
    role: str
    org: str
    is_active: bool = True


# -------------------------
# Base64 URL-safe encoding helpers
# -------------------------
def _b64url_encode(data: bytes) -> str:
    """
    URL-safe base64 encoding χωρίς '=' padding.
    Χρησιμοποιείται για payload και signature.
    """
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    """
    URL-safe base64 decoding.
    Επειδή αφαιρούμε '=' στο encode, χρειάζεται να το επαναφέρουμε (padding).
    """
    pad = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


# -------------------------
# Token signing helpers
# -------------------------
def _sign(payload_b64: str) -> str:
    """
    Δημιουργεί signature (HMAC-SHA256) πάνω στο base64 payload string.
    Επιστρέφει signature base64url.
    """
    sig = hmac.new(AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(sig)


def issue_token(actor: Actor) -> str:
    """
    Εκδίδει token για έναν Actor.

    Token format:
      <payload_b64>.<sig_b64>

    payload fields:
    - sub: subject (username)
    - role, org: claims που χρησιμοποιούνται σε RBAC/scoping
    - iat: issued at (unix time)
    - exp: expiration (iat + TTL)
    - active: bool (για revoke/disable user)

    Σημείωση:
    - Δεν είναι full JWT standard, αλλά η λογική είναι JWT-like.
    """
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


def verify_token(token: str) -> Actor:
    """
    Επαληθεύει token και επιστρέφει Actor.

    Βήματα:
    1) split σε payload_b64 και sig
    2) verify signature (constant-time compare)
    3) decode payload JSON
    4) έλεγχος exp (expiry)
    5) validation claims:
       - username, org non-empty
       - role ∈ ALL_ROLES
       - active == True

    Σε αποτυχία:
    - 401 Unauthorized για invalid token/signature/claims/expired
    - 403 Forbidden για inactive user
    """
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

    return Actor(username=username, role=role, org=org, is_active=True)


def get_actor(request: Request) -> Actor:
    """
    Extracts Actor από HTTP request.

    Περιμένει header:
      Authorization: Bearer <token>

    Αν λείπει ή δεν είναι Bearer:
    - 401 Missing bearer token
    """
    auth = request.headers.get("Authorization", "").strip()
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    return verify_token(token)


def require_roles(*roles: str):
    """
    FastAPI dependency generator για RBAC.

    Χρήση:
      @router.get("/datasets")
      def list_datasets(actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK))):
          ...

    Όταν το endpoint καλείται:
    - FastAPI εκτελεί get_actor() -> verify_token()
    - παίρνει Actor
    - αν actor.role ∉ allowed roles -> 403 Forbidden
    """
    allowed: Sequence[str] = roles

    def _dep(actor: Actor = Depends(get_actor)) -> Actor:
        if actor.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden for role '{actor.role}'",
            )
        return actor

    return _dep


# -------------------------------------------------------------------
# Password hashing helpers (PBKDF2)
# -------------------------------------------------------------------
# Σημείωση:
# Αυτές οι συναρτήσεις χρησιμοποιούνται από admin_routes.py για users που δημιουργεί Admin.
# Είναι ισχυρότερες από το sha256+salt scheme του auth_routes.py register.
# Δεν αλλάζουμε τη λογική τώρα, απλώς το εξηγούμε.
import base64
import hashlib
import os


def hash_password(password: str) -> str:
    """
    Δημιουργεί PBKDF2-SHA256 hash.

    Format που αποθηκεύεται:
      pbkdf2_sha256$iterations$salt_b64$dk_b64

    - iterations: από env PWD_ITERATIONS (default 210000)
    - salt: random 16 bytes
    - dk (derived key): 32 bytes

    Αυτό το format είναι self-contained:
    - περιέχει iterations + salt, ώστε verify_password να μπορεί να κάνει re-derivation.
    """
    if not password or len(password) < 4:
        raise ValueError("Password too short")

    iterations = int(os.getenv("PWD_ITERATIONS", "210000"))
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=32)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("utf-8").rstrip("=")
    dk_b64 = base64.urlsafe_b64encode(dk).decode("utf-8").rstrip("=")
    return f"pbkdf2_sha256${iterations}${salt_b64}${dk_b64}"


def verify_password(password: str, stored: str) -> bool:
    """
    Επαληθεύει password έναντι stored PBKDF2 hash string.

    Βήματα:
    - parse stored format
    - decode salt και expected dk
    - recompute dk με ίδιες iterations και salt
    - constant-time compare

    Επιστρέφει:
    - True αν ταιριάζει
    - False αν όχι ή αν υπάρχει parsing error
    """
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
