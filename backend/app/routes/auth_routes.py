# backend/app/routes/auth_routes.py
from __future__ import annotations

"""
Authentication routes (FastAPI router): register + login.

Τι καλύπτει:
- /api/v1/auth/register
  Δημιουργεί χρήστη στο store (SQLite) με role/org, με απλούς κανόνες invite codes.
- /api/v1/auth/login
  Κάνει έλεγχο credentials και επιστρέφει token + user info.
- /api/v1/auth/me
  Είναι σκόπιμα απενεργοποιημένο (405) στο PoC.

Σημαντικές παρατηρήσεις για την υλοποίηση:
1) Invite code logic:
   - Researcher: μπορεί να κάνει register χωρίς invite code
   - Hospital/Biobank: απαιτούν invite codes από env
2) Password hashing:
   - Εδώ χρησιμοποιείται _hash_password() = sha256(PASSWORD_SALT + password).
   - Αυτό είναι “PoC/simple scheme” και διαφέρει από το PBKDF2 scheme στο app.auth.hash_password().
   - Δεν αλλάζουμε τίποτα τώρα, απλώς το τεκμηριώνουμε για να ξέρεις γιατί υπάρχουν δύο διαφορετικοί μηχανισμοί.
3) Token issuance:
   - Το token εκδίδεται από app.auth.issue_token(actor)
   - Το token περιέχει role/org, άρα το backend κάνει RBAC/scoping χωρίς να ρωτάει DB σε κάθε request.
"""

import hashlib
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import Actor, issue_token, ROLE_BIOBANK, ROLE_HOSPITAL, ROLE_RESEARCHER
from app.services.sqlite_store import get_store

router = APIRouter()

# -------------------------
# Password hashing (PoC)
# -------------------------
# PASSWORD_SALT: environment salt που μπαίνει πριν το password ώστε να μη γίνεται “straight sha256(password)”.
# Για PoC είναι ΟΚ, αλλά δεν είναι τόσο ισχυρό όσο PBKDF2.
PASSWORD_SALT = os.getenv("PASSWORD_SALT", "dev-salt-change-me")

# -------------------------
# Invite codes (PoC)
# -------------------------
# Για να περιορίζεις ποιος μπορεί να δημιουργήσει Hospital/Biobank λογαριασμό.
HOSPITAL_INVITE_CODE = os.getenv("HOSPITAL_INVITE_CODE", "HOSPITAL2026")
BIOBANK_INVITE_CODE = os.getenv("BIOBANK_INVITE_CODE", "BIOBANK2026")


def _hash_password(pw: str) -> str:
    """
    Simple password hash (PoC):
      sha256(PASSWORD_SALT + password)

    Χρησιμοποιείται μόνο στο flow register/login αυτού του router.
    """
    x = (PASSWORD_SALT + pw).encode("utf-8")
    return hashlib.sha256(x).hexdigest()


# -------------------------
# Pydantic request/response models
# -------------------------
class RegisterIn(BaseModel):
    """
    Payload για /register.
    """
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)

    # role expected: "Hospital" | "Researcher" | "Biobank"
    role: str = Field(..., description="Hospital | Researcher | Biobank")
    org: str = Field(..., min_length=2, max_length=120)

    # optional invite code (required depending on role)
    invite_code: Optional[str] = Field(default=None)


class LoginIn(BaseModel):
    """
    Payload για /login.
    """
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserOut(BaseModel):
    """
    Public user representation που επιστρέφεται στο UI.
    (Δεν περιέχει password_hash.)
    """
    username: str
    role: str
    org: str
    is_active: bool
    created_at: datetime


class TokenOut(BaseModel):
    """
    Response για /login.
    Περιέχει token και embedded user object.
    """
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# -------------------------
# Routes
# -------------------------
@router.post("/register", response_model=UserOut)
def register(payload: RegisterIn):
    """
    Public register endpoint.

    Κανόνες:
    - Researcher: allowed χωρίς invite_code
    - Hospital: απαιτεί invite_code == HOSPITAL_INVITE_CODE
    - Biobank: απαιτεί invite_code == BIOBANK_INVITE_CODE

    Αποθηκεύει νέο χρήστη στο SQLite store (users table) μέσω store.create_user().
    """
    role = payload.role.strip()
    org = payload.org.strip()
    username = payload.username.strip()

    # --- role gating via invite code ---
    if role == ROLE_HOSPITAL:
        if (payload.invite_code or "").strip() != HOSPITAL_INVITE_CODE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid hospital invite code")
    elif role == ROLE_BIOBANK:
        if (payload.invite_code or "").strip() != BIOBANK_INVITE_CODE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid biobank invite code")
    elif role == ROLE_RESEARCHER:
        # no invite code required
        pass
    else:
        # reject unknown role strings
        raise HTTPException(status_code=400, detail="Invalid role")

    # store access
    store = get_store()

    # hash password (PoC scheme)
    pw_hash = _hash_password(payload.password)

    # create user record in sqlite
    created = store.create_user(
        username=username,
        password_hash=pw_hash,
        role=role,
        org=org,
    )
    return created


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn):
    """
    Login endpoint.

    Ροή:
    1) Αναζητούμε user στο store
    2) Ελέγχουμε is_active (αν είναι disabled, reject)
    3) Ελέγχουμε password hash
    4) Δημιουργούμε Actor από τα στοιχεία του user
    5) Εκδίδουμε token με issue_token(actor)
    6) Επιστρέφουμε token + user πληροφορίες
    """
    store = get_store()

    # Επιστρέφει dict με keys: username, role, org, password_hash, is_active, created_at
    u = store.get_user(payload.username.strip())

    # Για λόγους ασφάλειας δεν ξεχωρίζουμε “δεν υπάρχει” vs “λάθος κωδικός”
    if not u or not u["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # password check
    if u["password_hash"] != _hash_password(payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Actor object (χρησιμοποιείται σε RBAC στο υπόλοιπο API)
    actor = Actor(username=u["username"], role=u["role"], org=u["org"], is_active=True)

    # Issue signed token
    token = issue_token(actor)

    # Return token + user info
    return TokenOut(
        access_token=token,
        user=UserOut(
            username=u["username"],
            role=u["role"],
            org=u["org"],
            is_active=u["is_active"],
            created_at=u["created_at"],
        ),
    )


@router.get("/me", response_model=UserOut)
def me():
    """
    Αυτό το endpoint είναι intentionally disabled (405).

    Γιατί:
    - Το Streamlit UI δεν το χρειάζεται απαραίτητα.
    - Το UI μπορεί να βασιστεί στα token claims (role/org/username) για display.
    - Αν θες μελλοντικά, μπορεί να υλοποιηθεί σωστά με Depends(get_actor)
      και store lookup για freshness (π.χ. is_active changes).
    """
    raise HTTPException(status_code=405, detail="Use token claims in frontend (or implement /me with Depends(get_actor)).")
