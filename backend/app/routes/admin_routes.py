# backend/app/routes/admin_routes.py
from __future__ import annotations

"""
Admin routes (FastAPI router): διαχείριση οργανισμών και χρηστών.

Τι καλύπτει:
- /api/v1/admin/orgs (GET, POST)
  Λίστα οργανισμών και δημιουργία οργανισμού.
- /api/v1/admin/users (GET, POST)
  Λίστα χρηστών και δημιουργία χρήστη από Admin.
- /api/v1/admin/users/{username}/active (PATCH)
  Ενεργοποίηση/απενεργοποίηση λογαριασμού.
- /api/v1/admin/users/{username}/reset-password (POST)
  Reset password χρήστη.

RBAC:
- Όλα τα endpoints απαιτούν role = "Admin".
  Αυτό επιβάλλεται μέσω require_roles("Admin").
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

# Actor: token claims (username/role/org)
# hash_password: PBKDF2 password hashing (ισχυρότερο από sha256 scheme στο auth_routes.py)
# require_roles: dependency για RBAC
from app.auth import Actor, hash_password, require_roles

# Persistence layer (SQLite)
from app.services.sqlite_store import get_store

router = APIRouter()

# -------------------------
# Pydantic models (requests / responses)
# -------------------------
class OrgCreate(BaseModel):
    """Payload για δημιουργία οργανισμού."""
    org_name: str = Field(..., min_length=1)


class OrgOut(BaseModel):
    """Response οργανισμού."""
    org_name: str


class UserCreatePayload(BaseModel):
    """
    Payload για δημιουργία χρήστη από Admin.

    Σημείωση:
    - Το Admin μπορεί να δημιουργήσει user οποιουδήποτε ρόλου.
    - Παρέχεται “plain” password εδώ και αποθηκεύεται ως hash στο DB.
    """
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    org_name: str = Field(..., min_length=1)


class UserOut(BaseModel):
    """Public representation χρήστη που επιστρέφεται σε admin views."""
    username: str
    role: str
    org_name: str
    is_active: bool
    created_at: str


class ToggleActivePayload(BaseModel):
    """Payload για ενεργοποίηση/απενεργοποίηση χρήστη."""
    is_active: bool


class ResetPasswordPayload(BaseModel):
    """Payload για reset password."""
    new_password: str = Field(..., min_length=1)


# -------------------------
# Routes: Orgs
# -------------------------
@router.get("/orgs", response_model=List[OrgOut], dependencies=[Depends(require_roles("Admin"))])
def list_orgs(actor: Actor = Depends(require_roles("Admin"))):
    """
    Επιστρέφει λίστα οργανισμών.

    - Store: store.list_orgs()
    - RBAC: Admin only
    """
    store = get_store()
    return store.list_orgs()


@router.post("/orgs", response_model=OrgOut, dependencies=[Depends(require_roles("Admin"))])
def create_org(payload: OrgCreate, actor: Actor = Depends(require_roles("Admin"))):
    """
    Δημιουργεί οργανισμό.

    - Store: store.create_org(org_name)
    - RBAC: Admin only
    """
    store = get_store()
    store.create_org(payload.org_name)
    return {"org_name": payload.org_name}


# -------------------------
# Routes: Users
# -------------------------
@router.get("/users", response_model=List[UserOut], dependencies=[Depends(require_roles("Admin"))])
def list_users(actor: Actor = Depends(require_roles("Admin"))):
    """
    Επιστρέφει λίστα χρηστών.

    - Store: store.list_users()
    - RBAC: Admin only
    """
    store = get_store()
    return store.list_users()


@router.post("/users", response_model=UserOut, dependencies=[Depends(require_roles("Admin"))])
def create_user(payload: UserCreatePayload, actor: Actor = Depends(require_roles("Admin"))):
    """
    Δημιουργεί χρήστη από Admin.

    Validations:
    - Επιτρέπουμε μόνο τους 4 γνωστούς ρόλους: Admin, Hospital, Biobank, Researcher
    - Username uniqueness (store.get_user_by_username)

    Password hashing:
    - Χρησιμοποιεί app.auth.hash_password() (PBKDF2)
      και όχι το sha256(PASSWORD_SALT+pw) του public register.
    - Άρα αυτό το flow είναι “admin-managed accounts”.
    """
    store = get_store()

    # basic role guard
    if payload.role not in ("Admin", "Hospital", "Biobank", "Researcher"):
        raise HTTPException(status_code=400, detail="Invalid role")

    # avoid duplicates
    if store.get_user_by_username(payload.username) is not None:
        raise HTTPException(status_code=409, detail="Username already exists")

    # create in store with PBKDF2 hash
    user = store.create_user(
        username=payload.username,
        password_hash=hash_password(payload.password),
        role=payload.role,
        org_name=payload.org_name,
        is_active=True,
    )

    # normalize response types
    return {
        "username": user["username"],
        "role": user["role"],
        "org_name": user["org_name"],
        "is_active": bool(int(user["is_active"])),
        "created_at": user["created_at"],
    }


@router.patch("/users/{username}/active", response_model=UserOut, dependencies=[Depends(require_roles("Admin"))])
def set_user_active(username: str, payload: ToggleActivePayload, actor: Actor = Depends(require_roles("Admin"))):
    """
    Ενεργοποίηση/Απενεργοποίηση χρήστη.

    Ροή:
    - βρίσκουμε user
    - ενημερώνουμε is_active στο store
    - επιστρέφουμε updated record

    Αυτό επηρεάζει:
    - login (στο auth_routes.py) απορρίπτει inactive χρήστες
    - verify_token στο auth.py επίσης απορρίπτει token αν active=False στα claims
      (αν και τα claims “παγώνουν” στη στιγμή issuance — άρα /me με DB lookup
       θα ήταν καλύτερο αν θες instant revocation στο PoC).
    """
    store = get_store()
    user = store.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    updated = store.set_user_active(username, payload.is_active)
    return {
        "username": updated["username"],
        "role": updated["role"],
        "org_name": updated["org_name"],
        "is_active": bool(int(updated["is_active"])),
        "created_at": updated["created_at"],
    }


@router.post("/users/{username}/reset-password", response_model=Dict[str, Any], dependencies=[Depends(require_roles("Admin"))])
def reset_password(username: str, payload: ResetPasswordPayload, actor: Actor = Depends(require_roles("Admin"))):
    """
    Reset password χρήστη από Admin.

    - Store: store.set_user_password(username, new_password_hash)
    - Password hash: PBKDF2 (app.auth.hash_password)
    - RBAC: Admin only

    Response:
    - επιστρέφει status + username (χωρίς ευαίσθητες πληροφορίες).
    """
    store = get_store()
    user = store.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    store.set_user_password(username, hash_password(payload.new_password))
    return {"status": "ok", "username": username}
