from __future__ import annotations
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import Actor, hash_password, require_roles

from app.services.sqlite_store import get_store

router = APIRouter()


# Pydantic models (requests / responses)

class OrgCreate(BaseModel):
    org_name: str = Field(..., min_length=1)

class OrgOut(BaseModel):
    org_name: str

class UserCreatePayload(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    org_name: str = Field(..., min_length=1)


class UserOut(BaseModel):
    username: str
    role: str
    org_name: str
    is_active: bool
    created_at: str


class ToggleActivePayload(BaseModel):
    is_active: bool


class ResetPasswordPayload(BaseModel):
    new_password: str = Field(..., min_length=1)


# Routes: Orgs

@router.get("/orgs", response_model=List[OrgOut], dependencies=[Depends(require_roles("Admin"))])
def list_orgs(actor: Actor = Depends(require_roles("Admin"))):
    store = get_store()
    return store.list_orgs()


@router.post("/orgs", response_model=OrgOut, dependencies=[Depends(require_roles("Admin"))])
def create_org(payload: OrgCreate, actor: Actor = Depends(require_roles("Admin"))):
    store = get_store()
    store.create_org(payload.org_name)
    return {"org_name": payload.org_name}



# Routes: Users

@router.get("/users", response_model=List[UserOut], dependencies=[Depends(require_roles("Admin"))])
def list_users(actor: Actor = Depends(require_roles("Admin"))):
    store = get_store()
    return store.list_users()


@router.post("/users", response_model=UserOut, dependencies=[Depends(require_roles("Admin"))])
def create_user(payload: UserCreatePayload, actor: Actor = Depends(require_roles("Admin"))):
    store = get_store()


    if payload.role not in ("Admin", "Hospital", "Biobank", "Researcher"):
        raise HTTPException(status_code=400, detail="Invalid role")

    # avoid duplicates
    if store.get_user_by_username(payload.username) is not None:
        raise HTTPException(status_code=409, detail="Username already exists")

    user = store.create_user(
        username=payload.username,
        password_hash=hash_password(payload.password),
        role=payload.role,
        org_name=payload.org_name,
        is_active=True,
    )

    return {
        "username": user["username"],
        "role": user["role"],
        "org_name": user["org_name"],
        "is_active": bool(int(user["is_active"])),
        "created_at": user["created_at"],
    }


@router.patch("/users/{username}/active", response_model=UserOut, dependencies=[Depends(require_roles("Admin"))])
def set_user_active(username: str, payload: ToggleActivePayload, actor: Actor = Depends(require_roles("Admin"))):
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
    store = get_store()
    user = store.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    store.set_user_password(username, hash_password(payload.new_password))
    return {"status": "ok", "username": username}
