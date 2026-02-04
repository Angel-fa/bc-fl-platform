from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
from app.services.sqlite_store import get_store

from app.auth import (
    Actor,
    issue_token,
    require_roles,
    ROLE_BIOBANK,
    ROLE_HOSPITAL,
    ROLE_RESEARCHER,
    ROLE_ADMIN
)

router = APIRouter()


PASSWORD_SALT = os.getenv("PASSWORD_SALT", "dev-salt-change-me")


# Invite codes

HOSPITAL_INVITE_CODE = os.getenv("HOSPITAL_INVITE_CODE", "HOSPITAL2026")
BIOBANK_INVITE_CODE = os.getenv("BIOBANK_INVITE_CODE", "BIOBANK2026")
ADMIN_INVITE_CODE  = os.getenv("ADMIN_INVITE_CODE", "ADMIN-security")

def _hash_password(pw: str) -> str:
    x = (PASSWORD_SALT + pw).encode("utf-8")
    return hashlib.sha256(x).hexdigest()


# models

class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=200)

    role: str = Field(..., description="Hospital | Researcher | Biobank")
    org: str = Field(..., min_length=2, max_length=120)

    invite_code: Optional[str] = Field(default=None)


class LoginIn(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserOut(BaseModel):
    username: str
    role: str
    org: str
    is_active: bool
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# Routes

@router.post("/register", response_model=UserOut)  # Αποθηκεύει νέο χρήστη στο SQLite
def register(payload: RegisterIn):

    role = payload.role.strip()
    org = payload.org.strip()
    username = payload.username.strip()

    if role == ROLE_HOSPITAL:
        if (payload.invite_code or "").strip() != HOSPITAL_INVITE_CODE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid hospital invite code")
    elif role == ROLE_BIOBANK:
        if (payload.invite_code or "").strip() != BIOBANK_INVITE_CODE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid biobank invite code")
    elif role == ROLE_ADMIN:
        if (payload.invite_code or "").strip() != ADMIN_INVITE_CODE:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid biobank invite code")
    elif role == ROLE_RESEARCHER:
        pass
    else:
        raise HTTPException(status_code=400, detail="Invalid role")

    # store access
    store = get_store()

    # hash password
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

    store = get_store()

    u = store.get_user(payload.username.strip())

    if not u or not u["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    if u["password_hash"] != _hash_password(payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    actor = Actor(username=u["username"], role=u["role"], org=u["org"], is_active=True)

    token = issue_token(actor)

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
def me(actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER, ROLE_ADMIN))):

    store = get_store()

    u = store.get_user(actor.username.strip())

    if not u:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not u["is_active"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User is inactive")

    return UserOut(
        username=u["username"],
        role=u["role"],
        org=u["org"],
        is_active=bool(u["is_active"]),
        created_at=u["created_at"],
    )