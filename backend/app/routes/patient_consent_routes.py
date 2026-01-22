from __future__ import annotations

import os

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.auth import Actor
from app.services.blockchain_service import get_blockchain, sha256_hex_str


router = APIRouter(prefix="", tags=["patient-consent"])


class PublicConsentRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)

    patient_id: str = Field(..., min_length=1)

    decision: str = Field(..., pattern="^(allow|deny)$")

    secret: Optional[str] = None


class ConsentCheckRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)


# Helper -> secret check

def _portal_secret_ok(secret: Optional[str]) -> bool:   #   Ελέγχει αν το secret που ήρθε στο request ταιριάζει με το PATIENT_PORTAL_SECRET

    expected = (os.getenv("PATIENT_PORTAL_SECRET") or "").strip()
    if not expected:
        return False
    return (secret or "").strip() == expected


# Public endpoint: set consent

@router.post("/public/consent")
def public_set_consent(req: PublicConsentRequest) -> Dict[str, Any]:

    if not _portal_secret_ok(req.secret):
        raise HTTPException(status_code=403, detail="Invalid patient portal secret")

    bc = get_blockchain()

    patient_key_hex = sha256_hex_str(req.patient_id)
    allowed = req.decision == "allow"

    actor = Actor(
        username=f"patient:{req.patient_id}",
        role="Patient",
        org="PublicPortal",
        is_active=True
    )

    out = bc.set_patient_consent(
        dataset_id=req.dataset_id,
        patient_key_hex=patient_key_hex,
        allowed=allowed,
        actor=actor,
    )
    return out


#  check consent

@router.post("/consents/has")  # Ελέγχει αν υπάρχει consent για (dataset_id, patient_id)
def has_consent(req: ConsentCheckRequest) -> Dict[str, Any]:
    bc = get_blockchain()
    patient_key_hex = sha256_hex_str(req.patient_id)

    ok = bc.has_patient_consent(
        dataset_id=req.dataset_id,
        patient_key_hex=patient_key_hex
    )

    return {
        "ok": True,
        "dataset_id": req.dataset_id,
        "patient_key": "0x" + patient_key_hex,
        "has_consent": bool(ok)
    }
