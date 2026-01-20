# backend/app/routes/patient_consent_routes.py
from __future__ import annotations

from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.auth import Actor
from app.services.blockchain_service import get_blockchain, sha256_hex_str


# NOTE (πολύ σημαντικό για routing):
# Αυτό το router γίνεται include από το backend/app/api/routes.py,
# το οποίο ήδη έχει prefix="/api/v1".
#
# Άρα εδώ κρατάμε prefix="" για να ΜΗΝ έχουμε διπλό /api/v1/api/v1.
# Τελικές διαδρομές (όταν γίνει include στο api/routes.py):
# - POST /api/v1/public/consent
# - POST /api/v1/consents/has
router = APIRouter(prefix="", tags=["patient-consent"])


# -------------------------
# Request schemas (Pydantic)
# -------------------------
# Ορίζουμε τα input payloads που θα δέχεται το API, με validation.
# Αυτό βοηθάει:
# - να μην περνάνε κενά πεδία
# - να είναι σαφές τι σχήμα δεδομένων περιμένουμε
# - να έχεις “αυτόματη” τεκμηρίωση στο OpenAPI/Swagger


class PublicConsentRequest(BaseModel):
    """
    Payload για δημόσια υποβολή συγκατάθεσης (public portal / PoC).

    Σημείωση:
    - Αυτό είναι PoC public endpoint.
    - Το προστατεύουμε με shared secret (PATIENT_PORTAL_SECRET),
      ώστε να μην είναι πλήρως ανοιχτό.
    """
    dataset_id: str = Field(..., min_length=1)
    # patient_id: εδώ είναι raw identifier/pseudonymous string που στέλνει το portal.
    # Δεν το γράφουμε αυτούσιο on-chain. Το κάνουμε hash -> bytes32 key.
    patient_id: str = Field(..., min_length=1)
    # decision: επιτρέπουμε μόνο allow ή deny (regex pattern).
    decision: str = Field(..., pattern="^(allow|deny)$")
    # secret: shared secret για να αποτρέπουμε public misuse
    secret: Optional[str] = None


class ConsentCheckRequest(BaseModel):
    """
    Payload για έλεγχο αν υπάρχει συγκατάθεση για patient+dataset.
    Δεν απαιτεί “secret”, γιατί είναι read check (ανάλογα με το PoC design).
    """
    dataset_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)


# -------------------------
# Helper: shared secret check
# -------------------------
def _portal_secret_ok(secret: Optional[str]) -> bool:
    """
    Ελέγχει αν το secret που ήρθε στο request ταιριάζει με το PATIENT_PORTAL_SECRET.

    Σημαντική συμπεριφορά:
    - Αν το PATIENT_PORTAL_SECRET δεν είναι ρυθμισμένο (κενό),
      επιστρέφει False και ΔΕΝ επιτρέπει public writes.
      Αυτό είναι “secure by default”.
    """
    import os

    expected = (os.getenv("PATIENT_PORTAL_SECRET") or "").strip()
    if not expected:
        # Αν δεν έχει ρυθμιστεί env var, κλείνουμε το public write endpoint.
        return False
    return (secret or "").strip() == expected


# -------------------------
# Public endpoint: set consent
# -------------------------
@router.post("/public/consent")
def public_set_consent(req: PublicConsentRequest) -> Dict[str, Any]:
    """
    Δημόσιο endpoint (PoC) για υποβολή patient consent.

    Ροή:
    1) Ελέγχουμε shared secret (PATIENT_PORTAL_SECRET)
    2) Παίρνουμε BlockchainService (είτε Noop είτε Web3, ανάλογα με BC_ENABLED)
    3) Υπολογίζουμε patient_key_hex = sha256(patient_id)
       - έτσι αποφεύγουμε να αποθηκεύουμε raw patient IDs on-chain
       - παίρνουμε 64 hex chars (bytes32)
    4) Μετατρέπουμε decision -> allowed boolean
    5) Δημιουργούμε “Actor” για audit trail (χωρίς login, PoC)
    6) Καλούμε bc.set_patient_consent(...)
       - αν είμαστε σε Web3 mode, γράφει στο Consent contract + anchor receipt
       - αν είμαστε σε Noop mode, κρατά in-memory consent + γράφει receipt στη sqlite

    Επιστροφή:
    - dict (π.χ. ok/mode/dataset_id/patient_key/allowed/tx_hash κ.λπ.),
      όπως το επιστρέφει το BlockchainService.
    """
    # 1) Guard: shared secret
    if not _portal_secret_ok(req.secret):
        raise HTTPException(status_code=403, detail="Invalid patient portal secret")

    # 2) Resolve blockchain service (Noop ή Web3)
    bc = get_blockchain()

    # 3) Derive bytes32 key from patient_id (GDPR-friendly: no raw IDs on-chain)
    # sha256_hex_str επιστρέφει 64 hex chars χωρίς "0x"
    patient_key_hex = sha256_hex_str(req.patient_id)
    allowed = req.decision == "allow"

    # 4) PoC actor for audit (χωρίς login)
    # Προσοχή: εδώ το username βάζει raw patient_id μέσα στο string.
    # Δεν πάει on-chain ως raw field (γίνεται hash στο receipt anchoring),
    # αλλά το manifest μπορεί να αποθηκεύεται off-chain στο sqlite (ανάλογα με mode).
    # Αν ποτέ το κάνεις production, θα το κάνεις πιο “privacy safe”.
    actor = Actor(
        username=f"patient:{req.patient_id}",
        role="Patient",
        org="PublicPortal",
        is_active=True
    )

    # 5) Persist consent μέσω blockchain service (contract ή noop)
    out = bc.set_patient_consent(
        dataset_id=req.dataset_id,
        patient_key_hex=patient_key_hex,
        allowed=allowed,
        actor=actor,
    )
    return out


# -------------------------
# Read endpoint: check consent
# -------------------------
@router.post("/consents/has")
def has_consent(req: ConsentCheckRequest) -> Dict[str, Any]:
    """
    Ελέγχει αν υπάρχει consent για (dataset_id, patient_id).

    Ροή:
    1) bc = get_blockchain()
    2) patient_key_hex = sha256(patient_id)
    3) ok = bc.has_patient_consent(...)
       - σε Web3 mode: call στο contract (view function)
       - σε Noop mode: lookup στο in-memory map

    Επιστρέφει:
    - ok: True (δηλώνει “endpoint executed normally”)
    - dataset_id: echo
    - patient_key: 0x + patient_key_hex (για να βλέπεις το derived key)
    - has_consent: True/False
    """
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
