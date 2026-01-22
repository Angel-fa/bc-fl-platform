from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from app.routes.auth_routes import router as auth_router
from app.routes.admin_routes import router as admin_router
from app.routes.patient_consent_routes import router as patient_consent_router
from app.services.blockchain_service import get_blockchain, sha256_hex
from app.services.sqlite_store import get_store

from app.auth import Actor, require_roles, ROLE_BIOBANK, ROLE_HOSPITAL, ROLE_RESEARCHER

from app.schemas.domain import (
    AccessRequest,
    AccessRequestCreate,
    AuditLog,
    ConsentPolicy,
    ConsentPolicyCreate,
    ConsentStatus,
    Dataset,
    DatasetCreate,
    DatasetFeaturesUpdate,
    DescriptorStatus,
    FLJob,
    FLJobCreate,
    FLJobStatus,
    Node,
    NodeRegister,
    NodeStatus,
    RequestStatus,
    Run,
    RunCreate,
)


router = APIRouter(prefix="/api/v1")

router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(admin_router, prefix="/admin", tags=["admin"])
router.include_router(patient_consent_router, tags=["patient-consent"])


# Debug

@router.get("/debug/version", tags=["core"])
def debug_version():
    return {"routes_version": "routes-2026-01-11-fljobs-hospital-enabled"}


# Config

AGENT_REG_SECRET = os.getenv("AGENT_REG_SECRET", "dev-secret")

AGENT_CALL_TIMEOUT = float(os.getenv("AGENT_CALL_TIMEOUT", "8"))


def _http_json_post(url: str, payload: dict, headers: Optional[dict] = None, timeout: float = 8.0) -> dict:

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


@router.get("/health", tags=["core"])
def health():
    return {"status": "ok"}



# Helpers (scoping)

def _dataset_owned_by_org(ds: Dataset, org: str) -> bool:
    return (ds.owner_org or "").strip().lower() == (org or "").strip().lower()


def _ensure_dataset_visible(actor: Actor, ds: Dataset) -> None:
    if actor.role == ROLE_HOSPITAL:
        if not _dataset_owned_by_org(ds, actor.org):
            raise HTTPException(status_code=403, detail="Dataset not in your organization scope")


def _ensure_dataset_writable(actor: Actor, ds: Dataset) -> None:
    if actor.role != ROLE_HOSPITAL:
        raise HTTPException(status_code=403, detail="Only Hospital can modify this resource")
    if not _dataset_owned_by_org(ds, actor.org):
        raise HTTPException(status_code=403, detail="Dataset not in your organization scope")


def _same_org(a: str, b: str) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def _sanitize_dataset_for_actor(actor: Actor, ds: Dataset) -> Dataset:
    if actor.role in (ROLE_BIOBANK, ROLE_RESEARCHER): # δεν βλέπουν raw columns list-Θα βλέπουν μόνο exposed_features (που έχει εγκρίνει το Hospital).
        ds.columns = None
    return ds



# Nodes

@router.post("/nodes/register", response_model=Node, tags=["federation"])
def register_node(payload: NodeRegister, secret: str = Query(..., description="AGENT_REG_SECRET")):
    if secret != AGENT_REG_SECRET:
        raise HTTPException(status_code=403, detail="Invalid registration secret")
    store = get_store()
    return store.register_node(payload, actor="agent")


@router.get("/nodes", response_model=List[Node], tags=["federation"])
def list_nodes(actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER))):
    store = get_store()
    return store.list_nodes()


@router.patch("/nodes/{node_id}/heartbeat", response_model=Node, tags=["federation"])
def heartbeat_node(node_id: UUID, secret: str = Query(...), status: NodeStatus = Query(default=NodeStatus.online)):
    if secret != AGENT_REG_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    store = get_store()
    try:
        return store.heartbeat_node(node_id, status=status, actor="agent")
    except KeyError:
        raise HTTPException(status_code=404, detail="Node not found")



# Datasets (Federated descriptors)

@router.post("/datasets", response_model=Dataset, tags=["core"])
def create_dataset(payload: DatasetCreate, actor: Actor = Depends(require_roles(ROLE_HOSPITAL))):
    store = get_store()

    fixed = DatasetCreate(
        name=payload.name,
        description=payload.description,
        owner_org=actor.org,
        sensitivity_level=payload.sensitivity_level,
        schema_id=payload.schema_id,
        local_uri=payload.local_uri,
        node_id=payload.node_id,
    )

    node = store.get_node(fixed.node_id)     # το dataset πρέπει να υπάρχει σε υπαρκτό node
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    if not _same_org(node.org, actor.org):
        raise HTTPException(status_code=403, detail="Node org must match your hospital org")  # το dataset node org πρέπει να ταιριάζει με hospital org

    ds = store.create_dataset(fixed, actor=actor.username)

    bc = get_blockchain()
    bc.anchor(
        event_type="DATASET_REGISTERED",
        ref_id=str(ds.dataset_id),
        payload={
            "dataset_id": str(ds.dataset_id),
            "owner_org": ds.owner_org,
            "node_id": str(ds.node_id),
            "schema_id": ds.schema_id,
            "sensitivity_level": str(ds.sensitivity_level),
            "status": str(ds.status),
            "created_at": str(ds.created_at),
        },
        actor=actor,
    )
    return ds


@router.get("/datasets", response_model=List[Dataset], tags=["core"])
def list_datasets(actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER))):
    store = get_store()
    items = store.list_datasets()

    if actor.role == ROLE_HOSPITAL:
        return [d for d in items if _dataset_owned_by_org(d, actor.org)]

    return [_sanitize_dataset_for_actor(actor, d) for d in items]


@router.get("/datasets/{dataset_id}", response_model=Dataset, tags=["core"])
def get_dataset(dataset_id: UUID, actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER))): #visible
    store = get_store() #προσβαση στο dataset
    ds = store.get_dataset(dataset_id) # φόρτωση dataset
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _ensure_dataset_visible(actor, ds)
    return _sanitize_dataset_for_actor(actor, ds)


@router.post("/datasets/{dataset_id}/validate", response_model=Dataset, tags=["federation"])
def validate_dataset(dataset_id: UUID, actor: Actor = Depends(require_roles(ROLE_HOSPITAL))):
    store = get_store()  #προσβαση στο dataset
    ds = store.get_dataset(dataset_id) # Το dataset είναι συνδεδεμένο με node_id
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _ensure_dataset_writable(actor, ds)

    node = store.get_node(ds.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    try:
        url = f"{node.base_url.rstrip('/')}/validate"
        report = _http_json_post(
            url,
            payload={"local_uri": ds.local_uri, "schema_id": ds.schema_id},
            headers={"X-Agent-Secret": AGENT_REG_SECRET},
            timeout=AGENT_CALL_TIMEOUT,
        )

        ok = bool(report.get("ok", False))
        status = DescriptorStatus.validated if ok else DescriptorStatus.unavailable
        row_count = int(report.get("row_count", 0)) if report.get("row_count") is not None else None
        cols = report.get("columns")

        updated = store.update_dataset_validation(
            dataset_id,
            status=status,
            row_count=row_count,
            columns=cols,
            report=report,
            actor=actor.username,
        )


        if (getattr(updated, "exposed_features", None) is None) and updated.columns:
            updated = store.update_dataset_exposed_features(
                dataset_id=updated.dataset_id,
                exposed_features=updated.columns,
                actor=actor.username,
            )

        #  blockchain -> αποθηκεύει (hash) validation event
        bc = get_blockchain()
        bc.anchor(
            event_type="DATASET_VALIDATED",
            ref_id=str(updated.dataset_id),
            payload={
                "dataset_id": str(updated.dataset_id),
                "status": str(updated.status),
                "row_count": updated.row_count,
                "columns": updated.columns,
                "validation_report_hash": sha256_hex(report or {}),
                "node_id": str(updated.node_id),
            },
            actor=actor,
        )
        return updated

    except Exception as e:
        updated = store.update_dataset_validation(
            dataset_id,
            status=DescriptorStatus.unavailable,
            row_count=None,
            columns=None,
            report={"ok": False, "error": str(e)},
            actor=actor.username,
        )
        bc = get_blockchain()
        bc.anchor(
            event_type="DATASET_VALIDATION_FAILED",
            ref_id=str(updated.dataset_id),
            payload={
                "dataset_id": str(updated.dataset_id),
                "status": str(updated.status),
                "error_hash": sha256_hex({"error": str(e)}),
                "node_id": str(updated.node_id),
            },
            actor=actor,
        )
        return updated


@router.patch("/datasets/{dataset_id}/features", response_model=Dataset, tags=["core"])
def set_dataset_features(                  # set features exposed
    dataset_id: UUID,
    payload: DatasetFeaturesUpdate,
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL)),
):
    store = get_store()
    ds = store.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    _ensure_dataset_writable(actor, ds)

    cols = ds.columns or []
    requested = payload.exposed_features or []

    if cols:
        invalid = [f for f in requested if f not in cols]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Invalid features (not in columns): {invalid}")

    updated = store.update_dataset_exposed_features(dataset_id, requested, actor=actor.username)

    bc = get_blockchain()
    bc.anchor(
        event_type="DATASET_FEATURES_EXPOSED",
        ref_id=str(updated.dataset_id),
        payload={
            "dataset_id": str(updated.dataset_id),
            "owner_org": updated.owner_org,
            "exposed_features": requested,
            "count": len(requested),
        },
        actor=actor,
    )
    return updated



# Consent Policy

@router.post("/consents", response_model=ConsentPolicy, tags=["core"])
def create_consent(payload: ConsentPolicyCreate, actor: Actor = Depends(require_roles(ROLE_HOSPITAL))):
    store = get_store()
    ds = store.get_dataset(payload.dataset_id) # upload
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found for consent policy")
    _ensure_dataset_writable(actor, ds) # έλεγχος authrization (org)

    cp = store.create_consent_policy(payload, actor=actor.username)

    bc = get_blockchain()
    bc.anchor(
        event_type="CONSENT_POLICY_SET",
        ref_id=str(cp.policy_id),
        payload={
            "policy_id": str(cp.policy_id),
            "dataset_id": str(cp.dataset_id),
            "status": str(cp.status),
            "allow_external": bool(cp.allow_external),
            "allowed_roles": cp.allowed_roles,
            "export_methods": cp.export_methods,
            "expiry_days": cp.expiry_days,
            "policy_text_hash": sha256_hex({"policy_text": cp.policy_text}),
            "created_at": str(cp.created_at),
        },
        actor=actor,
    )
    return cp


@router.get("/consents", response_model=List[ConsentPolicy], tags=["core"])
def list_consents(
    dataset_id: Optional[UUID] = Query(default=None),
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER)),
):

    store = get_store()

    if dataset_id is not None:
        ds = store.get_dataset(dataset_id)
        if ds is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        _ensure_dataset_visible(actor, ds) # visible

    policies = store.list_consent_policies(dataset_id=dataset_id)

    if actor.role == ROLE_HOSPITAL and dataset_id is None:
        org_ds_ids = {str(d.dataset_id) for d in store.list_datasets() if _dataset_owned_by_org(d, actor.org)}
        policies = [p for p in policies if str(p.dataset_id) in org_ds_ids]

    return policies


@router.get("/consents/active/{dataset_id}", response_model=Optional[ConsentPolicy], tags=["core"])
def get_active_consent(
    dataset_id: UUID,
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER)),
):

    store = get_store()
    ds = store.get_dataset(dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _ensure_dataset_visible(actor, ds)

    policies = store.list_consent_policies(dataset_id=dataset_id)
    for p in reversed(policies):    # Επιστρέφει την τελευταία πιο πρόσφατη policy για dataset.
        if getattr(p, "status", None) == ConsentStatus.active:
            return p
    return None


@router.patch("/consents/{policy_id}/status", response_model=ConsentPolicy, tags=["core"])
def update_consent_status(  # έλεγχος αν ο Hospital αλλάξει status σε consent policy (draft/active/retired).
    policy_id: UUID,
    status: ConsentStatus = Query(...),
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL)),
):

    store = get_store()
    cp = store.get_consent_policy(policy_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Consent policy not found")

    ds = store.get_dataset(cp.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found for consent policy")
    _ensure_dataset_writable(actor, ds)

    updated = store.update_consent_status(policy_id, status, actor=actor.username)

    bc = get_blockchain()
    bc.anchor(
        event_type="CONSENT_STATUS_CHANGED",
        ref_id=str(updated.policy_id),
        payload={
            "policy_id": str(updated.policy_id),
            "dataset_id": str(updated.dataset_id),
            "new_status": str(updated.status),
            "updated_at": str(updated.updated_at),
        },
        actor=actor,
    )
    return updated



# Access Requests

@router.post("/access-requests", response_model=AccessRequest, tags=["core"])
def create_access_request(
    payload: AccessRequestCreate,
    actor: Actor = Depends(require_roles(ROLE_RESEARCHER, ROLE_BIOBANK)),
):

    store = get_store()
    ds = store.get_dataset(payload.dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found for access request")

    if payload.requester_org and not _same_org(payload.requester_org, actor.org):
        raise HTTPException(status_code=403, detail="requester_org must match your account org")

    return store.create_access_request(payload, actor=actor.username)


@router.get("/access-requests", response_model=List[AccessRequest], tags=["core"])
def list_access_requests(
    dataset_id: Optional[UUID] = Query(default=None),
    status: Optional[RequestStatus] = Query(default=None),
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER)),
):

    store = get_store()
    items = store.list_access_requests(dataset_id=dataset_id, status=status)

    if actor.role == ROLE_HOSPITAL:
        org_ds_ids = {str(d.dataset_id) for d in store.list_datasets() if _dataset_owned_by_org(d, actor.org)}
        items = [r for r in items if str(r.dataset_id) in org_ds_ids]
    elif actor.role == ROLE_RESEARCHER:
        items = [r for r in items if (r.requested_by or "").strip().lower() == actor.username.strip().lower()]
    elif actor.role == ROLE_BIOBANK:
        items = [
            r for r in items
            if (r.requested_by or "").strip().lower() == actor.username.strip().lower()
            or _same_org(r.requester_org, actor.org)
        ]
    return items


@router.patch("/access-requests/{request_id}/decision", response_model=AccessRequest, tags=["core"])
def decide_access_request(
    request_id: UUID,
    decision: RequestStatus = Query(..., description="approved or denied"),
    notes: Optional[str] = Query(default=None),
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL)),
):

    store = get_store()

    req = None
    for r in store.list_access_requests():
        if str(r.request_id) == str(request_id):
            req = r
            break
    if not req:
        raise HTTPException(status_code=404, detail="Access request not found")

    ds = store.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found for request")

    _ensure_dataset_writable(actor, ds)

    updated = store.decide_access_request(request_id, decision=decision, decision_notes=notes, actor=actor.username)

    bc = get_blockchain()
    bc.anchor(
        event_type="ACCESS_REQUEST_DECIDED",
        ref_id=str(updated.request_id),
        payload={
            "request_id": str(updated.request_id),
            "dataset_id": str(updated.dataset_id),
            "decision": str(updated.status),
            "decided_at": str(updated.decided_at),
            "decision_notes_hash": sha256_hex({"notes": notes or ""}),
        },
        actor=actor,
    )
    return updated


# Federated Jobs (PoC)

@router.post("/fl/jobs", response_model=FLJob, tags=["federation"])
def create_fl_job(
    payload: FLJobCreate,
    actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_RESEARCHER, ROLE_BIOBANK)),
):

    store = get_store()

    ds = store.get_dataset(payload.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if actor.role == ROLE_HOSPITAL and not _dataset_owned_by_org(ds, actor.org):
        raise HTTPException(status_code=403, detail="Dataset not in your organization scope")

    return store.create_fl_job(payload, actor=actor.username)


@router.get("/fl/jobs/{job_id}", response_model=FLJob, tags=["federation"])
def get_fl_job(job_id: UUID, actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER))):

    store = get_store()
    job = store.get_fl_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/fl/jobs/{job_id}/start", response_model=FLJob, tags=["federation"])
def start_fl_job(
    job_id: UUID, actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_RESEARCHER, ROLE_BIOBANK)),):

    store = get_store()

    job = store.get_fl_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    ds = store.get_dataset(job.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found for job")

    if actor.role == ROLE_HOSPITAL and not _dataset_owned_by_org(ds, actor.org):
        raise HTTPException(status_code=403, detail="Dataset not in your organization scope")

    node = store.get_node(ds.node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found for dataset")

    # Mark running
    job.status = FLJobStatus.running
    job.current_round = 0
    job.last_error = None
    store.update_fl_job(job, actor=actor.username, audit_event=None)

    try:
        url = f"{node.base_url.rstrip('/')}/train_round"
        global_model: dict = {}

        for r in range(1, job.rounds + 1):
            # Agent computes round update
            resp = _http_json_post(
                url,
                payload={
                    "job_id": str(job.job_id),
                    "round": r,
                    "dataset_id": str(ds.dataset_id),
                    "local_uri": ds.local_uri,
                    "schema_id": ds.schema_id,
                    "features": job.features,
                    "label": job.label,
                },
                headers={"X-Agent-Secret": AGENT_REG_SECRET},
                timeout=AGENT_CALL_TIMEOUT,
            )

            row_count = int(resp.get("row_count", 0) or 0)
            update = resp.get("update", {}) or {}

            # PoC: “global_model” είναι απλώς το update από τον agent
            # (σε multi-node FL θα γινόταν aggregation)
            if row_count > 0:
                global_model = update # Προς το παρόν τρέχει ένα dataset ανά νοσοκομείο -> δυνατότητα μελλοντικής επέκτασης για πολλά datasets και πολλά νοσοκομεία.

            job.current_round = r
            job.global_model = {k: float(v) for k, v in (global_model or {}).items()}
            agent_metrics = resp.get("metrics", {}) or {}


            round_trends = (job.metrics or {}).get("round_trends") or {} # κρατάμε ιστορικό mean ανά feature σε κάθε round
            for k, v in (global_model or {}).items():
                key = f"{k}_mean"
                round_trends.setdefault(key, [])
                round_trends[key].append(float(v))

            merged = dict(job.metrics or {})
            merged.update(agent_metrics)
            merged.update(
                {
                    "last_round_row_count": row_count,
                    "last_round": r,
                    "node_id": str(node.node_id),
                    "round_trends": round_trends,
                }
            )

            job.metrics = merged
            store.update_fl_job(job, actor=actor.username, audit_event=None)

        job.status = FLJobStatus.finished
        store.update_fl_job(job, actor=actor.username, audit_event=None)
        return job

    except Exception as e:
        job.status = FLJobStatus.failed
        job.last_error = str(e)
        store.update_fl_job(job, actor=actor.username, audit_event=None)
        raise HTTPException(status_code=500, detail=f"FL job failed: {e}")



# Runs / History

@router.post("/runs", response_model=Run, tags=["core"])
def create_run(payload: RunCreate, actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER))):
    store = get_store()
    entry = store.create_run(actor=actor.username, run_type=payload.run_type, payload=payload.payload)

    bc = get_blockchain()
    bc.anchor(
        event_type="RUN_RECORDED",
        ref_id=str(entry.get("run_id")),
        payload={
            "run_id": str(entry.get("run_id")),
            "run_type": entry.get("run_type"),
            "actor": entry.get("actor"),
            "created_at": entry.get("created_at"),
            "payload_hash": sha256_hex(entry.get("payload", {})),
        },
        actor=actor,
    )
    return entry


@router.get("/runs", response_model=List[Run], tags=["core"])
def list_runs(mine: int = Query(default=0), actor: Actor = Depends(require_roles(ROLE_HOSPITAL, ROLE_BIOBANK, ROLE_RESEARCHER))):
    store = get_store()
    if mine:
        return store.list_runs(actor=actor.username)
    return store.list_runs()



# Audit

@router.get("/audit", response_model=List[AuditLog], tags=["core"])
def list_audit(limit: int = Query(default=100, ge=1, le=500), actor: Actor = Depends(require_roles(ROLE_HOSPITAL))):
    store = get_store()
    logs = store.list_audit(limit=limit)

    org_ds_ids = {str(d.dataset_id) for d in store.list_datasets() if _dataset_owned_by_org(d, actor.org)} # Επιστρέφει μόνο τα δικά του logs (Νοσοκομείου)
    scoped: List[AuditLog] = []
    for e in logs:
        d = e.details or {}
        dsid = str(d.get("dataset_id", "")).strip()
        if dsid and dsid in org_ds_ids: #Κράτα το log αν αφορά dataset του οργανισμού
            scoped.append(e)
            continue
        if (e.actor or "").strip().lower() == actor.username.strip().lower():
            scoped.append(e)

    return scoped



# Blockchain receipts

@router.get("/blockchain/receipts", response_model=List[dict], tags=["core"])
def list_blockchain_receipts(limit: int = Query(default=200, ge=1, le=500),actor: Actor = Depends(require_roles("Admin", ROLE_HOSPITAL))):

    store = get_store() # αποδείξεις ότι οι ενέργειες αυτές αγκυρώθηκαν στο blockchain χωρίς να αλλοιωθούν
    return store.list_bc_receipts(limit=limit)

