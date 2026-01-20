# backend/app/schemas/domain.py
from __future__ import annotations

"""
Domain schemas (Pydantic models) για την πλατφόρμα BC-FL.

Σκοπός αυτού του αρχείου:
- Να ορίσει ενιαία “συμβόλαια” (request/response models) που χρησιμοποιούνται από:
  - backend/app/api/routes.py (FastAPI endpoints)
  - backend/app/services/sqlite_store.py (αποθήκευση/ανάκτηση μοντέλων)
  - agent/agent_app.py (payloads για validate/train_round, όπου χρειάζεται)
  - Streamlit UI (έμμεσα: ό,τι επιστρέφει το API)

Τι περιλαμβάνει (σε υψηλό επίπεδο):
- Enums για statuses (SensitivityLevel, DescriptorStatus, ConsentStatus, NodeStatus, FLJobStatus, RequestStatus κ.λπ.)
- Models για:
  - Users / Auth (αν υπάρχουν εδώ)
  - Nodes (agents)
  - Dataset descriptors (metadata για datasets που “μένουν” στο node)
  - Consent Policies (dataset-level governance)
  - Access Requests (workflow έγκρισης)
  - Federated Jobs (PoC orchestration runs)
  - Audit logs
  - Runs / History (καταγραφή ενεργειών/εκτελέσεων)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# -------------------------
# Time helper
# -------------------------
def utc_now() -> datetime:
    """
    Επιστρέφει timezone-aware UTC datetime.

    Γιατί:
    - Θέλουμε consistent timestamps σε DB/logs
    - Αποφεύγουμε naive datetimes που προκαλούν ambiguity
    """
    return datetime.now(timezone.utc)


# -------------------------
# Roles (RBAC)
# -------------------------
class Role(str, Enum):
    """
    Canonical roles στην πλατφόρμα.

    Σημείωση:
    - Στο auth.py κρατάμε τα role strings και το require_roles κάνει έλεγχο.
    - Εδώ το enum βοηθάει στο να “κλειδώσουμε” valid values σε policies / payloads.
    """
    Admin = "Admin"
    Hospital = "Hospital"
    Biobank = "Biobank"
    Researcher = "Researcher"


# -------------------------
# Enums (statuses, methods)
# -------------------------
class SensitivityLevel(str, Enum):
    """
    Sensitivity label dataset descriptor (PoC classification).
    Χρησιμοποιείται για governance/consent decisions και μελλοντικούς κανόνες.
    """
    low = "low"
    medium = "medium"
    high = "high"


class DescriptorStatus(str, Enum):
    """
    Status του dataset descriptor στο backend.

    - registered: δημιουργήθηκε descriptor
    - validated: agent validation ok (γνωρίζουμε columns/row_count)
    - unavailable: validation failed ή dataset δεν βρέθηκε/δεν ανοίγει
    """
    registered = "registered"
    validated = "validated"
    unavailable = "unavailable"


class ConsentStatus(str, Enum):
    """Status μιας consent policy (dataset-level)."""
    draft = "draft"
    active = "active"
    retired = "retired"


class RequestStatus(str, Enum):
    """Lifecycle ενός access request."""
    submitted = "submitted"
    approved = "approved"
    denied = "denied"


class ExportMethod(str, Enum):
    """
    Θεωρητική/μελλοντική κατηγορία “εξαγωγής” (PoC).

    - federated: computation stays at node, only aggregates return
    - aggregated: pre-aggregated results shared
    - synthetic: synthetic dataset shared
    """
    federated = "federated"
    aggregated = "aggregated"
    synthetic = "synthetic"


class AuditEvent(str, Enum):
    """
    Audit event types για logging.
    Αυτά γράφονται στη SQLite (audit table) από sqlite_store.py.

    Σημείωση:
    - Στο sqlite_store.py χρησιμοποιούνται και άλλα events (π.χ. NODE_REGISTERED),
      οπότε εδώ πρέπει να συμφωνεί με την πραγματική χρήση.
    - Αν λείπουν events, θα έχεις runtime error όταν προσπαθεί να κάνει AuditEvent.X
      (γιατί είναι Enum).
    """
    DATASET_CREATED = "DATASET_CREATED"
    DESCRIPTOR_CREATED = "DESCRIPTOR_CREATED"
    DESCRIPTOR_VALIDATED = "DESCRIPTOR_VALIDATED"

    NODE_REGISTERED = "NODE_REGISTERED"
    NODE_HEARTBEAT = "NODE_HEARTBEAT"

    CONSENT_CREATED = "CONSENT_CREATED"
    CONSENT_UPDATED = "CONSENT_UPDATED"

    ACCESS_REQUEST_CREATED = "ACCESS_REQUEST_CREATED"
    ACCESS_REQUEST_DECIDED = "ACCESS_REQUEST_DECIDED"

    FLJOB_CREATED = "FLJOB_CREATED"

    EXPORT_CREATED = "EXPORT_CREATED"


class NodeStatus(str, Enum):
    """Κατάσταση ενός agent node."""
    online = "online"
    offline = "offline"
    unknown = "unknown"


class FLJobStatus(str, Enum):
    """Lifecycle ενός federated job."""
    created = "created"
    running = "running"
    finished = "finished"
    failed = "failed"


# -------------------------
# Nodes (Agents)
# -------------------------
class NodeRegister(BaseModel):
    """
    Payload που στέλνει ο agent κατά το registration.

    Περιλαμβάνει τα βασικά στοιχεία ταυτότητας του node:
    - org: σε ποιον οργανισμό/νοσοκομείο ανήκει
    - base_url: πού θα το καλέσει ο backend (π.χ. http://agent-hosp1:9001)
    - name: human-friendly label
    """
    org: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    name: Optional[str] = None


class Node(NodeRegister):
    """
    Node record που επιστρέφεται από το API και αποθηκεύεται στο store.
    """
    node_id: UUID
    created_at: datetime
    last_seen_at: Optional[datetime] = None
    status: NodeStatus = NodeStatus.unknown


# -------------------------
# Dataset descriptors
# -------------------------
class DatasetCreate(BaseModel):
    """
    Δημιουργία dataset descriptor.

    Σημείωση:
    - Δεν ανεβάζουμε raw αρχείο στο backend.
    - local_uri δείχνει σε path ή identifier που μπορεί να ανοίξει ο agent.
    - schema_id είναι label για expected schema (PoC / validation rules).
    """
    name: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None, max_length=5000)

    owner_org: str = Field(..., min_length=1)  # enforced στο routes.py από actor.org
    sensitivity_level: SensitivityLevel = SensitivityLevel.low

    # “routing” προς node:
    schema_id: Optional[str] = None
    local_uri: Optional[str] = None
    node_id: UUID


class Dataset(DatasetCreate):
    """
    Dataset descriptor record με runtime metadata.

    - status/row_count/columns/validation_report προκύπτουν από validate.
    - exposed_features: subset features που επιτρέπεται να δουν external actors.
    """
    dataset_id: UUID
    created_at: datetime

    status: DescriptorStatus = DescriptorStatus.registered
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    validation_report: Optional[Dict[str, Any]] = None

    exposed_features: Optional[List[str]] = None


class DatasetFeaturesUpdate(BaseModel):
    """
    Payload για update των exposed features από Hospital.
    """
    exposed_features: List[str] = Field(default_factory=list)


# -------------------------
# Consent Policies (dataset-level)
# -------------------------
class ConsentPolicyCreate(BaseModel):
    """
    Consent policy για dataset.

    Σημείωση:
    - Αυτό ΔΕΝ είναι “per-patient consent”.
      Το per-patient consent το κρατάς μέσω ConsentRegistry (smart contract)
      και endpoints στο patient_consent_routes.py.
    """
    dataset_id: UUID
    policy_text: str = Field(..., min_length=1, max_length=20000)

    status: ConsentStatus = ConsentStatus.active
    allow_external: bool = True

    # Roles που επιτρέπονται να ζητούν/λαμβάνουν πρόσβαση (PoC governance)
    allowed_roles: List[Role] = Field(default_factory=list)

    expiry_days: Optional[int] = Field(default=None, ge=1)

    # Επιτρεπτές μέθοδοι (federated/aggregated/synthetic)
    export_methods: List[ExportMethod] = Field(default_factory=list)

    # Structured policy payload (π.χ. GDPR constraints / conditions)
    policy_structured: Optional[Dict[str, Any]] = None


class ConsentPolicy(ConsentPolicyCreate):
    """Policy record με metadata."""
    policy_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None


# -------------------------
# Access Requests
# -------------------------
class AccessRequestCreate(BaseModel):
    """
    External actor δημιουργεί αίτημα πρόσβασης σε dataset.

    Fields:
    - requester_org: οργανισμός του αιτούντος (πρέπει να ταιριάζει με actor.org)
    - requested_by: username του αιτούντος (actor.username)
    - role: role του αιτούντος (Researcher/Biobank)
    - purpose: περιγραφή σκοπού
    """
    dataset_id: UUID
    requester_org: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1, max_length=5000)
    requested_by: str = Field(..., min_length=1)
    role: Role


class AccessRequest(AccessRequestCreate):
    """Access request record με lifecycle fields."""
    request_id: UUID
    created_at: datetime
    decided_at: Optional[datetime] = None
    decision_notes: Optional[str] = None
    status: RequestStatus = RequestStatus.submitted


# -------------------------
# Federated Jobs (PoC)
# -------------------------
class FLJobCreate(BaseModel):
    """
    Payload για δημιουργία federated job.

    - dataset_id: πάνω σε ποιο dataset descriptor τρέχει
    - rounds: πόσα “rounds” (PoC loop)
    - features/label: selection του computation schema (π.χ. mean ανά feature)
    """
    dataset_id: UUID
    rounds: int = Field(..., ge=1, le=100)
    features: List[str] = Field(default_factory=list)
    label: Optional[str] = None


class FLJob(FLJobCreate):
    """
    Job record που αποθηκεύεται στο store.

    - global_model: PoC dict (π.χ. means per feature)
    - metrics: dict με round_trends + agent metrics
    """
    job_id: UUID
    created_at: datetime

    status: FLJobStatus = FLJobStatus.created
    current_round: int = 0
    last_error: Optional[str] = None

    global_model: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Runs / History
# -------------------------
class RunCreate(BaseModel):
    """
    Run log entry (UI history).

    - run_type: label π.χ. "validate_dataset", "start_fl_job"
    - payload: details (non-sensitive ideally; hash goes on-chain via bc.anchor)
    """
    run_type: str = Field(..., min_length=1)
    payload: Dict[str, Any] = Field(default_factory=dict)


class Run(BaseModel):
    """Stored run record."""
    run_id: str
    created_at: str
    actor: str
    run_type: str
    payload: Dict[str, Any]


# -------------------------
# Audit Logs
# -------------------------
class AuditLog(BaseModel):
    """
    Audit entry (immutable log record).

    - event_type: enum AuditEvent
    - actor: username (όταν υπάρχει)
    - details: structured metadata για traceability
    """
    event_id: UUID
    event_type: AuditEvent
    created_at: datetime
    actor: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# Exports (placeholder)
# -------------------------
class DataExportCreate(BaseModel):
    """
    Placeholder για μελλοντική υλοποίηση export object.
    """
    dataset_id: UUID
    request_id: UUID
    method: ExportMethod = ExportMethod.federated


class DataExport(DataExportCreate):
    """Stored export record."""
    export_id: UUID
    export_ref: str
    created_at: datetime
