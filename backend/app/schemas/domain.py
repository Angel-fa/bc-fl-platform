from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Role Base Access Control

class Role(str, Enum):
    Admin = "Admin"
    Hospital = "Hospital"
    Biobank = "Biobank"
    Researcher = "Researcher"


class SensitivityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class DescriptorStatus(str, Enum):
    registered = "registered" # δημιουργήθηκε descriptor
    validated = "validated"
    unavailable = "unavailable" # validation failed ή dataset δεν βρέθηκε/δεν ανοίγει


class ConsentStatus(str, Enum):
    draft = "draft"
    active = "active"
    retired = "retired"


class RequestStatus(str, Enum):   # Lifecycle ενός access request
    submitted = "submitted"
    approved = "approved"
    denied = "denied"


class ExportMethod(str, Enum):   # δυνατότητες προέκτασης
    federated = "federated"
    aggregated = "aggregated"
    synthetic = "synthetic"


class AuditEvent(str, Enum):

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
    online = "online"
    offline = "offline"
    unknown = "unknown"


class FLJobStatus(str, Enum):
    created = "created"
    running = "running"
    finished = "finished"
    failed = "failed"


# Nodes (Agents)

class NodeRegister(BaseModel):  # βασικά στοιχεία ταυτότητας του node
    org: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    name: Optional[str] = None


class Node(NodeRegister):
    node_id: UUID
    created_at: datetime
    last_seen_at: Optional[datetime] = None
    status: NodeStatus = NodeStatus.unknown


# Dataset descriptors

class DatasetCreate(BaseModel):

    name: str = Field(..., min_length=1)
    description: Optional[str] = Field(default=None, max_length=500)

    owner_org: str = Field(..., min_length=1)
    sensitivity_level: SensitivityLevel = SensitivityLevel.low

    schema_id: Optional[str] = None
    local_uri: Optional[str] = None
    node_id: UUID


class Dataset(DatasetCreate):

    dataset_id: UUID
    created_at: datetime

    status: DescriptorStatus = DescriptorStatus.registered
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    validation_report: Optional[Dict[str, Any]] = None

    exposed_features: Optional[List[str]] = None


class DatasetFeaturesUpdate(BaseModel):  # update των exposed features
    exposed_features: List[str] = Field(default_factory=list)


# Consent Policies

class ConsentPolicyCreate(BaseModel):

    dataset_id: UUID
    policy_text: str = Field(..., min_length=1, max_length=20000)

    status: ConsentStatus = ConsentStatus.active
    allow_external: bool = True

    # Ρόλοι που επιτρέπονται να ζητούν και να λαμβάνουν πρόσβαση
    allowed_roles: List[Role] = Field(default_factory=list)

    expiry_days: Optional[int] = Field(default=None, ge=1)

    #  μέθοδοι (federated/aggregated/synthetic)
    export_methods: List[ExportMethod] = Field(default_factory=list)

    policy_structured: Optional[Dict[str, Any]] = None


class ConsentPolicy(ConsentPolicyCreate):  # Policy record με metadata
    policy_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None


# Access Requests

class AccessRequestCreate(BaseModel):
    dataset_id: UUID
    requester_org: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1, max_length=5000)
    requested_by: str = Field(..., min_length=1)
    role: Role
    notes: Optional[str] = None

class AccessRequest(AccessRequestCreate): # με lifecycle
    request_id: UUID
    created_at: datetime
    decided_at: Optional[datetime] = None
    decision_notes: Optional[str] = None
    status: RequestStatus = RequestStatus.submitted


# Federated Jobs

class FLJobCreate(BaseModel):
    dataset_id: UUID

    #  single vs multi
    scope: Literal["single_node", "multi_node"] = "single_node"
    dataset_ids: List[UUID] = Field(default_factory=list)

    rounds: int = Field(..., ge=1, le=100)
    features: List[str] = Field(default_factory=list)
    label: Optional[str] = None

    created_by: Optional[str] = None
    created_by_org: Optional[str] = None
    created_by_role: Optional[Role] = None

class FLJob(FLJobCreate):
    job_id: UUID
    created_at: datetime

    status: FLJobStatus = FLJobStatus.created
    current_round: int = 0
    last_error: Optional[str] = None

    global_model: Dict[str, float] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


# Runs / History

class RunCreate(BaseModel):
    run_type: str = Field(..., min_length=1) # label π.χ. --> "start_fl_job"
    payload: Dict[str, Any] = Field(default_factory=dict) # details


class Run(BaseModel):
    run_id: str
    created_at: str
    actor: str
    run_type: str
    payload: Dict[str, Any]


# Audit Logs

class AuditLog(BaseModel):
    event_id: UUID
    event_type: AuditEvent
    created_at: datetime
    actor: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# Exports

class DataExportCreate(BaseModel):  # για methods
    dataset_id: UUID
    request_id: UUID
    method: ExportMethod = ExportMethod.federated


class DataExport(DataExportCreate):
    export_id: UUID
    export_ref: str
    created_at: datetime
