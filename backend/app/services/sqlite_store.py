
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from app.schemas.domain import (
    AccessRequest,
    AccessRequestCreate,
    AuditEvent,
    AuditLog,
    ConsentPolicy,
    ConsentPolicyCreate,
    ConsentStatus,
    DataExport,
    DataExportCreate,
    Dataset,
    DatasetCreate,
    DescriptorStatus,
    FLJob,
    FLJobCreate,
    FLJobStatus,
    Node,
    NodeRegister,
    NodeStatus,
    RequestStatus,
    utc_now,
)


DEFAULT_DB_PATH = os.getenv("DB_PATH", "/code/.data/bcfl.db")


# Helpers

# Εξασφαλίζει ότι κάτι είναι UUID object. Αν  είναι το επιστρέφει, αλλιώς προσπαθεί να το μετατρέψει από string.
def _uuid(u: Any) -> UUID:
    return u if isinstance(u, UUID) else UUID(str(u))


def _to_json(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _from_json(s: str) -> Any:
    return json.loads(s) if s else None


def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any((r[1] == col) for r in rows) # Ελέγχει αν ένας πίνακας έχει συγκεκριμένη στήλη.


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, coldef_sql: str) -> None:
    if not _table_has_column(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coldef_sql}") # Αν λείπει στήλη, κάνει ALTER TABLE ADD COLUMN



@dataclass
class SQLiteStore:
    db_path: str = DEFAULT_DB_PATH

    def _conn(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn # Φτιάχνει τον φάκελο αν δεν υπάρχει (/code/.data) - συνδέεται στο SQLite αρχείο

    def init(self) -> None:   # δημιουργεί όλους τους πίνακες που χρειαζόμαστε
        with self._conn() as c:
            #  blockchain receipts
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bc_receipts (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    ref_id TEXT,
                    tx_hash TEXT,
                    chain_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # users
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            #  αν δεν έχει τις στήλες, τις προσθέτουμε
            _ensure_column(c, "users", "payload_json", "TEXT")
            _ensure_column(c, "users", "created_at", "TEXT")
            _ensure_column(c, "users", "is_active", "INTEGER")

            # federated descriptors
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # registered agents/hospital nodes
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # fl_jobs
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS fl_jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # Consent Policies
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS consents (
                    policy_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    dataset_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # access_requests
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS access_requests (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # exports
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS exports (
                    export_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # audit
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS audit (
                    event_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

            # runs/history
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )

    # Users

    def create_user(self, username: str, password_hash: str, role: str, org: str) -> Dict[str, Any]:
        username = (username or "").strip()
        org = (org or "").strip()
        role = (role or "").strip()
        if not username:
            raise ValueError("username required")
        if not password_hash:
            raise ValueError("password_hash required")
        if not role:
            raise ValueError("role required")
        if not org:
            raise ValueError("org required")

        created_at = utc_now()
        user_obj = {
            "username": username,
            "password_hash": password_hash,
            "role": role,
            "org": org,
            "is_active": True,
            "created_at": created_at,
        }

        with self._conn() as c:
            existing = c.execute("SELECT username FROM users WHERE username = ?", (username,)).fetchone()
            if existing:
                raise ValueError("Username already exists")

            c.execute(
                "INSERT INTO users(username, created_at, is_active, payload_json) VALUES (?, ?, ?, ?)",
                (username, created_at.isoformat(), 1, _to_json(user_obj)),
            )

        return {"username": username, "role": role, "org": org, "is_active": True, "created_at": created_at}

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        username = (username or "").strip()
        if not username:
            return None
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return None

        data = _from_json(row["payload_json"]) or {}

        ca = data.get("created_at")
        if isinstance(ca, str):
            try:
                data["created_at"] = datetime.fromisoformat(ca)
            except Exception:
                pass
        return data

    def set_user_active(self, username: str, is_active: bool) -> Dict[str, Any]: #   Ενεργοποιεί/απενεργοποιεί χρήστη
        username = (username or "").strip()
        if not username:
            raise ValueError("username required")

        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                raise KeyError("User not found")

            data = _from_json(row["payload_json"]) or {}
            data["is_active"] = bool(is_active)

            c.execute(
                "UPDATE users SET is_active = ?, payload_json = ? WHERE username = ?",
                (1 if is_active else 0, _to_json(data), username),
            )

        ca = data.get("created_at")
        if isinstance(ca, str):
            try:
                ca = datetime.fromisoformat(ca)
            except Exception:
                pass

        return {
            "username": data.get("username", username),
            "role": data.get("role", ""),
            "org": data.get("org", ""),
            "is_active": bool(data.get("is_active", False)),
            "created_at": ca,
        }

    # Audit

    def log(self, event_type: AuditEvent, actor: Optional[str] = None, details: Optional[dict] = None) -> AuditLog:

        entry = AuditLog(          # Καταγράφει audit event off-chain log
            event_id=uuid4(),
            event_type=event_type,
            created_at=utc_now(),
            actor=actor,
            details=details or {},
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit(event_id, created_at, payload_json) VALUES (?, ?, ?)",
                (str(entry.event_id), entry.created_at.isoformat(), _to_json(entry.model_dump())),
            )
        return entry

    def list_audit(self, limit: int = 100) -> List[AuditLog]:

        with self._conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM audit ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [AuditLog(**_from_json(r["payload_json"])) for r in rows]       #  Επιστρέφει audit events (latest first)

    # Runs / History

    # Runs / History

    def create_run(self, actor: str, run_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        run_id = str(uuid4())
        created_at = utc_now()
        entry = {
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "actor": (actor or "").strip(),
            "run_type": (run_type or "").strip(),
            "payload": payload or {},
        }
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs(run_id, created_at, actor, run_type, payload_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, created_at.isoformat(), entry["actor"], entry["run_type"], _to_json(entry)),
            )
        return entry

    def list_runs_all(self, limit: int = 200) -> List[Dict[str, Any]]:

        with self._conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_from_json(r["payload_json"]) for r in rows]

    def list_runs_for_user(self, username: str, limit: int = 200) -> List[Dict[str, Any]]:

        username = (username or "").strip()
        if not username:
            return []

        with self._conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM runs WHERE actor = ? ORDER BY created_at DESC LIMIT ?",
                (username, int(limit)),
            ).fetchall()
        return [_from_json(r["payload_json"]) for r in rows]

    # Blockchain Receipts

    def save_bc_receipt(
        self,
        event_type: str,
        ref_id: Optional[str],
        payload: Dict[str, Any],
        tx_hash: Optional[str] = None,
        chain_id: Optional[str] = None,
    ) -> Dict[str, Any]:     #   Αποθηκεύει “receipt”  στο sqlite.

        rid = str(uuid4())
        created_at = utc_now()
        entry = {
            "id": rid,
            "created_at": created_at.isoformat(),
            "event_type": (event_type or "").strip(),
            "ref_id": (ref_id or "").strip() if ref_id else None,
            "tx_hash": (tx_hash or "").strip() if tx_hash else None,
            "chain_id": (chain_id or "").strip() if chain_id else None,
            "payload": payload or {},
        }
        with self._conn() as c:
            c.execute(
                "INSERT INTO bc_receipts(id, created_at, event_type, ref_id, tx_hash, chain_id, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry["id"],
                    entry["created_at"],
                    entry["event_type"],
                    entry["ref_id"],
                    entry["tx_hash"],
                    entry["chain_id"],
                    _to_json(entry),
                ),
            )
        return entry

    #   Επιστρέφει receipts
    def list_bc_receipts(
            self,
            limit: int = 200,
            event_type: Optional[str] = None,
            ref_id: Optional[str] = None,) \
            -> List[Dict[str, Any]]:
        q = "SELECT payload_json FROM bc_receipts"
        params: List[Any] = []
        where: List[str] = []

        if event_type:
            where.append("event_type = ?")
            params.append((event_type or "").strip())

        if ref_id:
            where.append("ref_id = ?")
            params.append((ref_id or "").strip())

        if where:
            q += " WHERE " + " AND ".join(where)

        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))

        with self._conn() as c:
            rows = c.execute(q, params).fetchall()

        return [_from_json(r["payload_json"]) for r in rows]



    # Nodes
    def register_node(self, payload: NodeRegister, actor: Optional[str] = None) -> Node:   # Καταχωρεί έναν hospital agent node.

        node = Node(
            node_id=uuid4(),
            created_at=utc_now(),
            last_seen_at=utc_now(),
            status=NodeStatus.online,
            **payload.model_dump(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO nodes(node_id, created_at, last_seen_at, payload_json) VALUES (?, ?, ?, ?)",
                (str(node.node_id), node.created_at.isoformat(), node.last_seen_at.isoformat(), _to_json(node.model_dump())),
            )
        # Audit event -> node registered
        self.log(
            AuditEvent.NODE_REGISTERED,
            actor=actor,
            details={"node_id": str(node.node_id), "org": node.org, "base_url": node.base_url},
        )
        return node

    def list_nodes(self) -> List[Node]:
        with self._conn() as c:
            rows = c.execute("SELECT payload_json FROM nodes ORDER BY created_at DESC").fetchall()
        return [Node(**_from_json(r["payload_json"])) for r in rows]  # Επιστρέφει όλους τους nodes

    def get_node(self, node_id: UUID) -> Optional[Node]:
        nid = str(_uuid(node_id))
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM nodes WHERE node_id = ?", (nid,)).fetchone()
        return Node(**_from_json(row["payload_json"])) if row else None # Επιστρέφει έναν node με βάση το node_id

    def heartbeat_node(self, node_id: UUID, status: NodeStatus = NodeStatus.online, actor: Optional[str] = None) -> Node:

        nid = str(_uuid(node_id))
        now = utc_now().isoformat()

        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM nodes WHERE node_id = ?", (nid,)).fetchone()
            if not row:
                raise KeyError("Node not found")
            data = _from_json(row["payload_json"]) or {}
            data["last_seen_at"] = now
            data["status"] = str(status.value)
            c.execute(
                "UPDATE nodes SET last_seen_at = ?, payload_json = ? WHERE node_id = ?",
                (now, _to_json(data), nid),
            )

        node = Node(**data)
        self.log(AuditEvent.NODE_HEARTBEAT, actor=actor, details={"node_id": str(node.node_id), "status": node.status})
        return node # Ενημερώνει heartbeat ενός node

    def find_node_by_org(self, org: str) -> Optional[Node]:  # βρίσκει node με βάση org για να συσχετίσει dataset με node συγκεκριμένου org
        org = (org or "").strip().lower()
        for n in self.list_nodes():
            if (n.org or "").strip().lower() == org:
                return n
        return None

    # Datasets Descriptors

    def create_dataset(self, payload: DatasetCreate, actor: Optional[str] = None) -> Dataset:    #Δημιουργεί dataset descriptor

        ds = Dataset(
            dataset_id=uuid4(),
            created_at=utc_now(),
            status=DescriptorStatus.registered,
            row_count=None,
            columns=None,
            validation_report=None,
            **payload.model_dump(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO datasets(dataset_id, created_at, payload_json) VALUES (?, ?, ?)",
                (str(ds.dataset_id), ds.created_at.isoformat(), _to_json(ds.model_dump())),
            )
        self.log(
            AuditEvent.DESCRIPTOR_CREATED,
            actor=actor,
            details={"dataset_id": str(ds.dataset_id), "name": ds.name, "node_id": str(ds.node_id)},
        )                                                                # δείχνει ποιος agent έχει πρόσβαση στα δεδομένα.
        return ds

    def list_datasets(self) -> List[Dataset]:
        with self._conn() as c:
            rows = c.execute("SELECT payload_json FROM datasets ORDER BY created_at DESC").fetchall()
        return [Dataset(**_from_json(r["payload_json"])) for r in rows]  # Επιστρέφει όλα τα dataset descriptors

    def get_dataset(self, dataset_id: UUID) -> Optional[Dataset]:
        did = str(_uuid(dataset_id))
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM datasets WHERE dataset_id = ?", (did,)).fetchone()
        return Dataset(**_from_json(row["payload_json"])) if row else None # Επιστρέφει dataset descriptor με βάση dataset_id

    def update_dataset_validation(  # Αποθηκεύει τα validation αποτελέσματα του agent

        self,
        dataset_id: UUID,
        status: DescriptorStatus,
        row_count: Optional[int],
        columns: Optional[List[str]],
        report: Optional[Dict[str, Any]],
        actor: Optional[str] = None,
    ) -> Dataset:

        did = str(_uuid(dataset_id))
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM datasets WHERE dataset_id = ?", (did,)).fetchone()
            if not row:
                raise KeyError("Dataset not found")
            data = _from_json(row["payload_json"]) or {}
            data["status"] = str(status.value)
            data["row_count"] = row_count
            data["columns"] = columns
            data["validation_report"] = report or {}
            c.execute("UPDATE datasets SET payload_json = ? WHERE dataset_id = ?", (_to_json(data), did))

        ds = Dataset(**data)
        self.log(
            AuditEvent.DESCRIPTOR_VALIDATED,
            actor=actor,
            details={"dataset_id": str(ds.dataset_id), "status": ds.status, "row_count": ds.row_count},
        )
        return ds

    def update_dataset_exposed_features(   # Ενημερώνει exposed features
        self,
        dataset_id: UUID,
        exposed_features: List[str],
        actor: Optional[str] = None,
    ) -> Dataset:

        did = str(_uuid(dataset_id))
        exposed_features = [str(x).strip() for x in (exposed_features or []) if str(x).strip()]

        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM datasets WHERE dataset_id = ?", (did,)).fetchone()
            if not row:
                raise KeyError("Dataset not found")

            data = _from_json(row["payload_json"]) or {}
            data["exposed_features"] = exposed_features
            c.execute("UPDATE datasets SET payload_json = ? WHERE dataset_id = ?", (_to_json(data), did))

        ds = Dataset(**data)
        return ds

    # Federated Jobs

    def create_fl_job(self, payload: FLJobCreate, actor: Optional[str] = None) -> FLJob:

        job = FLJob(   # Δημιουργεί FL job record
            job_id=uuid4(),
            created_at=utc_now(),
            status=FLJobStatus.created,
            current_round=0,
            last_error=None,
            global_model={},
            metrics={},
            **payload.model_dump(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO fl_jobs(job_id, created_at, payload_json) VALUES (?, ?, ?)",
                (str(job.job_id), job.created_at.isoformat(), _to_json(job.model_dump())),
            )
        self.log(
            AuditEvent.FLJOB_CREATED,
            actor=actor,
            details={"job_id": str(job.job_id), "dataset_id": str(job.dataset_id), "rounds": job.rounds},
        )
        return job

    def get_fl_job(self, job_id: UUID) -> Optional[FLJob]:
        jid = str(_uuid(job_id))
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM fl_jobs WHERE job_id = ?", (jid,)).fetchone()
        return FLJob(**_from_json(row["payload_json"])) if row else None   # Ανάκτηση FL job από sqlite

    def list_fl_jobs(self, limit: int = 200) -> List[FLJob]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT payload_json FROM fl_jobs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [FLJob(**_from_json(r["payload_json"])) for r in rows]

    def update_fl_job(
        self,
        job: FLJob,
        actor: Optional[str] = None,
        audit_event: Optional[AuditEvent] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> FLJob:
        with self._conn() as c:
            c.execute(
                "UPDATE fl_jobs SET payload_json = ? WHERE job_id = ?",
                (_to_json(job.model_dump()), str(job.job_id))
            )
        if audit_event:
            self.log(audit_event, actor=actor, details=details or {})
        return job

    # Consent Policies

    def get_consent_policy(self, policy_id: UUID) -> Optional[ConsentPolicy]:
        pid = str(_uuid(policy_id))
        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM consents WHERE policy_id = ?", (pid,)).fetchone()
        return ConsentPolicy(**_from_json(row["payload_json"])) if row else None

    def create_consent_policy(self, payload: ConsentPolicyCreate, actor: Optional[str] = None) -> ConsentPolicy:  # Ανάκτηση Consent Policy
        cp = ConsentPolicy(   # Δημιουργία Consent Policy
            policy_id=uuid4(),
            created_at=utc_now(),
            updated_at=None,
            **payload.model_dump(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO consents(policy_id, created_at, updated_at, dataset_id, payload_json) VALUES (?, ?, ?, ?, ?)",
                (str(cp.policy_id), cp.created_at.isoformat(), None, str(cp.dataset_id), _to_json(cp.model_dump())),
            )
        self.log(
            AuditEvent.CONSENT_CREATED,
            actor=actor,
            details={"policy_id": str(cp.policy_id), "dataset_id": str(cp.dataset_id), "status": cp.status},
        )
        return cp

    def list_consent_policies(self, dataset_id: Optional[UUID] = None) -> List[ConsentPolicy]:
        q = "SELECT payload_json FROM consents"
        params: List[Any] = []
        if dataset_id is not None:
            q += " WHERE dataset_id = ?"
            params.append(str(_uuid(dataset_id)))
        q += " ORDER BY created_at DESC"

        with self._conn() as c:
            rows = c.execute(q, params).fetchall()
        return [ConsentPolicy(**_from_json(r["payload_json"])) for r in rows]

    def update_consent_status(self, policy_id: UUID, status: ConsentStatus, actor: Optional[str] = None) -> ConsentPolicy:
      # Ενημέρωση status (draft/active/retired)

        pid = str(_uuid(policy_id))
        now_iso = utc_now().isoformat()

        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM consents WHERE policy_id = ?", (pid,)).fetchone()
            if not row:
                raise KeyError("Consent policy not found")

            data = _from_json(row["payload_json"]) or {}
            data["status"] = str(status.value) if hasattr(status, "value") else str(status)
            data["updated_at"] = now_iso

            c.execute(
                "UPDATE consents SET updated_at = ?, payload_json = ? WHERE policy_id = ?",
                (now_iso, _to_json(data), pid),
            )

        cp = ConsentPolicy(**data)
        self.log(
            AuditEvent.CONSENT_UPDATED,
            actor=actor,
            details={"policy_id": str(cp.policy_id), "dataset_id": str(cp.dataset_id), "status": cp.status},
        )
        return cp


    # Access Requests

    def create_access_request(self, payload: AccessRequestCreate, actor: Optional[str] = None) -> AccessRequest:
        req = AccessRequest(
            request_id=uuid4(),
            created_at=utc_now(),
            decided_at=None,
            decision_notes=None,
            status=RequestStatus.submitted,
            **payload.model_dump(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO access_requests(request_id, created_at, decided_at, payload_json) VALUES (?, ?, ?, ?)",
                (str(req.request_id), req.created_at.isoformat(), None, _to_json(req.model_dump())),
            )
        self.log(
            AuditEvent.ACCESS_REQUEST_CREATED,
            actor=actor,
            details={
                "request_id": str(req.request_id),
                "dataset_id": str(req.dataset_id),
                "role": req.role,
                "requested_by": req.requested_by,
                "requester_org": req.requester_org
            },
        )
        return req

    def list_access_requests(
        self,
        dataset_id: Optional[UUID] = None,
        status: Optional[RequestStatus] = None
    ) -> List[AccessRequest]:

        with self._conn() as c:
            rows = c.execute("SELECT payload_json FROM access_requests ORDER BY created_at DESC").fetchall()
        items = [AccessRequest(**_from_json(r["payload_json"])) for r in rows]

        if dataset_id is not None:
            did = _uuid(dataset_id)
            items = [x for x in items if _uuid(x.dataset_id) == did]

        if status is not None:
            items = [x for x in items if x.status == status]

        return items

    def decide_access_request(   # Ενημέρωση Access Request decision (approved/denied).
        self,
        request_id: UUID,
        decision: RequestStatus,
        decision_notes: Optional[str],
        actor: Optional[str]
    ) -> AccessRequest:

        if decision not in (RequestStatus.approved, RequestStatus.denied):
            raise ValueError("Decision must be 'approved' or 'denied'.")

        rid = str(_uuid(request_id))
        now_iso = utc_now().isoformat()

        with self._conn() as c:
            row = c.execute("SELECT payload_json FROM access_requests WHERE request_id = ?", (rid,)).fetchone()
            if not row:
                raise KeyError("Access request not found")

            data = _from_json(row["payload_json"]) or {}
            data["status"] = str(decision.value) if hasattr(decision, "value") else str(decision)
            data["decided_at"] = now_iso
            data["decision_notes"] = decision_notes

            c.execute(
                "UPDATE access_requests SET decided_at = ?, payload_json = ? WHERE request_id = ?",
                (now_iso, _to_json(data), rid),
            )

        req = AccessRequest(**data)
        self.log(
            AuditEvent.ACCESS_REQUEST_DECIDED,
            actor=actor,
            details={"request_id": str(req.request_id), "decision": req.status, "dataset_id": str(req.dataset_id)},
        )
        return req


    # Exports

    def create_export(self, payload: DataExportCreate, actor: Optional[str] = None) -> DataExport:

        exp = DataExport(
            export_id=uuid4(),
            export_ref=f"export://{uuid4()}",
            created_at=utc_now(),
            **payload.model_dump(),
        )
        with self._conn() as c:
            c.execute(
                "INSERT INTO exports(export_id, created_at, payload_json) VALUES (?, ?, ?)",
                (str(exp.export_id), exp.created_at.isoformat(), _to_json(exp.model_dump())),
            )
        self.log(
            AuditEvent.EXPORT_CREATED,
            actor=actor,
            details={"export_id": str(exp.export_id), "request_id": str(exp.request_id), "method": exp.method},
        )
        return exp

    def list_exports(self, dataset_id: Optional[UUID] = None) -> List[DataExport]:
        with self._conn() as c:
            rows = c.execute("SELECT payload_json FROM exports ORDER BY created_at DESC").fetchall()
        items = [DataExport(**_from_json(r["payload_json"])) for r in rows]
        if dataset_id is not None:
            did = _uuid(dataset_id)
            items = [x for x in items if _uuid(x.dataset_id) == did]
        return items

_STORE: Optional[SQLiteStore] = None

def get_store() -> SQLiteStore:  # πρόσβαση στο data base
    global _STORE
    if _STORE is None:
        _STORE = SQLiteStore()
        _STORE.init()
    return _STORE
