# backend/app/services/blockchain_service.py
from __future__ import annotations

"""
BlockchainService: abstraction layer για anchoring/audit στο blockchain (ή σε dev/noop mode).

Βασική ιδέα:
- Η πλατφόρμα θέλει “ιχνηλασιμότητα” (traceability) και “ακεραιότητα” (integrity) για κρίσιμα γεγονότα:
  - καταχώρηση dataset descriptor
  - validation events
  - αλλαγές consent policy
  - αποφάσεις access requests
  - runs/history
  - (και στο PoC) patient consent registry

- Για PoC/ανάπτυξη:
  - μπορούμε να δουλεύουμε χωρίς πραγματικό blockchain node.
  - όταν BC_ENABLED=0 -> χρησιμοποιούμε NoopBlockchainService:
    - αποθηκεύει receipts στη SQLite (bc_receipts table)
    - κρατά per-process memory για patient consents (dataset_id -> patient_key -> allowed)

- Για πραγματική αλυσίδα:
  - όταν BC_ENABLED=1 και υπάρχουν σωστά env vars -> Web3BlockchainService:
    - συνδέεται σε RPC node (π.χ. Anvil / Hardhat / testnet)
    - καλεί smart contracts:
      - Receipt contract: anchorReceipt(...)
      - Consent contract: setConsent(...) / hasConsent(...)
    - επίσης αποθηκεύει receipt record στη SQLite για local audit.

Άρα:
- Το API layer (routes.py) καλεί get_blockchain() και στη συνέχεια bc.anchor(...) ή bc.set_patient_consent(...).
- Το store layer (sqlite_store.py) αποθηκεύει τα bc_receipts που δημιουργούνται εδώ.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from web3 import Web3

from app.auth import Actor
from app.services.sqlite_store import get_store


# -------------------------
# Hash helpers
# -------------------------
def _canonical_json(obj: Any) -> str:
    """
    Canonical JSON representation:
    - sort_keys=True ώστε το JSON να είναι deterministic (ίδια σειρά keys)
    - separators=(",", ":") ώστε να μη μπαίνουν κενά
    - default=str για datetime/UUID κ.λπ.
    - ensure_ascii=False για unicode
    Αυτό εξασφαλίζει ότι το sha256 hash είναι σταθερό για το ίδιο payload.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)


def sha256_hex(obj: Any) -> str:
    """
    Hash ενός arbitrary JSON-serializable αντικειμένου.
    Χρήση:
    - στο API layer, όταν θέλουμε να anchor hash (και όχι raw data).
    """
    s = _canonical_json(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def sha256_hex_str(s: str) -> str:
    """
    Hash ενός string όπως είναι (bytes of UTF-8).
    Χρήση:
    - patient_id -> patient_key_hex (bytes32) ώστε να μην ανεβαίνουν raw IDs στο chain.
    - dataset_id -> dataset_key (bytes32) για consent contract.
    """
    b = (s or "").encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _hex_to_bytes32(hex_str: str) -> bytes:
    """
    Μετατρέπει 64-hex string σε bytes32.
    - δέχεται και "0x..." prefix.
    """
    h = (hex_str or "").strip().lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) != 64:
        raise ValueError(f"Expected 64 hex chars for bytes32, got len={len(h)}")
    return bytes.fromhex(h)


def _load_contract_abi(path: str) -> list:
    """
    Φορτώνει ABI JSON από αρχείο.
    Περιμένουμε JSON με key "abi" (τυπικό output από foundry/hardhat artifacts).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    abi = data.get("abi")
    if not abi:
        raise ValueError(f"ABI not found in {path} (expected key 'abi')")
    return abi


# -------------------------
# Data structures
# -------------------------
@dataclass
class AnchorReceipt:
    """
    Unified “receipt” object που επιστρέφει το bc.anchor().

    Fields:
    - event_type, ref_id: semantic identifiers
    - payload_hash: sha256 hash του manifest (event+actor+payload)
    - tx_hash: αν είμαστε σε contract mode
    - chain_id: chain identifier
    - contract_address: ποιο contract έκανε anchor
    - receipt_id: internal receipt id (αν emitted από event logs)
    """
    event_type: str
    ref_id: str
    payload_hash: str
    tx_hash: Optional[str] = None
    chain_id: Optional[str] = None
    contract_address: Optional[str] = None
    receipt_id: Optional[str] = None


# -------------------------
# Interface / Abstract class
# -------------------------
class BlockchainService:
    """
    Abstract interface.

    Οποιαδήποτε υλοποίηση (Noop ή Web3) πρέπει να προσφέρει:
    - anchor()
    - set_patient_consent()
    - has_patient_consent()
    """

    def anchor(self, event_type: str, ref_id: str, payload: Dict[str, Any], actor: Actor) -> AnchorReceipt:
        raise NotImplementedError

    # --- Consent API (PoC) ---
    def set_patient_consent(self, dataset_id: str, patient_key_hex: str, allowed: bool, actor: Actor) -> Dict[str, Any]:
        raise NotImplementedError

    def has_patient_consent(self, dataset_id: str, patient_key_hex: str) -> bool:
        raise NotImplementedError


# -------------------------
# Noop implementation
# -------------------------
class NoopBlockchainService(BlockchainService):
    """
    Dev fallback (χωρίς πραγματικό blockchain).

    Τι κάνει:
    - anchor(): δημιουργεί manifest, υπολογίζει payload_hash,
      και αποθηκεύει ένα receipt στη SQLite (bc_receipts).
    - consents: κρατάει in-memory consent map (ανά process) για PoC.

    Σημαντικό limitation:
    - Τα consents εδώ ΔΕΝ είναι persistent (χάνονται σε restart).
    - Τα receipts όμως αποθηκεύονται persistent (SQLite).
    """

    # dataset_id -> patient_key_hex -> allowed
    _consent_mem: Dict[str, Dict[str, bool]] = {}

    def anchor(self, event_type: str, ref_id: str, payload: Dict[str, Any], actor: Actor) -> AnchorReceipt:
        """
        “Anchoring” σε noop mode:
        - Δεν γράφει σε chain.
        - Αποθηκεύει receipt στο sqlite store (bc_receipts).
        """
        manifest = {
            "event_type": event_type,
            "ref_id": ref_id,
            "actor": {"username": actor.username, "role": actor.role, "org": actor.org},
            "payload": payload,
        }
        payload_hash = sha256_hex(manifest)

        store = get_store()
        store.save_bc_receipt(
            event_type=event_type,
            ref_id=ref_id,
            payload={"payload_hash": payload_hash, "manifest": manifest, "mode": "noop"},
            tx_hash=None,
            chain_id=None,
        )

        return AnchorReceipt(
            event_type=event_type,
            ref_id=ref_id,
            payload_hash=payload_hash,
            tx_hash=None,
            chain_id=None,
            contract_address=None,
            receipt_id=None,
        )

    def set_patient_consent(self, dataset_id: str, patient_key_hex: str, allowed: bool, actor: Actor) -> Dict[str, Any]:
        """
        PoC patient consent storage in memory.

        Εισαγωγή:
        - patient_key_hex είναι sha256(patient_id) (64 hex chars) και όχι raw ID.

        Επιπλέον:
        - Κάνει και anchor ενός PATIENT_CONSENT_UPDATED event (noop),
          ώστε να υπάρχει audit trail στη SQLite.
        """
        ds = (dataset_id or "").strip()
        pk = (patient_key_hex or "").strip().lower()
        if pk.startswith("0x"):
            pk = pk[2:]
        if len(pk) != 64:
            raise ValueError("patient_key_hex must be 64 hex chars (bytes32)")

        self._consent_mem.setdefault(ds, {})[pk] = bool(allowed)

        # Audit receipt (noop) για αλλαγή consent
        self.anchor(
            event_type="PATIENT_CONSENT_UPDATED",
            ref_id=f"{ds}:{pk}",
            payload={"dataset_id": ds, "patient_key": "0x" + pk, "allowed": bool(allowed), "mode": "noop"},
            actor=actor,
        )

        return {"ok": True, "mode": "noop", "dataset_id": ds, "patient_key": "0x" + pk, "allowed": bool(allowed)}

    def has_patient_consent(self, dataset_id: str, patient_key_hex: str) -> bool:
        """
        Ελέγχει αν υπάρχει consent=True για dataset_id + patient_key.
        """
        ds = (dataset_id or "").strip()
        pk = (patient_key_hex or "").strip().lower()
        if pk.startswith("0x"):
            pk = pk[2:]
        return bool(self._consent_mem.get(ds, {}).get(pk, False))


# -------------------------
# Web3 (real chain) implementation
# -------------------------
class Web3BlockchainService(BlockchainService):
    """
    Real chain anchoring + patient consent registry.

    Απαιτούμενα env vars (όλα πρέπει να υπάρχουν):
      BC_ENABLED=1
      BC_RPC_URL
      BC_PRIVATE_KEY
      BC_CHAIN_ID

      Receipt Contract:
      BC_RECEIPT_CONTRACT_ADDRESS
      BC_RECEIPT_CONTRACT_ABI_PATH

      Consent Contract:
      BC_CONSENT_CONTRACT_ADDRESS
      BC_CONSENT_CONTRACT_ABI_PATH

    Ρόλοι contracts:
    - Receipt contract:
      anchorReceipt(eventType, refId, payloadHash(bytes32), actorHash(bytes32))
      εκπέμπει event ReceiptAnchored(receiptId,...)
    - Consent contract:
      setConsent(datasetKey(bytes32), patientKey(bytes32), allowed(bool))
      hasConsent(datasetKey, patientKey) -> bool
    """

    def __init__(
        self,
        rpc_url: str,
        private_key: str,
        chain_id: int,
        receipt_address: str,
        receipt_abi_path: str,
        consent_address: str,
        consent_abi_path: str,
    ):
        # Config
        self.rpc_url = rpc_url
        self.private_key = private_key
        self.chain_id = int(chain_id)

        # Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Blockchain RPC not reachable: {rpc_url}")

        # Account που υπογράφει τα tx (backend signer)
        self.acct = self.w3.eth.account.from_key(private_key)

        # Contract bindings
        self.receipt_address = Web3.to_checksum_address(receipt_address)
        self.consent_address = Web3.to_checksum_address(consent_address)

        receipt_abi = _load_contract_abi(receipt_abi_path)
        consent_abi = _load_contract_abi(consent_abi_path)

        self.receipt_contract = self.w3.eth.contract(address=self.receipt_address, abi=receipt_abi)
        self.consent_contract = self.w3.eth.contract(address=self.consent_address, abi=consent_abi)

    def _send_tx(self, fn) -> tuple[str, Any]:
        """
        Χτίζει και στέλνει signed transaction.

        - nonce: transaction count του signer
        - gas estimate: προσπαθεί estimate_gas, αλλιώς fallback
        - fees: προτιμά EIP-1559 αν υπάρχουν baseFeePerGas στοιχεία
        - wait_for_transaction_receipt: περιμένει confirmation για να πάρουμε blockNumber κ.λπ.
        """
        nonce = self.w3.eth.get_transaction_count(self.acct.address)

        tx = fn.build_transaction(
            {
                "chainId": self.chain_id,
                "from": self.acct.address,
                "nonce": nonce,
            }
        )

        # Gas handling
        try:
            est = self.w3.eth.estimate_gas(tx)
            tx["gas"] = int(est) + 25_000
        except Exception:
            tx["gas"] = 350_000

        # Fee handling (EIP-1559 preferred)
        try:
            latest = self.w3.eth.get_block("latest")
            base_fee = int(latest.get("baseFeePerGas", 0) or 0)
            priority = self.w3.to_wei(1, "gwei")
            tx["maxPriorityFeePerGas"] = priority
            tx["maxFeePerGas"] = base_fee + (2 * priority) + self.w3.to_wei(2, "gwei")
        except Exception:
            tx["gasPrice"] = self.w3.eth.gas_price

        signed = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash_bytes = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        tx_hash = self.w3.to_hex(tx_hash_bytes)

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=30)
        return tx_hash, receipt

    def anchor(self, event_type: str, ref_id: str, payload: Dict[str, Any], actor: Actor) -> AnchorReceipt:
        """
        Anchor event στο Receipt contract.

        Flow:
        1) Φτιάχνουμε manifest = {event_type, ref_id, actor, payload}
        2) payload_hash = sha256(manifest) -> bytes32
        3) actor_hash = sha256(actor fields) -> bytes32
        4) contract.anchorReceipt(...)
        5) διαβάζουμε event logs για receiptId (αν υπάρχει)
        6) αποθηκεύουμε local receipt στη SQLite (bc_receipts) για audit
        """
        manifest = {
            "event_type": event_type,
            "ref_id": ref_id,
            "actor": {"username": actor.username, "role": actor.role, "org": actor.org},
            "payload": payload,
        }

        payload_hash_hex = sha256_hex(manifest)
        payload_hash_b32 = _hex_to_bytes32(payload_hash_hex)

        actor_hash_hex = sha256_hex({"username": actor.username, "role": actor.role, "org": actor.org})
        actor_hash_b32 = _hex_to_bytes32(actor_hash_hex)

        fn = self.receipt_contract.functions.anchorReceipt(
            str(event_type),
            str(ref_id),
            payload_hash_b32,
            actor_hash_b32,
        )

        tx_hash, receipt = self._send_tx(fn)

        # Προσπάθεια εξαγωγής receiptId από event logs
        receipt_id_hex: Optional[str] = None
        try:
            events = self.receipt_contract.events.ReceiptAnchored().process_receipt(receipt)
            if events:
                rid = events[0]["args"].get("receiptId")
                receipt_id_hex = Web3.to_hex(rid)
        except Exception:
            receipt_id_hex = None

        # Block timestamp (αν δεν βγει, fallback σε time.time())
        try:
            block = self.w3.eth.get_block(receipt.blockNumber)
            block_ts = int(block.timestamp)
        except Exception:
            block_ts = int(time.time())

        # Local persistence (sqlite)
        store = get_store()
        store.save_bc_receipt(
            event_type=event_type,
            ref_id=ref_id,
            payload={
                "mode": "contract",
                "payload_hash": payload_hash_hex,
                "actor_hash": actor_hash_hex,
                "manifest": manifest,
                "tx_hash": tx_hash,
                "contract_address": self.receipt_address,
                "receipt_id": receipt_id_hex,
                "block_number": int(receipt.blockNumber),
                "block_timestamp": block_ts,
            },
            tx_hash=tx_hash,
            chain_id=str(self.chain_id),
        )

        return AnchorReceipt(
            event_type=event_type,
            ref_id=ref_id,
            payload_hash=payload_hash_hex,
            tx_hash=tx_hash,
            chain_id=str(self.chain_id),
            contract_address=self.receipt_address,
            receipt_id=receipt_id_hex,
        )

    # -------------------------
    # Consent (on-chain)
    # -------------------------
    def _dataset_key(self, dataset_id: str) -> bytes:
        """
        Δημιουργεί deterministic bytes32 datasetKey από dataset_id.
        - datasetKey = sha256(dataset_id string) -> bytes32
        """
        return _hex_to_bytes32(sha256_hex_str(dataset_id))

    def set_patient_consent(self, dataset_id: str, patient_key_hex: str, allowed: bool, actor: Actor) -> Dict[str, Any]:
        """
        Γράφει patient consent στο Consent contract.

        Flow:
        1) dataset_key = sha256(dataset_id) -> bytes32
        2) patient_key = bytes32(patient_key_hex) (που ήδη είναι sha256(patient_id))
        3) consent_contract.setConsent(dataset_key, patient_key, allowed)
        4) anchor audit receipt (PATIENT_CONSENT_UPDATED) στο Receipt contract
        """
        ds = (dataset_id or "").strip()
        if not ds:
            raise ValueError("dataset_id is required")

        pk = (patient_key_hex or "").strip().lower()
        if pk.startswith("0x"):
            pk = pk[2:]
        if len(pk) != 64:
            raise ValueError("patient_key_hex must be 64 hex chars (bytes32)")

        dataset_key = self._dataset_key(ds)
        patient_key = _hex_to_bytes32(pk)

        fn = self.consent_contract.functions.setConsent(dataset_key, patient_key, bool(allowed))
        tx_hash, receipt = self._send_tx(fn)

        # Anchor audit receipt (για traceability)
        self.anchor(
            event_type="PATIENT_CONSENT_UPDATED",
            ref_id=f"{ds}:{pk}",
            payload={
                "dataset_id": ds,
                "dataset_key": Web3.to_hex(dataset_key),
                "patient_key": "0x" + pk,
                "allowed": bool(allowed),
                "consent_contract": self.consent_address,
                "consent_tx_hash": tx_hash,
            },
            actor=actor,
        )

        return {
            "ok": True,
            "mode": "contract",
            "dataset_id": ds,
            "dataset_key": Web3.to_hex(dataset_key),
            "patient_key": "0x" + pk,
            "allowed": bool(allowed),
            "tx_hash": tx_hash,
            "consent_contract_address": self.consent_address,
            "block_number": int(receipt.blockNumber),
        }

    def has_patient_consent(self, dataset_id: str, patient_key_hex: str) -> bool:
        """
        Διαβάζει consent από Consent contract.
        - safe fallback: αν υπάρχει exception, επιστρέφει False.
        """
        ds = (dataset_id or "").strip()
        pk = (patient_key_hex or "").strip().lower()
        if pk.startswith("0x"):
            pk = pk[2:]
        if not ds or len(pk) != 64:
            return False

        dataset_key = self._dataset_key(ds)
        patient_key = _hex_to_bytes32(pk)
        try:
            return bool(self.consent_contract.functions.hasConsent(dataset_key, patient_key).call())
        except Exception:
            return False


# -------------------------
# Singleton accessor
# -------------------------
_BC: Optional[BlockchainService] = None


def get_blockchain() -> BlockchainService:
    """
    Factory + singleton cache για BlockchainService.

    Επιλογή mode:
    - BC_ENABLED=0 (default) -> NoopBlockchainService
    - BC_ENABLED=1 -> προσπαθεί Web3BlockchainService ΑΝ υπάρχουν όλα τα env vars

    Readiness checks:
    - κάνει ένα lightweight eth_chainId call στο RPC URL.
    - αν αποτύχει ή λείπουν vars -> fallback σε Noop
      (ώστε το backend να μην “πέφτει” αν δεν υπάρχει chain).
    """
    global _BC
    if _BC is not None:
        return _BC

    enabled = str(os.getenv("BC_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        _BC = NoopBlockchainService()
        return _BC

    rpc_url = (os.getenv("BC_RPC_URL") or "").strip()
    pk = (os.getenv("BC_PRIVATE_KEY") or "").strip()
    chain_id = int(os.getenv("BC_CHAIN_ID", "31337"))

    receipt_addr = (os.getenv("BC_RECEIPT_CONTRACT_ADDRESS") or "").strip()
    receipt_abi = (os.getenv("BC_RECEIPT_CONTRACT_ABI_PATH") or "").strip()

    consent_addr = (os.getenv("BC_CONSENT_CONTRACT_ADDRESS") or "").strip()
    consent_abi = (os.getenv("BC_CONSENT_CONTRACT_ABI_PATH") or "").strip()

    # Backward-compat: παλιότερα vars (αν υπήρχαν)
    if not receipt_addr:
        receipt_addr = (os.getenv("BC_CONTRACT_ADDRESS") or "").strip()
    if not receipt_abi:
        receipt_abi = (os.getenv("BC_CONTRACT_ABI_PATH") or "").strip()

    # Αν λείπει κάτι, πέφτουμε σε noop
    if not rpc_url or not pk or not receipt_addr or not receipt_abi or not consent_addr or not consent_abi:
        _BC = NoopBlockchainService()
        return _BC

    # Readiness probe στο RPC
    try:
        r = requests.post(
            rpc_url,
            json={"jsonrpc": "2.0", "method": "eth_chainId", "params": [], "id": 1},
            timeout=2,
        )
        if r.status_code != 200:
            _BC = NoopBlockchainService()
            return _BC
    except Exception:
        _BC = NoopBlockchainService()
        return _BC

    # Τελική προσπάθεια init Web3 service
    try:
        _BC = Web3BlockchainService(
            rpc_url=rpc_url,
            private_key=pk,
            chain_id=chain_id,
            receipt_address=receipt_addr,
            receipt_abi_path=receipt_abi,
            consent_address=consent_addr,
            consent_abi_path=consent_abi,
        )
        return _BC
    except Exception:
        _BC = NoopBlockchainService()
        return _BC
