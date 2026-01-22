from __future__ import annotations

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


# Hash helpers

def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True, default=str)
                         # Εξασφαλίζει ότι το hash είναι σταθερό για το ίδιο payload

def sha256_hex(obj: Any) -> str:
    s = _canonical_json(obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()  # όταν θέλουμε να τραβήξουμε hash και όχι raw data


def sha256_hex_str(s: str) -> str:
    b = (s or "").encode("utf-8")  # όταν θέλουμε να τραβήξουμε hash και όχι raw data (για str)
    return hashlib.sha256(b).hexdigest()

# Μετατρέπει ένα hex hash (64 χαρακτήρες) σε bytes 32, γιατί τα smart contracts συνήθως θέλουν bytes32
def _hex_to_bytes32(hex_str: str) -> bytes:
    h = (hex_str or "").strip().lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) != 64:
        raise ValueError(f"Expected 64 hex chars for bytes32, got len={len(h)}")
    return bytes.fromhex(h)


def _load_contract_abi(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    abi = data.get("abi")
    if not abi:
        raise ValueError(f"ABI not found in {path} (expected key 'abi')")
    return abi




@dataclass
class AnchorReceipt:
    event_type: str
    ref_id: str
    payload_hash: str
    tx_hash: Optional[str] = None
    chain_id: Optional[str] = None
    contract_address: Optional[str] = None
    receipt_id: Optional[str] = None



class BlockchainService:   # κάθε συμβόλαιο πρέπει να έχει...

      # γράφει ένα audit event
    def anchor(self, event_type: str, ref_id: str, payload: Dict[str, Any], actor: Actor) -> AnchorReceipt:
        raise NotImplementedError
       # αποθηκεύει consent (allowed/denied)
    def set_patient_consent(self, dataset_id: str, patient_key_hex: str, allowed: bool, actor: Actor) -> Dict[str, Any]:
        raise NotImplementedError
      # ελέγχει αν υπάρχει consent
    def has_patient_consent(self, dataset_id: str, patient_key_hex: str) -> bool:
        raise NotImplementedError



class NoopBlockchainService(BlockchainService):

    _consent_mem: Dict[str, Dict[str, bool]] = {}

    def anchor(self, event_type: str, ref_id: str, payload: Dict[str, Any], actor: Actor) -> AnchorReceipt:

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

        ds = (dataset_id or "").strip()
        pk = (patient_key_hex or "").strip().lower()
        if pk.startswith("0x"):
            pk = pk[2:]
        if len(pk) != 64:
            raise ValueError("patient_key_hex must be 64 hex chars (bytes32)")

        self._consent_mem.setdefault(ds, {})[pk] = bool(allowed)

        # Audit receipt --> για αλλαγή consent
        self.anchor(
            event_type="PATIENT_CONSENT_UPDATED",
            ref_id=f"{ds}:{pk}",
            payload={"dataset_id": ds, "patient_key": "0x" + pk, "allowed": bool(allowed), "mode": "noop"},
            actor=actor,
        )

        return {"ok": True, "mode": "noop", "dataset_id": ds, "patient_key": "0x" + pk, "allowed": bool(allowed)}

    def has_patient_consent(self, dataset_id: str, patient_key_hex: str) -> bool:
        ds = (dataset_id or "").strip()
        pk = (patient_key_hex or "").strip().lower()
        if pk.startswith("0x"):
            pk = pk[2:]
        return bool(self._consent_mem.get(ds, {}).get(pk, False))  # Ελέγχει αν υπάρχει consent=True για dataset_id + patient_key



class Web3BlockchainService(BlockchainService):

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

        # Web3
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Blockchain RPC not reachable: {rpc_url}")


        self.acct = self.w3.eth.account.from_key(private_key)

        self.receipt_address = Web3.to_checksum_address(receipt_address)
        self.consent_address = Web3.to_checksum_address(consent_address)

        receipt_abi = _load_contract_abi(receipt_abi_path)
        consent_abi = _load_contract_abi(consent_abi_path)

        self.receipt_contract = self.w3.eth.contract(address=self.receipt_address, abi=receipt_abi)
        self.consent_contract = self.w3.eth.contract(address=self.consent_address, abi=consent_abi)

    def _send_tx(self, fn) -> tuple[str, Any]:
        nonce = self.w3.eth.get_transaction_count(self.acct.address)

        tx = fn.build_transaction(
            {
                "chainId": self.chain_id,
                "from": self.acct.address,
                "nonce": nonce,
            }
        )


        try:
            est = self.w3.eth.estimate_gas(tx)
            tx["gas"] = int(est) + 25_000
        except Exception:
            tx["gas"] = 350_000

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


        try:
            block = self.w3.eth.get_block(receipt.blockNumber)
            block_ts = int(block.timestamp)  # Block timestamp
        except Exception:
            block_ts = int(time.time())


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


    # Consent (on-chain)

    def _dataset_key(self, dataset_id: str) -> bytes:
        return _hex_to_bytes32(sha256_hex_str(dataset_id))

    def set_patient_consent(self, dataset_id: str, patient_key_hex: str, allowed: bool, actor: Actor) -> Dict[str, Any]:
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

        # audit receipt
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




_BC: Optional[BlockchainService] = None


def get_blockchain() -> BlockchainService:
    global _BC
    if _BC is not None:
        return _BC

    enabled = str(os.getenv("BC_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on") # BC_ENABLED=0 -> NoopBlockchainService
    if not enabled:
        _BC = NoopBlockchainService() # BC_ENABLED=1 -> Web3BlockchainService
        return _BC

    rpc_url = (os.getenv("BC_RPC_URL") or "").strip()
    pk = (os.getenv("BC_PRIVATE_KEY") or "").strip()
    chain_id = int(os.getenv("BC_CHAIN_ID", "31337"))

    receipt_addr = (os.getenv("BC_RECEIPT_CONTRACT_ADDRESS") or "").strip()
    receipt_abi = (os.getenv("BC_RECEIPT_CONTRACT_ABI_PATH") or "").strip()

    consent_addr = (os.getenv("BC_CONSENT_CONTRACT_ADDRESS") or "").strip()
    consent_abi = (os.getenv("BC_CONSENT_CONTRACT_ABI_PATH") or "").strip()

    if not receipt_addr:
        receipt_addr = (os.getenv("BC_CONTRACT_ADDRESS") or "").strip()
    if not receipt_abi:
        receipt_abi = (os.getenv("BC_CONTRACT_ABI_PATH") or "").strip()

    if not rpc_url or not pk or not receipt_addr or not receipt_abi or not consent_addr or not consent_abi:
        _BC = NoopBlockchainService()
        return _BC

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
