from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    b = (s or "").encode("utf-8")
    return hashlib.sha256(b).hexdigest()

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

    # Evaluation metrics
    status: Optional[int] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    effective_gas_price: Optional[int] = None
    tx_cost_wei: Optional[int] = None
    latency_ms: Optional[int] = None




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
        t0 = time.perf_counter()
        # nothing on-chain
        t1 = time.perf_counter()

        store.save_bc_receipt(
            event_type=event_type,
            ref_id=ref_id,
            payload={
                "mode": "noop",
                "payload_hash": payload_hash,
                "manifest": manifest,

                "status": 1,
                "block_number": 0,
                "block_timestamp": int(time.time()),
                "gas_used": 0,
                "effective_gas_price": 0,
                "tx_cost_wei": 0,
                "latency_ms": int(round((t1 - t0) * 1000.0)),

                "offchain_compute_ms": int(round((t1 - t0) * 1000.0)),
                "payload_bytes": len(_canonical_json(manifest).encode("utf-8")),
                "hash_alg": "sha256",
            },
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
            status=1,
            block_number=0,
            gas_used=0,
            effective_gas_price=0,
            tx_cost_wei=0,
            latency_ms=int(round((t1 - t0) * 1000.0)),
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

    @staticmethod
    def _safe_get(receipt_obj: Any, key: str) -> Any:
        try:
            if hasattr(receipt_obj, "get"):
                return receipt_obj.get(key)
        except Exception:
            pass
        try:
            return receipt_obj[key]
        except Exception:
            return None

    def _send_tx(self, fn) -> tuple[str, Any, Dict[str, Any]]:
        nonce = self.w3.eth.get_transaction_count(self.acct.address)

        tx = fn.build_transaction(
            {
                "chainId": self.chain_id,
                "from": self.acct.address,
                "nonce": nonce,
            }
        )

        # Gas limit
        try:
            est = self.w3.eth.estimate_gas(tx)
            tx["gas"] = int(est) + 25_000
        except Exception:
            tx["gas"] = 350_000

        try:
            latest = self.w3.eth.get_block("latest")
            base_fee = int(latest.get("baseFeePerGas", 0) or 0)
            priority = int(self.w3.to_wei(1, "gwei"))
            tx["maxPriorityFeePerGas"] = priority
            tx["maxFeePerGas"] = base_fee + (2 * priority) + int(self.w3.to_wei(2, "gwei"))
        except Exception:
            tx["gasPrice"] = int(self.w3.eth.gas_price)

        signed = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)

        t_submit = time.perf_counter()
        tx_hash_bytes = self.w3.eth.send_raw_transaction(signed.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=30)
        t_done = time.perf_counter()

        tx_hash = self.w3.to_hex(tx_hash_bytes)
        latency_ms = int(round((t_done - t_submit) * 1000.0))

        # Receipt metrics
        gas_used = None
        effective_gas_price = None
        tx_cost_wei = None
        status = None
        block_number = None

        try:
            v = self._safe_get(receipt, "gasUsed")
            gas_used = int(v) if v is not None else None
        except Exception:
            gas_used = None

        try:
            v = self._safe_get(receipt, "effectiveGasPrice")
            effective_gas_price = int(v) if v is not None else None
        except Exception:
            effective_gas_price = None

        if effective_gas_price is None:
            try:
                if "gasPrice" in tx:
                    effective_gas_price = int(tx["gasPrice"])
                elif "maxFeePerGas" in tx:
                    # proxy (upper bound)
                    effective_gas_price = int(tx["maxFeePerGas"])
            except Exception:
                effective_gas_price = None

        if gas_used is not None and effective_gas_price is not None:
            try:
                tx_cost_wei = int(gas_used) * int(effective_gas_price)
            except Exception:
                tx_cost_wei = None

        try:
            v = self._safe_get(receipt, "status")
            status = int(v) if v is not None else None
        except Exception:
            status = None

        try:
            v = self._safe_get(receipt, "blockNumber")
            block_number = int(v) if v is not None else None
        except Exception:
            block_number = None

        # block timestamp
        block_timestamp = None
        try:
            bn = block_number
            if bn is None:
                bn = self._safe_get(receipt, "blockNumber")
                bn = int(bn) if bn is not None else None
            if bn:
                block = self.w3.eth.get_block(int(bn))
                # web3 returns AttributeDict; both styles work
                bt = getattr(block, "timestamp", None)
                if bt is None and hasattr(block, "get"):
                    bt = block.get("timestamp")
                block_timestamp = int(bt) if bt is not None else None
        except Exception:
            block_timestamp = None

        metrics = {
            "latency_ms": latency_ms,
            "gas_used": gas_used,
            "effective_gas_price": effective_gas_price,
            "tx_cost_wei": tx_cost_wei,
            "status": status,
            "block_number": block_number,
            "block_timestamp": block_timestamp,
        }

        return tx_hash, receipt, metrics

    def anchor(self, event_type: str, ref_id: str, payload: Dict[str, Any], actor: Actor) -> AnchorReceipt:
        t0_all = time.perf_counter()

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

        tx_hash, receipt, metrics = self._send_tx(fn)

        t1_all = time.perf_counter()
        offchain_compute_ms = int(round((t1_all - t0_all) * 1000.0))

        payload_bytes = len(_canonical_json(manifest).encode("utf-8"))

        bn = metrics.get("block_number")
        if bn is None:
            try:
                bn = int(getattr(receipt, "blockNumber", None) or receipt.get("blockNumber"))
            except Exception:
                bn = None

        receipt_id_hex: Optional[str] = None
        try:
            events = self.receipt_contract.events.ReceiptAnchored().process_receipt(receipt)
            if events:
                rid = events[0]["args"].get("receiptId")
                receipt_id_hex = Web3.to_hex(rid)
        except Exception:
            receipt_id_hex = None

        # Block timestamp
        block_ts = metrics.get("block_timestamp")
        if block_ts is None:
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

                "status": metrics.get("status"),
                "block_number": bn,
                "block_timestamp": block_ts,

                "gas_used": metrics.get("gas_used"),
                "effective_gas_price": metrics.get("effective_gas_price"),
                "tx_cost_wei": metrics.get("tx_cost_wei"),
                "latency_ms": metrics.get("latency_ms"),

                "offchain_compute_ms": offchain_compute_ms,
                "payload_bytes": payload_bytes,
                "hash_alg": "sha256",
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
            status=metrics.get("status"),
            block_number=bn,
            gas_used=metrics.get("gas_used"),
            effective_gas_price=metrics.get("effective_gas_price"),
            tx_cost_wei=metrics.get("tx_cost_wei"),
            latency_ms=metrics.get("latency_ms"),
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

        fn = self.consent_contract.functions.setRowConsent(dataset_key, patient_key, bool(allowed))

         #  START OFFCHAIN TIMER
        t0 = time.perf_counter()

        tx_hash, receipt, metrics = self._send_tx(fn)

        #  END OFFCHAIN TIMER
        t1 = time.perf_counter()
        offchain_compute_ms = int(round((t1 - t0) * 1000.0))

        bn = metrics.get("block_number")
        if bn is None:
            try:
                bn = int(getattr(receipt, "blockNumber", None) or receipt.get("blockNumber"))
            except Exception:
                bn = None

        # Dedicated receipt -> PATIENT_CONSENT_TX
        try:
            store = get_store()

            block_ts = metrics.get("block_timestamp")
            if block_ts is None:
                block_ts = int(time.time())

            payload = {
                "mode": "contract",
                "dataset_id": ds,
                "dataset_key": Web3.to_hex(dataset_key),
                "patient_key": "0x" + pk,
                "allowed": bool(allowed),

                "contract_address": self.consent_address,
                "tx_hash": tx_hash,

                "status": metrics.get("status"),
                "block_number": bn,
                "block_timestamp": block_ts,

                "gas_used": metrics.get("gas_used"),
                "effective_gas_price": metrics.get("effective_gas_price"),
                "tx_cost_wei": metrics.get("tx_cost_wei"),
                "latency_ms": metrics.get("latency_ms"),

                "offchain_compute_ms": offchain_compute_ms,
                "hash_alg": "sha256",
            }

            payload["payload_bytes"] = len(_canonical_json(payload).encode("utf-8"))

            store.save_bc_receipt(
                event_type="PATIENT_CONSENT_TX",
                ref_id=f"{ds}:{pk}",
                payload=payload,
                tx_hash=tx_hash,
                chain_id=str(self.chain_id),
            )
        except Exception:
            pass

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
                "gas_used": metrics.get("gas_used"),
                "tx_latency_ms": metrics.get("latency_ms"),
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
        row_id = _hex_to_bytes32(pk)

        try:
            exists, allowed, _ts = self.consent_contract.functions.getRowConsent(
                self.acct.address,
                dataset_key,
                row_id,
            ).call()

            return bool(exists and allowed)

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
