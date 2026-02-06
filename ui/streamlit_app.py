from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import requests
import streamlit as st
from PIL import Image


DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000/api/v1")
public_docs_url = os.getenv("PUBLIC_BACKEND_DOCS_URL", "http://localhost:8000/docs")

REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12")) # timeout αντί να κολλήσει

# Επιτρεπτοί ρόλοι χρηστών
ROLES = ["Admin", "Hospital", "Biobank", "Researcher"]

#  dropdowns
SENSITIVITY_LEVELS = ["low", "medium", "high"]
CONSENT_STATUS = ["draft", "active", "retired"]
REQUEST_STATUSES = ["submitted", "approved", "denied"]
EXPORT_METHODS = ["federated", "aggregated", "synthetic"]

# Ποιες σελίδες εμφανίζονται σε κάθε ρόλο
PAGES_BY_ROLE = {
    "Admin": ["Dashboard", "Nodes", "Datasets", "Consents", "Access Requests", "Federated Jobs", "Smart Contract", "Runs / History", "Settings"],
    "Hospital": ["Dashboard", "Nodes", "Datasets", "Consents", "Access Requests", "Federated Jobs", "Smart Contract", "Runs / History", "Settings"],
    "Biobank": ["Dashboard", "Nodes", "Datasets", "Access Requests", "Federated Jobs", "Smart Contract", "Runs / History", "Settings"],
    "Researcher": ["Dashboard", "Nodes", "Datasets", "Access Requests", "Federated Jobs", "Smart Contract", "Runs / History", "Settings"],
}



def role_norm() -> str:
    r = (st.session_state.get("role") or "").strip()
    mapping = {
        "hospital": "Hospital",
        "biobank": "Biobank",
        "researcher": "Researcher",
        "admin": "Admin",
    }
    return mapping.get(r.lower(), r)


def role() -> str:
    return st.session_state.get("role", "")


def org() -> str:
    return st.session_state.get("org", "")


# Helpers

def _assets_path(filename: str) -> str:  # path για files μέσα στο /assets (logo_1.png)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", filename)


def _load_image_optional(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None


def _auth_headers() -> Dict[str, str]:
    token = st.session_state.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}

def _dataset_label_by_id(datasets: list, did: str) -> str:
    did = str(did)
    for d in (datasets or []):
        if str(d.get("dataset_id")) == did:
            name = d.get("name") or "Unknown"
            owner = d.get("owner_org") or "UnknownOrg"
            return f"{name} | owner={owner} | {did}"
    return did


def api_get(path: str, params: Optional[dict] = None) -> Any:
    url = f"{DEFAULT_BASE_URL}{path}"
    r = requests.get(url, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else None


def api_post(path: str, payload: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    url = f"{DEFAULT_BASE_URL}{path}"
    r = requests.post(url, json=payload or {}, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else None


def api_patch(path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> Any:
    url = f"{DEFAULT_BASE_URL}{path}"
    r = requests.patch(url, json=payload, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"PATCH {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else None

def _receipt_visible_to_user(r: dict) -> bool:
    # Admin/Hospital  -> όλα
    # Biobank/Researcher -> περιορισμένα

    r_role = role_norm()
    if r_role in ("Admin", "Hospital"):
        return True

    my_user = (st.session_state.get("username") or "").strip().lower()
    my_org = (st.session_state.get("org") or "").strip().lower()

    payload = r.get("payload") or {}
    manifest = payload.get("manifest") or {}
    actor = manifest.get("actor") or {}

    a_user = (actor.get("username") or "").strip().lower()
    a_org = (actor.get("org") or "").strip().lower()

    if my_user and a_user == my_user:
        return True
    if my_org and a_org == my_org:
        return True
    return False


def _json_size_kb(obj: Any) -> float:
    try:
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        return round(len(b) / 1024, 2)
    except Exception:
        return 0.0

def log_run(run_type: str, payload: Dict[str, Any]) -> None:  # Καταγράφει “ιστορικό ενεργειών” (Runs / History) στο backend
    try:
        api_post("/runs", payload={"run_type": run_type, "payload": payload})
    except Exception:
        pass


def require_login() -> None: # stop αν δεν δηλωθεί σωστό authentication
    if not st.session_state.get("token"):
        st.info("Please login to continue.")
        st.stop()


def _dataset_columns(ds: Dict[str, Any]) -> List[str]:
    cols = ds.get("columns") or []
    return [str(x) for x in cols if str(x).strip()] #  Επιστρέφει τα columns που έχει επιστρέψει το backend μετά από validate.


def _dataset_exposed_features(ds: Dict[str, Any]) -> List[str]:
    feats = ds.get("exposed_features")
    if feats is None:
        return []
    return [str(x) for x in (feats or []) if str(x).strip()]

def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def _rel_err(a, b, eps=1e-12):
    fa = _safe_float(a)
    fb = _safe_float(b)
    if fa is None or fb is None:
        return None
    return abs(fa - fb) / (abs(fb) + eps)

def _compare_feature_metrics(ma: dict, mb: dict) -> pd.DataFrame:
    fa = (ma.get("feature_metrics") or ma.get("feature_stats") or {}) if isinstance(ma, dict) else {}
    fb = (mb.get("feature_metrics") or mb.get("feature_stats") or {}) if isinstance(mb, dict) else {}

    feats = sorted(set(fa.keys()) | set(fb.keys()))
    rows = []
    for f in feats:
        a = fa.get(f) or {}
        b = fb.get(f) or {}
        if not isinstance(a, dict): a = {}
        if not isinstance(b, dict): b = {}

        for key in ["mean", "std", "min", "max", "missing_rate", "variance"]:
            va = a.get(key)
            vb = b.get(key)
            rows.append({
                "feature": f,
                "metric": key,
                "A": _safe_float(va),
                "B": _safe_float(vb),
                "abs_diff": None if (_safe_float(va) is None or _safe_float(vb) is None) else abs(float(va) - float(vb)),
                "rel_error": _rel_err(va, vb) if key in ("mean","std","variance") else None,
            })
    df = pd.DataFrame(rows)
    return df

def _compare_corr(ma: dict, mb: dict) -> dict:
    ca = ma.get("correlation_matrix") if isinstance(ma, dict) else None
    cb = mb.get("correlation_matrix") if isinstance(mb, dict) else None
    if not isinstance(ca, dict) or not isinstance(cb, dict) or not ca or not cb:
        return {"ok": False, "reason": "missing correlation_matrix"}

    da = pd.DataFrame(ca).apply(pd.to_numeric, errors="coerce")
    db = pd.DataFrame(cb).apply(pd.to_numeric, errors="coerce")

    common = sorted(set(da.columns) & set(db.columns))
    if not common:
        return {"ok": False, "reason": "no common correlation features"}

    da = da.reindex(index=common, columns=common)
    db = db.reindex(index=common, columns=common)

    diff = (da - db).abs()
    mad = float(diff.stack().mean(skipna=True))
    mx = float(diff.stack().max(skipna=True))
    fro = float(np.sqrt(np.nansum((da.values - db.values) ** 2)))

    return {
        "ok": True,
        "common_features": len(common),
        "corr_mad": round(mad, 6),
        "corr_max_abs": round(mx, 6),
        "corr_fro_norm": round(fro, 6),
    }

def _topk_overlap(da: dict, db: dict, k: int = 10) -> dict:
    if not isinstance(da, dict) or not isinstance(db, dict) or not da or not db:
        return {"ok": False, "reason": "missing normalized_feature"}

    sa = pd.Series(da).apply(pd.to_numeric, errors="coerce").dropna().sort_values(ascending=False)
    sb = pd.Series(db).apply(pd.to_numeric, errors="coerce").dropna().sort_values(ascending=False)

    topa = list(sa.head(k).index.astype(str))
    topb = list(sb.head(k).index.astype(str))
    inter = set(topa) & set(topb)
    union = set(topa) | set(topb)
    jacc = (len(inter) / len(union)) if union else None

    return {
        "ok": True,
        "k": int(k),
        "topk_overlap": int(len(inter)),
        "topk_jaccard": None if jacc is None else round(float(jacc), 4),
        "topA": topa,
        "topB": topb,
    }


def _features_available_to_requester(ds: Dict[str, Any]) -> List[str]:
    """
    Ορίζει ποια features μπορεί να επιλέξει ο χρήστης όταν δημιουργεί Federated Job.
    - Για Biobank/Researcher: μονο exposed_features
    - Για Hospital/Admin: πριν το στάδιο exposed_features
    """
    exposed = _dataset_exposed_features(ds)
    cols = _dataset_columns(ds)
    if role_norm() in ("Researcher", "Biobank"):
        return exposed
    return exposed or cols


def suggest_actions(metrics: dict) -> list[str]:
    actions: list[str] = []

    # Data quality
    feature_stats = (metrics.get("feature_metrics") or metrics.get("feature_stats") or {})
    if isinstance(feature_stats, dict) and feature_stats:
        for feat, s in feature_stats.items():
            if not isinstance(s, dict):
                continue
            mr = float(s.get("missing_rate") or 0.0)
            orr = float(s.get("outlier_rate") or 0.0)

            if mr >= 0.05:
                actions.append(f"[Data Quality] '{feat}' έχει missing_rate={mr:.2%}: πρότεινε imputation ή exclusion.")
            if orr >= 0.03:
                actions.append(f"[Data Quality] '{feat}' έχει outlier_rate={orr:.2%}: πρότεινε winsorization / robust scaling.")

            if s.get("is_constant") is True:
                actions.append(f"[Data Quality] '{feat}' φαίνεται constant/zero: πρότεινε αφαίρεση feature.")

    # Privacy governance
    privacy = metrics.get("privacy") or {}
    if isinstance(privacy, dict) and privacy:
        thr = privacy.get("min_row_threshold")
        is_suppressed = bool(privacy.get("suppressed") is True)

        if is_suppressed:
            actions.append(
                "[Privacy] Υπήρξε suppression σε τουλάχιστον έναν συμμετέχοντα ή/και στο συνολικό N. "
                "Πρότεινε περισσότερα nodes ή μεγαλύτερα datasets."
            )

        if thr is not None:
            actions.append(f"[Privacy] Ελάχιστο threshold={thr}. Πρότεινε rule: μην εμφανίζεις stats για N<threshold.")

    # Rounds μελλοντικά με ml
    trends = metrics.get("round_trends") or {}
    if isinstance(trends, dict) and trends:
        actions.append("[Trends] Υπάρχουν round_trends: δείξε convergence plot ή μείωσε rounds αν συγκλίνει γρήγορα.")

    return actions


# Flash messages

def flash_set(kind: str, text: str) -> None:
    st.session_state["flash_msg"] = {"kind": kind, "text": text}


def flash_clear() -> None:
    st.session_state.pop("flash_msg", None)


def flash_show() -> None:
    msg = st.session_state.get("flash_msg")
    if not msg:
        return
    kind = msg.get("kind", "info")
    text = msg.get("text", "")
    if kind == "success":
        st.success(text)
    elif kind == "warning":
        st.warning(text)
    elif kind == "error":
        st.error(text)
    else:
        st.info(text)


# UI: Auth

def ui_login_register() -> None:

    st.title("Πλατφόρμα Συνεργατικής Ιατρικής Ανάλυσης / BC-FL")
    st.markdown("### ")

    # Layout για logos (κεντραρισμένα)
    f1, f2, f3 = st.columns([1, 2, 1])
    with f2:
        s1, gap, s2 = st.columns([1, 0.25, 1])
        with s1:
            st.image(_assets_path("logo_1.png"), width=440)
        with gap:
            st.empty()
        with s2:
            st.image(_assets_path("logo_2.jpg"), width=440)

    # Δύο στήλες: αριστερά login, δεξιά register
    c1, c2 = st.columns(2)


    # Login

    with c1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        # Το κουμπί Login κάνει call στο backend για token.
        if st.button("Login", type="primary"):
            try:

                resp = api_post("/auth/login", {"username": username, "password": password})
                st.session_state["token"] = resp["access_token"]

                user = resp["user"]
                st.session_state["username"] = user["username"]
                st.session_state["role"] = user["role"]
                st.session_state["org"] = user["org"]

                st.success("Logged in.")

                # καταγραφή run history
                log_run("auth.login", {"username": username, "role": user.get("role"), "org": user.get("org")})

                # rerun για να αλλάξει σε logged in  και να ανοίξει navigation
                st.rerun()

            except Exception as e:
                # Σε αποτυχία login -> clear keys
                for k in ("token", "username", "role", "org"):
                    st.session_state.pop(k, None)
                st.error(str(e))


    # Register

    with c2:
        st.subheader("Register")
        r_username = st.text_input("New username", key="reg_username")
        r_password = st.text_input("New password", type="password", key="reg_password")
        r_role = st.selectbox("Role", ROLES, index=1, key="reg_role")
        r_org = st.text_input(
            "Organization",
            key="reg_org",
            placeholder="e.g., Hospital A / Biobank Center / Research Lab",
        )

        # Για Admin/Hospital/Biobank απαιτείται invite_code
        invite_code = ""
        if r_role in ("Admin", "Hospital", "Biobank"):
            invite_code = st.text_input(
                "Invite code",
                key="reg_invite_code",
                help="Required for Admin/Hospital/Biobank registration.",
            )

        if st.button("Register"):
            try:
                payload = {"username": r_username, "password": r_password, "role": r_role, "org": r_org}
                if r_role in ("Admin","Hospital", "Biobank"):
                    payload["invite_code"] = invite_code

                api_post("/auth/register", payload)

                st.success("Registered. You can now login.")
                log_run("auth.register", {"username": r_username, "role": r_role, "org": r_org})

            except Exception as e:
                st.error(str(e))

    st.divider()


    # Public Patient Consent
    st.subheader("Patient Consent (Public Portal)")
    patient_id = st.text_input("Patient Pseudonymous ID (e.g., PAT-0001)", key="pt_patient_id")
    dataset_id = st.text_input("Dataset ID", key="pt_dataset_id", help="Paste the dataset_id you were given.")
    decision = st.selectbox("Decision", ["allow", "deny"], key="pt_decision")
    portal_secret = st.text_input(
        "Portal Secret",
        type="password",
        key="pt_secret",
        help="Set PATIENT_PORTAL_SECRET in backend env.",
    )

    # Όταν πατηθεί Submit
    if st.button("Submit Consent", key="pt_submit", type="primary"):
        try:
            payload = {
                "dataset_id": dataset_id,
                "patient_id": patient_id,
                "decision": decision,
                "secret": portal_secret,
            }
            resp = api_post("/public/consent", payload)
            st.success("Consent recorded on-chain")

            st.code(f"""
            event_type        : {resp.get('event_type')}
            ref_id            : {resp.get('ref_id')}
            tx_hash           : {resp.get('tx_hash')}
            chain_id          : {resp.get('chain_id')}
            contract_address  : {resp.get('contract_address')}
            payload_hash      : {resp.get('payload_hash')}
            """, language="text")
        except Exception as e:
            st.error(str(e))


def ui_topbar() -> None: # πάνω τμήμα της σελίδας όταν συνδεθείς
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        st.markdown("[API Docs](http://localhost:8000/docs)")
    with c2:
        if st.session_state.get("token"):
            st.write(
                f"User: **{st.session_state.get('username')}** | "
                f"Role: **{role_norm()}** | Org: **{org()}**"
            )
    with c3:
        if st.session_state.get("token"):
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()



# Pages

def page_dashboard() -> None:
    st.header("Dashboard")
    f1, f2, f3 = st.columns([1, 3, 1])
    with f2:
        s1, gap, s2 = st.columns([1, 0.25, 1])
        with s1:
            st.image(_assets_path("logo_1.png"), width=440)
        with gap:
            st.empty()
        with s2:
            st.image(_assets_path("logo_2.jpg"), width=440)

    st.markdown("<br>", unsafe_allow_html=True)

    require_login()
    flash_show()

    try:
        health = api_get("/health")  # Health check προς backend API
        st.success(f"Backend status: {health.get('status', 'unknown')}")
    except Exception as e:
        st.error(str(e))

    st.write(
        """
This platform supports federated-only descriptors:
- Hospitals register dataset descriptors to local files hosted by their agents.
- Biobanks and Researchers request access.
- Hospitals approve/deny.
- Approved parties can run federated jobs (no raw data transfer).
        """
    )


def page_nodes() -> None:
    st.header("Nodes (Hospital Agents)")
    require_login()
    flash_show()

    try:
        nodes = api_get("/nodes")
    except Exception as e:
        st.error(str(e))
        return

    if not nodes:
        st.info("No nodes registered yet.")
        return

    st.write("Registered nodes:")
    st.dataframe(nodes, use_container_width=True)


    if role_norm() == "Hospital":
        st.caption("If you changed Docker compose service names, ensure node.base_url matches the internal Docker DNS name (e.g., http://bc-fl-hospital-a-agent:9001).")


def page_datasets() -> None:
    st.header("Datasets Descriptors")
    require_login()
    flash_show()

    # Hospital-only -> Create dataset descriptor
    if role_norm() == "Hospital":
        st.subheader("Create dataset descriptor")
        try:
            # Παίρνουμε nodes ώστε ο Hospital να διαλέξει πού θα ανεβάσει το αρχείο
            nodes = api_get("/nodes")
        except Exception as e:
            st.error(str(e))
            nodes = []

        if not nodes:
            st.warning("No nodes available. Register nodes first.")
        else:
            # Dropdown
            node_opts = {f"{n['name']} ({n['org']})": n["node_id"] for n in nodes}
            selected_node_label = st.selectbox("Hosting node", list(node_opts.keys()), key="ds_host_node_sel")
            selected_node_id = node_opts[selected_node_label]

            # Φόρμα metadata του descriptor
            name = st.text_input("Name", value="Admissions Dataset", key="ds_name")
            description = st.text_area(
                "Description",
                value="Federated descriptor pointing to local hospital dataset.",
                key="ds_desc",
            )
            sensitivity = st.selectbox("Sensitivity", SENSITIVITY_LEVELS, index=0, key="ds_sens")
            schema_id = st.text_input("Schema ID", value="admissions_v1", key="ds_schema")

            uploaded = st.file_uploader("Attach dataset file (CSV or Excel)", type=["csv", "xlsx"], key="ds_uploader")

            if uploaded is not None:
                st.info(f"Selected file: {uploaded.name}")

                # Upload file στο agent
                if st.button("Upload to hosting node", type="secondary", key="ds_upload_btn"):
                    try:

                        node = next(n for n in nodes if str(n["node_id"]) == str(selected_node_id))
                        agent_base = node["base_url"].rstrip("/")

                        files = {"file": (uploaded.name, uploaded.getvalue())}
                        r = requests.post(
                            f"{agent_base}/upload",
                            files=files,
                            headers={"X-Agent-Secret": os.getenv("AGENT_REG_SECRET", "dev-secret")},
                            timeout=REQUEST_TIMEOUT,
                        )
                        if r.status_code >= 400:
                            raise RuntimeError(f"Upload failed: {r.status_code} {r.text}")

                        # Ο agent επιστρέφει local_uri
                        st.session_state["uploaded_local_uri"] = r.json().get("local_uri", "")
                        st.success(f"Uploaded to: {st.session_state['uploaded_local_uri']}")
                    except Exception as e:
                        st.error(str(e))


            local_uri = st.session_state.get("uploaded_local_uri", "")
            st.text_input("Local URI (inside agent container)", value=local_uri, disabled=True, key="ds_local_uri")

            # Create descriptor στο backend
            if st.button("Create descriptor", type="primary", key="ds_create_btn"):
                try:
                    if not local_uri:
                        st.error("Upload a file first (Local URI is empty).")
                        st.stop()

                    payload = {
                        "name": name,
                        "description": description,
                        "owner_org": org(),
                        "sensitivity_level": sensitivity,
                        "schema_id": schema_id,
                        "local_uri": local_uri,
                        "node_id": selected_node_id,
                    }
                    ds = api_post("/datasets", payload)
                    st.success("Descriptor created.")

                    # Καταγραφή -> history
                    log_run(
                        "datasets.create",
                        {
                            "dataset_id": ds.get("dataset_id"),
                            "node_id": selected_node_id,
                            "local_uri": local_uri,
                            "schema_id": schema_id,
                        },
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.divider()


    # List datasets (όλοι οι ρόλοι)

    st.subheader("List datasets")
    try:
        datasets = api_get("/datasets")
    except Exception as e:
        st.error(str(e))
        return

    if not datasets:
        st.info("No datasets found.")
        return

    st.dataframe(datasets, use_container_width=True)


    # Hospital only -->  Validate + Expose features
    if role_norm() == "Hospital":
        st.subheader("Validate dataset descriptor")
        ds_map = {f"{d['name']} | {d['dataset_id']}": d["dataset_id"] for d in datasets}
        selected_label = st.selectbox("Select dataset", list(ds_map.keys()), key="ds_validate_sel")
        selected_id = ds_map[selected_label]

        # Validate
        if st.button("Validate", type="secondary", key="ds_validate_btn"):
            try:
                ds = api_post(f"/datasets/{selected_id}/validate", payload={})
                st.success(f"Validated. status={ds.get('status')}, rows={ds.get('row_count')}")

                log_run(
                    "datasets.validate",
                    {
                        "dataset_id": ds.get("dataset_id"),
                        "status": ds.get("status"),
                        "row_count": ds.get("row_count"),
                        "columns": ds.get("columns"),
                    },
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))

        ds_current = next((d for d in datasets if str(d.get("dataset_id")) == str(selected_id)), None)
        if not ds_current:
            st.info("Select a dataset above.")
            return

        cols = ds_current.get("columns") or []
        exposed = ds_current.get("exposed_features") or []

        st.subheader("Expose features to Biobank/Researcher")
        if not cols:
            st.warning("This dataset has no columns yet. Run Validate first to extract column names from the uploaded file.")
        else:
            st.caption("Select which columns will be visible as selectable features to Biobank/Researcher.")
            default_vals = exposed if exposed else cols
            chosen = st.multiselect(
                "Allowed features (exposed_features)",
                options=cols,
                default=[c for c in default_vals if c in cols],
                key="ds_exposed_multiselect",
            )

            # Save exposed features στο backend
            if st.button("Save exposed features", type="primary", key="ds_save_exposed_btn"):
                try:
                    updated = api_patch(f"/datasets/{selected_id}/features", payload={"exposed_features": chosen})
                    st.success(f"Saved exposed features: {len(updated.get('exposed_features') or [])}")
                    log_run("datasets.exposed_features", {"dataset_id": selected_id, "exposed_features": chosen})
                    st.rerun()
                except Exception as e:
                    st.error(str(e))


def page_consents() -> None:
    st.header("Consent Policies")
    require_login()
    flash_show()

    if role_norm() != "Hospital":
        st.info("Consent management is Hospital only")
        return

    try:
        datasets = api_get("/datasets")
    except Exception as e:
        st.error(str(e))
        return

    if not datasets:
        st.warning("No datasets available. Create a dataset descriptor first.")
        return

    st.subheader("Create consent policy")
    ds_map = {f"{d['name']} | {d['dataset_id']}": d["dataset_id"] for d in datasets}
    ds_sel = st.selectbox("Dataset", list(ds_map.keys()))
    policy_text = st.text_area("Policy text", value="Approved for federated analytics. No raw data leaves hospital boundary.")
    status = st.selectbox("Status", CONSENT_STATUS, index=1)
    allow_external = st.checkbox("Allow external parties", value=True)
    allowed_roles = st.multiselect("Allowed roles", ["Researcher", "Biobank"], default=["Researcher", "Biobank"])
    export_methods = st.multiselect("Allowed export methods", EXPORT_METHODS, default=["federated"])

    if st.button("Create policy", type="primary"):
        try:
            payload = {
                "dataset_id": ds_map[ds_sel],
                "policy_text": policy_text,
                "status": status,
                "allow_external": allow_external,
                "allowed_roles": allowed_roles,
                "expiry_days": None,
                "policy_structured": None,
                "export_methods": export_methods,
            }
            cp = api_post("/consents", payload)
            st.success("Consent policy created.")

            log_run(
                "consents.create",
                {
                    "policy_id": cp.get("policy_id"),
                    "dataset_id": cp.get("dataset_id"),
                    "status": cp.get("status"),
                    "allowed_roles": allowed_roles,
                    "export_methods": export_methods,
                },
            )
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("List consent policies")
    try:
        cps = api_get("/consents")
    except Exception as e:
        st.error(str(e))
        return
    st.dataframe(cps or [], use_container_width=True)


def page_access_requests() -> None:
    st.header("Access Requests")
    require_login()
    flash_show()


    # Submit access request
    if role_norm() in ("Researcher", "Biobank"):
        st.subheader("Submit access request")
        try:
            datasets = api_get("/datasets")
        except Exception as e:
            st.error(str(e))
            datasets = []

        if not datasets:
            st.warning("No datasets available.")
        else:
            ds_map = {f"{d['name']} | {d['owner_org']} | {d['dataset_id']}": d["dataset_id"] for d in datasets}
            ds_sel = st.selectbox("Dataset", list(ds_map.keys()))
            purpose = st.text_area("Purpose", value="Federated analysis for research/biobank purposes under GDPR.")
            notes = st.text_input("Notes (optional)", value="")

            if st.button("Submit request", type="primary"):
                try:
                    payload = {
                        "dataset_id": ds_map[ds_sel],
                        "requester_org": org(),
                        "purpose": purpose,
                        "requested_by": st.session_state.get("username"),
                        "role": role_norm(),
                        "notes": notes or None,
                    }
                    req = api_post("/access-requests", payload)
                    st.success("Request submitted.")

                    log_run(
                        "access_requests.submit",
                        {
                            "request_id": req.get("request_id"),
                            "dataset_id": req.get("dataset_id"),
                            "requester_org": org(),
                            "role": role_norm(),
                            "requested_by": st.session_state.get("username"),
                        },
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.divider()


    # List access requests

    st.subheader("List access requests")
    try:
        items = api_get("/access-requests")
    except Exception as e:
        st.error(str(e))
        return

    if not items:
        st.info("No access requests found.")
        return

    st.dataframe(items, use_container_width=True)


    # Hospital -> Approve / Deny

    if role_norm() == "Hospital":
        st.subheader("Approve / Deny")
        req_map = {
            f"{r['status']} | {r['request_id']} | dataset={r['dataset_id']} | org={r['requester_org']} | by={r['requested_by']}": r
            for r in items
        }
        selected_label = st.selectbox("Select request", list(req_map.keys()), key="ar_selected_req", on_change=flash_clear)
        selected_req = req_map[selected_label]

        decision = st.selectbox("Decision", ["approved", "denied"], index=0, key="ar_decision", on_change=flash_clear)
        decision_notes = st.text_input(
            "Decision notes",
            value="Approved for federated computation only.",
            key="ar_decision_notes",
            on_change=flash_clear,
        )

        if st.button("Apply decision", type="secondary"):
            try:
                updated = api_patch(
                    f"/access-requests/{selected_req['request_id']}/decision",
                    params={"decision": decision, "notes": decision_notes or None},
                )
                flash_set("success", f"Updated: status={updated.get('status')}")

                log_run(
                    "access_requests.decision",
                    {
                        "request_id": updated.get("request_id"),
                        "decision": updated.get("status"),
                        "dataset_id": updated.get("dataset_id"),
                        "notes": decision_notes,
                    },
                )
                st.rerun()
            except Exception as e:
                flash_set("error", str(e))
                st.rerun()


def _is_request_approved_for_user(dataset_id: str) -> bool:

    if role_norm() not in ("Researcher", "Biobank"):
        return True

    try:
        reqs = api_get("/access-requests")
    except Exception:
        return False

    username = (st.session_state.get("username") or "").strip().lower()
    for r in reqs or []:
        if str(r.get("dataset_id")) != str(dataset_id):
            continue
        if (r.get("status") or "") != "approved":
            continue
        if (r.get("requested_by") or "").strip().lower() == username:
            return True
    return False

def page_federated_jobs() -> None:

    st.header("Federated Jobs")
    require_login()
    flash_show()
    st.caption(
        "This Platform runs federated computation against the hosting Hospital Agent. "
        "No raw dataset leaves the hospital boundary."
    )

    try:
        datasets = api_get("/datasets")
    except Exception as e:
        st.error(str(e))
        return

    if not datasets:
        st.warning("No datasets available.")
        return

    # Create job
    st.subheader("Create job")

    scope = st.selectbox(
        "Scope",
        options=["single_node", "multi_node"],
        format_func=lambda x: "Single hospital (1 node)" if x == "single_node" else "Multi-hospital (many nodes)",
        key="fl_scope",
    )

    ds_map = {
        f"{d['name']} | owner={d['owner_org']} | {d['dataset_id']}": d["dataset_id"]
        for d in datasets
    }
    ds_labels = list(ds_map.keys())

    # Επιλογή dataset(s)
    dataset_id = None
    dataset_ids = []

    if scope == "single_node":
        ds_sel = st.selectbox("Dataset", ds_labels, key="fl_ds_single")
        dataset_id = ds_map[ds_sel]
        dataset_ids = [dataset_id]

    else:
        ds_sel_multi = st.multiselect("Datasets (select 2+)", ds_labels, key="fl_ds_multi")
        dataset_ids = [ds_map[x] for x in ds_sel_multi]
        dataset_id = dataset_ids[0] if dataset_ids else None

    def _avail(ds: dict) -> list:
        return _features_available_to_requester(ds or {})

    chosen_ds = [d for d in datasets if str(d.get("dataset_id")) in {str(x) for x in (dataset_ids or [])}]

    if scope == "single_node":
        ds_current = chosen_ds[0] if chosen_ds else None
        available_features = _avail(ds_current or {})
    else:
        feature_sets = [set(_avail(d)) for d in chosen_ds]
        if feature_sets:
            common = set.intersection(*feature_sets)
            union = set.union(*feature_sets)

            available_features = sorted(list(common))

            st.caption(f"Common features across selected datasets: {len(common)}")

            for d in chosen_ds:
                dset = set(_avail(d))
                only_this = sorted(list(dset - common))
                if only_this:
                    st.info(
                        f"Only in {d.get('name')} (owner={d.get('owner_org')}): "
                        + ", ".join(only_this[:25])
                        + (" ..." if len(only_this) > 25 else "")
                    )
        else:
            available_features = []

    if scope == "multi_node" and len(dataset_ids) < 2:
        st.info("Select at least 2 datasets for multi-hospital mode.")

    rounds = st.number_input("Rounds", min_value=1, max_value=50, value=3, step=1)

    if role_norm() not in ("Researcher", "Biobank", "Hospital","Admin"):
        st.error("Only Hospital, Researcher and Biobank can create FL jobs.")
        st.stop()

    selected_features = st.multiselect(
        "Features (select one or more)",
        options=available_features,
        default=available_features[:2] if len(available_features) >= 2 else available_features,
    )

    label = st.text_input("Label (optional)", value="")
    notes = st.text_area("Notes", value="Compute federated statistics.")

    if st.button("Create FL Job", type="primary"):  # Κουμπί δημιουργίας job
        try:
            missing = []
            for did in (dataset_ids or []):
                if not _is_request_approved_for_user(did):
                    missing.append(_dataset_label_by_id(datasets, did))

            if missing:
                st.error(
                    "You need an APPROVED access request for ALL selected datasets:\n\n"
                    + "\n".join([f"- {x}" for x in missing])
                )
                st.stop()

            if not selected_features:
                st.error("Select at least one feature.")  # Validation: πρέπει να έχει επιλεγεί τουλάχιστον 1 feature
                st.stop()

            if not dataset_ids:
                st.error("Select at least one dataset.")
                st.stop()

            if scope == "multi_node" and len(dataset_ids) < 2:
                st.error("Multi-hospital mode requires at least 2 datasets.")
                st.stop()

            dataset_id = dataset_ids[0]

            payload = {
                "dataset_id": dataset_id,
                "dataset_ids": dataset_ids,
                "scope": scope,
                "rounds": int(rounds),
                "features": selected_features,
                "label": label.strip() or None,
                "notes": notes,
            }

            job = api_post("/fl/jobs", payload)

            st.success(f"Job created: {job.get('job_id')}")

            log_run(
                "fl_jobs.create",
                {
                    "job_id": job.get("job_id"),
                    "dataset_id": dataset_id,
                    "rounds": int(rounds),
                    "features": selected_features,
                },
            )

            st.session_state["last_job_id"] = job.get("job_id")

            st.rerun()

        except Exception as e:
            st.error(str(e))

    # Run job
    st.divider()
    st.subheader("Run job")

    st.subheader("Display options")
    sections = st.multiselect(
        "Select which analytics sections to display",
        options=[
            "Performance Summary",
            "Telemetry",
            "Privacy Governance",
            "Feature Metrics",
            "Normalized Feature",
            "Round Trends",
            "Correlation Matrix",
            #"Raw JSON (debug)",
        ],
        key="fl_display_sections",
    )

    job_id = st.text_input(
        "Job ID",
        value=st.session_state.get("last_job_id", "") or ""
    )

    if st.button("Start job", type="secondary"):
        try:
            if not job_id.strip():
                st.error("Provide a Job ID")
                st.stop()

            # Timer
            t0 = time.perf_counter()
            job = api_post(f"/fl/jobs/{job_id.strip()}/start", payload={})
            t1 = time.perf_counter()
            ui_duration_sec = round(t1 - t0, 4)

            st.success(
                f"Job finished: status={job.get('status')}, "
                f"round={job.get('current_round')}"
            )

            metrics = job.get("metrics") or {}
            metrics["_ui_call_duration_sec"] = ui_duration_sec
            metrics["_ui_response_size_kb"] = _json_size_kb(job)
            metrics["_ui_metrics_size_kb"] = _json_size_kb(metrics)

            # Execution / Telemetry
            if "Telemetry" in sections:
                st.subheader("Telemetry")

                exec_row = {
                    "scope": metrics.get("scope"),
                    "participants_count": metrics.get("participants_count"),
                    "last_round": metrics.get("last_round"),
                    "last_round_row_count": metrics.get("last_round_row_count"),
                    "job_total_duration_sec": metrics.get("job_total_duration_sec"),
                    "avg_round_duration_sec": metrics.get("avg_round_duration_sec"),
                    "avg_round_payload_kb": metrics.get("avg_round_payload_kb"),
                    "blockchain_events": metrics.get("blockchain_events"),
                    "_ui_call_duration_sec": metrics.get("_ui_call_duration_sec"),
                    "_ui_response_size_kb": metrics.get("_ui_response_size_kb"),
                    "_ui_metrics_size_kb": metrics.get("_ui_metrics_size_kb"),
                }
                st.dataframe(pd.DataFrame([exec_row]), use_container_width=True)

                rd = metrics.get("round_durations_sec") or []
                rp = metrics.get("round_payload_kb") or []
                if isinstance(rd, list) and (rd or rp):
                    n = max(len(rd), len(rp))
                    rounds_df = pd.DataFrame({
                        "round": list(range(1, n + 1)),
                        "round_duration_sec": (rd + [None] * (n - len(rd)))[:n],
                        "round_payload_kb": (rp + [None] * (n - len(rp)))[:n],
                    })
                    st.caption("Per-round metrics")
                    st.dataframe(rounds_df, use_container_width=True)

            """
            if "Raw JSON (debug)" in sections:
                with st.expander("Raw job JSON"):
                    st.json(job)
                with st.expander("Raw metrics JSON"):
                    st.json(metrics)
             """

            # Log run
            log_run(
                "fl_jobs.start",
                {
                    "job_id": job_id.strip(),
                    "status": job.get("status"),
                    "current_round": job.get("current_round"),
                    "dataset_id": job.get("dataset_id"),
                    "ui_call_duration_sec": ui_duration_sec,
                    "ui_response_size_kb": metrics["_ui_response_size_kb"],
                    "metrics": metrics,
                },
            )

            # Downloads
            st.download_button(
                "Download job result (JSON)",
                data=json.dumps(job, ensure_ascii=False, indent=2),
                file_name=f"job_{job_id.strip()}.json",
                mime="application/json",
            )
            st.download_button(
                "Download metrics (JSON)",
                data=json.dumps(metrics, ensure_ascii=False, indent=2),
                file_name=f"metrics_{job_id.strip()}.json",
                mime="application/json",
            )

            if job.get("last_error"):
                st.error(job.get("last_error"))

            # Performance summary
            if "Performance Summary" in sections:
                st.subheader("Performance Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("UI call duration (sec)", metrics.get("_ui_call_duration_sec"))
                col2.metric("Response size (KB)", metrics.get("_ui_response_size_kb"))
                col3.metric("Metrics size (KB)", metrics.get("_ui_metrics_size_kb"))

            # Privacy Governance
            privacy = metrics.get("privacy") or {}
            if "Privacy Governance" in sections and privacy:
                st.subheader("Privacy Governance")
                st.json(privacy)

            # Feature metrics
            feature_metrics = metrics.get("feature_metrics") or metrics.get("feature_stats") or {}
            if "Feature Metrics" in sections and feature_metrics:
                st.subheader("Feature Metrics")
                try:
                    fm_df = (
                        pd.DataFrame.from_dict(feature_metrics, orient="index")
                        .reset_index()
                        .rename(columns={"index": "feature"})
                    )
                    st.dataframe(fm_df, use_container_width=True)

                    st.download_button(
                        "Download feature metrics (CSV)",
                        data=fm_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"feature_metrics_{job_id.strip()}.csv",
                        mime="text/csv",
                    )

                except Exception:
                    st.json(feature_metrics)

            # Normalized
            norm_imp = metrics.get("normalized_feature")
            if "Normalized Feature" in sections and norm_imp:
                st.subheader("Normalized Feature")
                st.json(norm_imp)

            # Round trends
            round_trends = metrics.get("round_trends")
            if "Round Trends" in sections and round_trends:
                st.subheader("Round Trends")
                if isinstance(round_trends, dict) and round_trends:
                    rt_df = pd.DataFrame(round_trends)
                    st.dataframe(rt_df, use_container_width=True)
                    st.download_button(
                        "Download round trends (CSV)",
                        data=rt_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"round_trends_{job_id.strip()}.csv",
                        mime="text/csv",
                    )
            # Correlation matrix
            corr = metrics.get("correlation_matrix")

            if "Correlation Matrix" in sections:
                st.subheader("Federated Correlation Matrix (Pearson)")

                try:
                    if not corr or not isinstance(corr, dict):
                        st.info("Correlation matrix is not available.")
                    else:
                        corr_df = pd.DataFrame(corr)

                        if corr_df.empty or corr_df.shape[0] == 0 or corr_df.shape[1] == 0:
                            st.info("Correlation matrix is not available (possibly due to privacy suppression).")
                        else:
                            corr_df = corr_df.astype(float)
                            corr_df = corr_df.reindex(index=corr_df.columns, columns=corr_df.columns)

                            n = len(corr_df.columns)

                            fig_w = min(5, max(4, 0.6 * n))
                            fig_h = min(5, max(4, 0.6 * n))

                            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

                            norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

                            im = ax.imshow(
                                corr_df.values,
                                cmap="RdBu_r",
                                norm=norm,
                                aspect="equal",
                                interpolation="nearest",
                            )

                            cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
                            cbar.ax.tick_params(labelsize=6)
                            cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

                            ax.set_xticks(np.arange(n))
                            ax.set_yticks(np.arange(n))

                            ax.set_xticklabels(corr_df.columns, rotation=35, ha="right", fontsize=7)
                            ax.set_yticklabels(corr_df.index, fontsize=6)

                            for i in range(n):
                                for j in range(n):
                                    vv = float(corr_df.values[i, j])
                                    txt_color = "white" if abs(vv) >= 0.06 else "black"
                                    ax.text(j, i, f"{vv:.2f}", ha="center", va="center", fontsize=6, color=txt_color)


                            ax.set_title("Pearson correlation heatmap", fontsize=8, loc="left", pad=6)

                            fig.tight_layout()
                            st.pyplot(fig, use_container_width=False)

                except Exception as e:
                    st.error(str(e))

        except Exception as e:
            st.error(str(e))


    # Admin: Compare FL Jobs
    st.divider()
    st.subheader("Federated Learning vs Centralized model \n (only admin - for test cases hospital)")

    if role_norm() in ("Admin", "Hospital"):

        try:
            jobs = api_get("/fl/jobs", params={"limit": 200})
        except Exception as e:
            st.error(str(e))
            jobs = []

        if jobs:
            def job_label(j: dict) -> str:
                return (
                    f"{j.get('created_at', '')} | {j.get('status', '')} | "
                    f"{j.get('job_id')} | scope={j.get('scope')} | rounds={j.get('rounds')} | "
                    f"by={j.get('created_by')} ({j.get('created_by_org')})"
                )

            job_map = {job_label(j): j for j in jobs}

            a_label = st.selectbox("Job (FL)", list(job_map.keys()), key="cmp_job_a")
            ja = job_map.get(a_label) or {}
            ja_full = api_get(f"/fl/jobs/{ja.get('job_id')}")

            # Call backend baseline
            base = api_post(f"/fl/jobs/{ja.get('job_id')}/baseline", payload={})

            if not base or not isinstance(base, dict):
                st.error("Baseline endpoint returned empty response.")
                st.stop()

            if base.get("ok") is False:
                st.error(base.get("error") or "Baseline failed")
                st.stop()

            baseline = (base or {}).get("baseline") or {}
            if not isinstance(baseline, dict) or not baseline:
                st.error("Baseline response missing 'baseline' payload.")
                st.stop()

            mb = {
                "feature_metrics": baseline.get("feature_metrics") or {},
                "normalized_feature": baseline.get("normalized_feature") or {},
                "correlation_matrix": baseline.get("correlation_matrix") or {},
                "privacy": {"suppressed": False, "min_row_threshold": None},
            }

            ma = (ja_full.get("metrics") or {})

            st.caption("Baseline computed centrally (for benchmarking).")
            st.json({
                "baseline_total_rows": baseline.get("total_rows"),
                "baseline_datasets": base.get("dataset_ids"),
            })

            st.divider()
            st.subheader("Comparison Results")
            st.caption("Mode: FL vs centralized baseline (no FL/BC)")

            st.caption("FL telemetry (A)")
            st.dataframe(pd.DataFrame([{
                "participants_count": ma.get("participants_count"),
                "job_total_duration_sec": ma.get("job_total_duration_sec"),
                "avg_round_duration_sec": ma.get("avg_round_duration_sec"),
                "avg_round_payload_kb": ma.get("avg_round_payload_kb"),
                "last_round_row_count": ma.get("last_round_row_count"),
            }]), use_container_width=True)
            st.info("Telemetry metrics are not available for centralized baseline.")

            # Feature metrics comparison
            fm_cmp = _compare_feature_metrics(ma, mb)
            if fm_cmp.empty:
                st.info("No feature_metrics available to compare.")
            else:
                st.caption("Compare Feature Metrics")
                st.dataframe(fm_cmp, use_container_width=True)
                st.download_button(
                    "Download feature comparison (CSV)",
                    data=fm_cmp.to_csv(index=False).encode("utf-8"),
                    file_name="feature_metrics_compare_A_vs_B.csv",
                    mime="text/csv",
                )

                mean_rel = fm_cmp[(fm_cmp["metric"] == "mean") & (fm_cmp["rel_error"].notna())]["rel_error"]
                miss_abs = fm_cmp[(fm_cmp["metric"] == "missing_rate") & (fm_cmp["abs_diff"].notna())]["abs_diff"]

                agreement = {
                    "mean_rel_error_avg": None if mean_rel.empty else float(mean_rel.mean()),
                    "mean_rel_error_max": None if mean_rel.empty else float(mean_rel.max()),
                    "missing_rate_abs_diff_avg": None if miss_abs.empty else float(miss_abs.mean()),
                    "features_compared": int(fm_cmp["feature"].nunique()) if "feature" in fm_cmp.columns else 0,
                }
                st.caption("Overall Agreement Score (derived)")
                st.dataframe(pd.DataFrame([agreement]), use_container_width=True)

            # Correlation compare
            corr_summary = _compare_corr(ma, mb)
            st.caption("Correlation Matrix Agreement")
            st.json(corr_summary)

            # Normalized importance compare
            ni_a = ma.get("normalized_feature") or {}
            ni_b = mb.get("normalized_feature") or {}
            k = st.slider("Top-K for importance overlap", min_value=5, max_value=30, value=10, step=1, key="cmp_topk")
            ni_summary = _topk_overlap(ni_a, ni_b, k=k)
            st.caption("Normalized Feature (Top-K overlap)")
            st.json(ni_summary)

            # Privacy/Governance compare
            st.caption("Privacy Governance (A vs B)")
            st.dataframe(pd.DataFrame([{
                "A_suppressed": (ma.get("privacy") or {}).get("suppressed"),
                "B_suppressed": (mb.get("privacy") or {}).get("suppressed"),
                "A_min_row_threshold": (ma.get("privacy") or {}).get("min_row_threshold"),
                "B_min_row_threshold": (mb.get("privacy") or {}).get("min_row_threshold"),
            }]), use_container_width=True)

            st.caption("Side-by-side summary")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("A participants", ma.get("participants_count"))
            s2.metric("B participants", mb.get("participants_count"))
            s3.metric("A total sec", ma.get("job_total_duration_sec"))
            s4.metric("B total sec", mb.get("job_total_duration_sec"))

        else:
            st.info("No FL jobs found to compare yet.")

    else:
        st.caption("Comparison is available to Admin only.")


def page_runs_history() -> None:
    st.header("Runs / History")
    require_login()
    flash_show()

    st.caption("Keeps a record of actions run by the current user. Includes a download option per entry.")
    try:
        role = role_norm()

        params = {}
        if role != "Admin":
            params["mine"] = 1

        runs = api_get(path="/runs", params=params)

    except Exception as e:
        st.error(str(e))
        st.info("If this page errors, verify your backend includes GET /runs endpoint.")
        return

    if not runs:
        st.info("No runs saved yet.")
        return

    for idx, r in enumerate(runs):
        rid = r.get("run_id")
        created_at = r.get("created_at")
        run_type = r.get("run_type")
        payload = r.get("payload", {})

        with st.expander(f"{created_at} | {run_type} | {rid}", expanded=(idx == 0)):
            st.json(payload)

            st.download_button(
                "Download JSON",
                data=json.dumps(
                    {"run_id": rid, "created_at": created_at, "run_type": run_type, "payload": payload},
                    ensure_ascii=False,
                    indent=2,
                ),
                file_name=f"run_{run_type}_{rid}.json",
                mime="application/json",
            )

def page_smart_contract() -> None:
    st.header("Smart Contract")
    require_login()
    flash_show()

    # receipts
    try:
        receipts = api_get("/blockchain/receipts", params={"limit": 500})
    except Exception as e:
        st.error(str(e))
        st.info("Αν δεν υπάρχει endpoint, έλεγξε ότι το backend έχει GET /blockchain/receipts.")
        return

    receipts = receipts or []

    visible = [r for r in receipts if _receipt_visible_to_user(r)]

    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    if not visible:
        st.info("No blockchain receipts visible for your role/user yet.")
        return

    rows = []
    for r in visible:
        payload = r.get("payload") or {}
        manifest = payload.get("manifest") or {}
        actor = manifest.get("actor") or {}

        rows.append(
            {
                "created_at": r.get("created_at"),
                "event_type": r.get("event_type"),
                "ref_id": r.get("ref_id"),
                "tx_hash": r.get("tx_hash"),
                "chain_id": _first_not_none(r.get("chain_id"), payload.get("chain_id"), "offchain"),

                # chain fields
                "block_number": _first_not_none(
                    payload.get("block_number"),
                    payload.get("blockNumber"),
                    0 if payload.get("mode") == "noop" else None,
                ),
                "block_timestamp": _first_not_none(
                    payload.get("block_timestamp"),
                    payload.get("blockTimestamp"),
                    int(time.time()),
                ),
                "status": _first_not_none(
                    payload.get("status"),
                    1 if payload.get("mode") == "noop" else None,
                ),

                # actor + mode
                "actor_username": actor.get("username"),
                "actor_org": actor.get("org"),
                "mode": payload.get("mode"),

                # evaluation metrics
                "gas_used": _first_not_none(payload.get("gas_used"), payload.get("gasUsed")),
                "latency_ms": _first_not_none(payload.get("latency_ms"), payload.get("tx_latency_ms")),
                "effective_gas_price": _first_not_none(payload.get("effective_gas_price"),
                                                       payload.get("effectiveGasPrice")),
                "tx_cost_wei": _first_not_none(
                    payload.get("tx_cost_wei"),
                    payload.get("txCostWei"),
                    (
                            int(_first_not_none(payload.get("gas_used"), payload.get("gasUsed")))
                            * int(_first_not_none(payload.get("effective_gas_price"), payload.get("effectiveGasPrice")))
                    )
                    if _first_not_none(payload.get("gas_used"), payload.get("gasUsed")) is not None
                       and _first_not_none(payload.get("effective_gas_price"),
                                           payload.get("effectiveGasPrice")) is not None
                    else None,
                ),

                # offchain cost metrics
                "offchain_compute_ms": payload.get("offchain_compute_ms"),
                "payload_bytes": payload.get("payload_bytes"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No receipts data.")
        return

    # Filters
    st.subheader("Filters")
    ev_types = sorted([x for x in df["event_type"].dropna().unique().tolist()])
    pick = st.multiselect("event_type", options=ev_types, default=ev_types)

    fdf = df[df["event_type"].isin(pick)].copy()

    with st.expander("FY - INFORMATION", expanded=False):
        st.markdown("""
    - **created_at**  
      Χρόνος καταγραφής του γεγονότος (UTC).  
      ➜ Αποτελεί το σημείο εκκίνησης του audit trail από την πλευρά της εφαρμογής και επιτρέπει τη συσχέτιση ενεργειών του χρήστη με γεγονότα blockchain.

    - **event_type**  
      Τύπος ενέργειας (π.χ. έλεγχος συγκατάθεσης, δημιουργία federated job).  
      ➜ Περιγράφει *ποια λειτουργία* αποτυπώνεται.

    - **ref_id**  
      Αναγνωριστικό της οντότητας που αφορά το γεγονός (π.χ. job_id, request_id, ή dataset_id:patient_key).  
      ➜ Υλοποιεί τη σύνδεση μεταξύ blockchain εγγραφής και αντικειμένων του συστήματος χωρίς αποκάλυψη προσωπικών δεδομένων.

    - **tx_hash**  
      Μοναδικό αποτύπωμα της συναλλαγής στην αλυσίδα.  
      ➜ Χρησιμοποιείται για επαληθευσιμότητα, μη αμφισβητήσιμη απόδειξη και εξωτερικό έλεγχο.

    - **chain_id**  
      Αναγνωριστικό δικτύου blockchain.  
      ➜ Διασφαλίζει ότι η καταγραφή αναφέρεται στο σωστό περιβάλλον εκτέλεσης.

    - **block_number**  
      Αριθμός block όπου συμπεριλήφθηκε η συναλλαγή.  
      ➜ Παρέχει χρονολογική διάταξη και αμεταβλητότητα στο ιστορικό.

    - **block_timestamp**  
      Ακριβής χρόνος του blockchain σε δευτερόλεπτα.  

    - **status**  
      Αποτέλεσμα εκτέλεσης smart contract (**1 = επιτυχία, 0 = αποτυχία**).  
      ➜ Κρίσιμο για την αξιοπιστία του μηχανισμού συναίνεσης και την εγκυρότητα του ίχνους.

    - **mode**  
      `contract` = πραγματική on-chain καταγραφή  
      `noop` = off-chain  

    - **gas_used**  
      Υπολογιστικές μονάδες που απαιτήθηκαν για την ενέργεια.  
      ➜ Αντιπροσωπεύει το *υπολογιστικό κόστος* της πράξης στο smart contract.

    - **effective_gas_price**  
      Τιμή ανά μονάδα gas (wei/gas).  
      ➜ Εκφράζει το οικονομικό βάρος της εκτέλεσης στο συγκεκριμένο block.

    - **tx_cost_wei**  
      Συνολικό κόστος συναλλαγής = gas_used × effective_gas_price.  
      ➜ Μετρική αποδοτικότητας και συγκρισιμότητας μεταξύ διαφορετικών τύπων ενεργειών.

    - **latency_ms**  
      Χρόνος από την υποβολή έως την επιβεβαίωση (submit → receipt).  
      ➜ Δείκτης απόδοσης του δικτύου και της εμπειρίας χρήστη στο BC-FL.
    """)

    st.subheader("Receipts (filtered)")
    st.dataframe(fdf, use_container_width=True)

    st.download_button(
        "Download receipts (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="blockchain_receipts_filtered.csv",
        mime="text/csv",
    )

    for col in ["block_timestamp", "gas_used", "latency_ms", "effective_gas_price", "tx_cost_wei", "offchain_compute_ms", "payload_bytes", "block_number", "status"]:
        if col in fdf.columns:
            fdf[col] = pd.to_numeric(fdf[col], errors="coerce")

    """
    st.subheader("Stats")

    stats_row = {
        "receipts_count": int(len(fdf)),
        "unique_event_types": int(fdf["event_type"].nunique()) if "event_type" in fdf.columns else 0,
        "modes": str(fdf["mode"].dropna().unique().tolist()[:3]) if "mode" in fdf.columns else "N/A",
    }
    
    
    stats_row["blockchain_time_span_sec"] = int(ts.max() - ts.min()) if ts.size >= 2 else None
    st.dataframe(pd.DataFrame([stats_row]), use_container_width=True)
    
    """

    #  Execution Overview
    st.subheader("Execution Overview")
    col1, col2, col3 = st.columns(3)

    onchain_count = int((fdf["mode"] == "contract").sum()) if "mode" in fdf.columns else 0
    col1.metric("Total On-chain Events", onchain_count)
    col2.metric("Distinct Event Types", int(fdf["event_type"].nunique()) if "event_type" in fdf.columns else 0)

    ts = fdf["block_timestamp"].dropna() if "block_timestamp" in fdf.columns else pd.Series([], dtype=float)
    if ts.size >= 2:
        col3.metric("Blockchain Time Span (sec)", int(ts.max() - ts.min()))
    else:
        col3.metric("Blockchain Time Span (sec)", "—")

    #  Smart Contract Evaluation
    st.subheader("Smart Contract Evaluation")

    gas = fdf["gas_used"].dropna() if "gas_used" in fdf.columns else pd.Series([], dtype=float)
    lat = fdf["latency_ms"].dropna() if "latency_ms" in fdf.columns else pd.Series([], dtype=float)

    overall_avg_gas = gas.mean() if gas.size else np.nan
    overall_p95_latency = lat.quantile(0.95) if lat.size else np.nan

    summary_row = {
        "avg_gas_used": None if pd.isna(overall_avg_gas) else int(round(overall_avg_gas)),
        "p95_latency_ms": None if pd.isna(overall_p95_latency) else int(round(overall_p95_latency)),
        "n_with_gas": int(gas.size),
        "n_with_latency": int(lat.size),
    }
    st.dataframe(pd.DataFrame([summary_row]), use_container_width=True)

    # Per event_type table
    if "event_type" in fdf.columns:
        per_type = (
            fdf.groupby("event_type", dropna=False)
            .agg(
                n=("event_type", "size"),
                avg_gas_used=("gas_used", "mean"),
                p95_latency_ms=("latency_ms", lambda s: s.dropna().quantile(0.95) if s.dropna().size else np.nan),
            )
            .reset_index()
        )
        per_type["avg_gas_used"] = pd.to_numeric(per_type["avg_gas_used"], errors="coerce").round().astype("Int64")
        per_type["p95_latency_ms"] = pd.to_numeric(per_type["p95_latency_ms"], errors="coerce").round().astype("Int64")

        st.caption("Per event_type")
        st.dataframe(per_type, use_container_width=True)

    """
    Plot
    st.write("Event types in filtered df:", fdf["event_type"].value_counts())
    st.subheader("Gas Used Distribution by event type")

    plot_df = fdf.dropna(subset=["event_type", "gas_used"]).copy()
    if plot_df.empty:
        st.info("No gas_used values available yet.")
        return

    types = plot_df["event_type"].astype(str).unique().tolist()
    data = [plot_df.loc[plot_df["event_type"].astype(str) == t, "gas_used"].values for t in types]

    plot_df = fdf.dropna(subset=["event_type", "gas_used"]).copy()
    if plot_df.empty:
        st.info("No gas_used values available yet.")
        return

    wanted_order = ["RUN_RECORDED", "PATIENT_CONSENT_TX", "PATIENT_CONSENT_UPDATED"]
    types_present = [t for t in wanted_order if t in plot_df["event_type"].astype(str).unique().tolist()]
    others = [t for t in plot_df["event_type"].astype(str).unique().tolist() if t not in types_present]
    types = types_present + others

    data = [plot_df.loc[plot_df["event_type"].astype(str) == t, "gas_used"].values for t in types]

    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=150)
    ax.boxplot(data, labels=types, showfliers=False)
    ax.set_title("Gas per event_type", loc="left", fontsize=9)
    ax.set_ylabel("gas_used (units)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    """
    """
    plot 2
    st.subheader("Gas / Latency")

    time_df = fdf.copy()

    time_df["ts_utc"] = pd.to_datetime(time_df.get("block_timestamp"), unit="s", utc=True, errors="coerce")
    mask = time_df["ts_utc"].isna()
    time_df.loc[mask, "ts_utc"] = pd.to_datetime(time_df.loc[mask, "created_at"], utc=True, errors="coerce")

    time_df = time_df.dropna(subset=["ts_utc"])
    time_df["ts_local"] = time_df["ts_utc"].dt.tz_convert("Europe/Athens")

    metric_choice = st.selectbox("Metric", ["gas_used", "latency_ms"], index=0, key="sc_time_metric")

    plot_t = time_df.dropna(subset=[metric_choice, "event_type"]).copy()
    if plot_t.empty:
        st.info("No time-series values available yet for this metric.")
    else:
        fig, ax = plt.subplots(figsize=(3.6, 2.4), dpi=150)

        series_types = st.multiselect(
            "Event types to plot",
            options=sorted(plot_t["event_type"].astype(str).unique().tolist()),
            default=sorted(plot_t["event_type"].astype(str).unique().tolist()),
            key="sc_plot_types",
        )

        for et in series_types:
            sub = plot_t[plot_t["event_type"].astype(str) == et]
            if sub.empty:
                continue
            ax.scatter(sub["ts_local"], sub[metric_choice], s=10, label=et)

        ax.set_title(f"{metric_choice} over time", loc="left", fontsize=9)
        ax.set_ylabel(metric_choice, fontsize=8)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.tick_params(axis="y", labelsize=8)

        ax.legend(fontsize=6)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
     """

def page_settings() -> None:
    st.header("Settings")
    require_login()
    flash_show()

    st.subheader("Backend URL")
    st.code(DEFAULT_BASE_URL)
    st.caption("To change it, set API_BASE_URL env var for the UI container.")

    st.subheader("Session")
    st.json(
        {
            "username": st.session_state.get("username"),
            "role": st.session_state.get("role"),
            "org": st.session_state.get("org"),
            "token_present": bool(st.session_state.get("token")),
        }
    )


def main() -> None:

    st.set_page_config(page_title="BC-FL Platform", layout="wide")

    ui_topbar()

    if not st.session_state.get("token"):
        ui_login_register()
        return

    pages = PAGES_BY_ROLE.get(role_norm(), ["Dashboard", "Settings"])

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", pages, index=0)

    # Routing: ανάλογα με την επιλογή, καλούμε την αντίστοιχη συνάρτηση σελίδας
    if page == "Dashboard":
        page_dashboard()
    elif page == "Nodes":
        page_nodes()
    elif page == "Datasets":
        page_datasets()
    elif page == "Consents":
        page_consents()
    elif page == "Access Requests":
        page_access_requests()
    elif page == "Federated Jobs":
        page_federated_jobs()
    elif page == "Runs / History":
        page_runs_history()
    elif page == "Smart Contract":
        page_smart_contract()
    elif page == "Settings":
        page_settings()
    else:
        st.write("Unknown page.")


if __name__ == "__main__":
    main()


