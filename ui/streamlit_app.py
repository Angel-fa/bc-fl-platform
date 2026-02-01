from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List

import pandas as pd
import time
import matplotlib.pyplot as plt
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


def _features_available_to_requester(ds: Dict[str, Any]) -> List[str]:
    """
    Ορίζει ποια features μπορεί να επιλέξει ο χρήστης όταν δημιουργεί Federated Job.
    - Για Biobank/Researcher: ΜΟΝΟ exposed_features
    - Για Hospital/Admin: πριν το στάδιο exposed_features
    """
    exposed = _dataset_exposed_features(ds)
    cols = _dataset_columns(ds)
    if role_norm() in ("Researcher", "Biobank"):
        return exposed
    return exposed or cols


def suggest_actions(metrics: dict) -> list[str]:
    actions: list[str] = []

    # 1) Data quality
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

    # 2) Privacy / governance
    privacy = metrics.get("privacy") or {}
    if isinstance(privacy, dict) and privacy:
        thr = privacy.get("min_row_threshold")
        suppressed = privacy.get("suppressed_features") or []
        if suppressed:
            actions.append(
                f"[Privacy] Suppressed features λόγω μικρού N: {', '.join(map(str, suppressed))}. "
                f"Πρότεινε περισσότερα nodes ή αλλαγή threshold."
            )
        if thr is not None:
            actions.append(f"[Privacy] Ελάχιστο threshold={thr}. Πρότεινε rule: μην εμφανίζεις stats για N<threshold.")

    # 3) Rounds μελλοντικά με ml
    trends = metrics.get("round_trends") or {}
    if isinstance(trends, dict) and trends:
        actions.append("[Trends] Υπάρχουν round_trends: δείξε convergence plot ή μείωσε rounds αν συγκλίνει γρήγορα.")
    """
    # 4) Normalized importance: top 3 features
    norm_imp = metrics.get("normalized_importance")
    if isinstance(norm_imp, dict) and norm_imp:
        try:
            top = sorted(norm_imp.items(), key=lambda x: float(x[1]), reverse=True)[:3]
            actions.append(
                "[Importance] Top-3 σημαντικά features: "
                + ", ".join([f"{k} ({float(v):.3f})" for k, v in top])
                + ". Πρότεινε focus σε αυτά σε επόμενο job."
            )
        except Exception:
            pass
    """
    """
    # Αν δεν βγει τίποτα (π.χ. missing metrics), βάζουμε generic μήνυμα.
    if not actions:
        actions.append(
            "Δεν υπάρχουν αρκετά enriched metrics ακόμη για recommendations. "
            "Τρέξε job με feature_metrics / privacy / trends ενεργά."
        )
    """
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

        # Για Hospital/Biobank απαιτείται invite_code
        invite_code = ""
        if r_role in ("Hospital", "Biobank"):
            invite_code = st.text_input(
                "Invite code",
                key="reg_invite_code",
                help="Required for Hospital/Biobank registration.",
            )

        if st.button("Register"):
            try:
                payload = {"username": r_username, "password": r_password, "role": r_role, "org": r_org}
                if r_role in ("Hospital", "Biobank"):
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
                "portal_secret": portal_secret,
            }
            resp = api_post("/public/consent", payload)
            st.success(f"Consent recorded on-chain. tx={resp.get('tx_hash')}")
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
    st.dataframe(nodes, width="stretch")


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

            # Φόρμα metadata του descriptor (όχι το πραγματκό file)
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

                # 1) Upload file στο agent
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

            # 2) Local URI read -> upload
            local_uri = st.session_state.get("uploaded_local_uri", "")
            st.text_input("Local URI (inside agent container)", value=local_uri, disabled=True, key="ds_local_uri")

            # 3) Create descriptor στο backend
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

    st.dataframe(datasets, width="stretch")


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
    st.dataframe(cps or [], width="stretch")


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

    st.dataframe(items, width="stretch")


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

    require_login()      # αν δεν υπάρχει token, σταματάει τη σελίδα και ζητά login

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

    ds_map = {      # Δημιουργούμε mapping από "φιλικό label" -> dataset_id
        f"{d['name']} | owner={d['owner_org']} | {d['dataset_id']}": d["dataset_id"]
        for d in datasets
    }

    ds_sel = st.selectbox("Dataset", list(ds_map.keys())) # Dropdown επιλογής dataset

    dataset_id = ds_map[ds_sel]      # Πραγματική τιμή dataset_id που θα στείλουμε στο backend


    ds_current = next(
        (d for d in datasets if str(d.get("dataset_id")) == str(dataset_id)),
        None,
    )

    rounds = st.number_input("Rounds", min_value=1, max_value=50, value=3, step=1)

    if role_norm() not in ("Researcher", "Biobank", "Hospital"):
        st.error("Only Hospital, Researcher and Biobank can create FL jobs.")
        st.stop()

    available_features = _features_available_to_requester(ds_current or {})

    selected_features = st.multiselect(
        "Features (select one or more)",
        options=available_features,
        default=available_features[:2] if len(available_features) >= 2 else available_features,
    )

    label = st.text_input("Label (optional)", value="")

    notes = st.text_area("Notes", value="Compute federated statistics.")


    if st.button("Create FL Job", type="primary"): # Κουμπί δημιουργίας job
        try:
            if not _is_request_approved_for_user(dataset_id):
                st.error(
                    "You need an APPROVED access request for this dataset "
                    "before creating/running a federated job."
                )
                st.stop()

            if not selected_features:
                st.error("Select at least one feature.")     # Validation: πρέπει να έχει επιλεγεί τουλάχιστον 1 feature
                st.stop()

            payload = {             # Payload που θα σταλεί στο backend endpoint
                "dataset_id": dataset_id,
                "rounds": int(rounds),
                "features": selected_features,
                "label": label.strip() or None,
                "notes": notes,
            }

            job = api_post("/fl/jobs", payload)  # Δημιουργία job στο backend (επιστρέφει FLJob record)

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
            "Privacy / Governance",
            "Feature Metrics (Distribution & Data Quality)",
            "Normalized Importance",
            "Round Trends",
            "Correlation Matrix",
        ],
        default=[
            "Performance Summary",
            "Feature Metrics (Distribution & Data Quality)",
            "Normalized Importance",
            "Round Trends",
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

            # Extract + enrich metrics
            metrics = job.get("metrics") or {}
            metrics["_ui_call_duration_sec"] = ui_duration_sec
            metrics["_ui_response_size_kb"] = _json_size_kb(job)
            metrics["_ui_metrics_size_kb"] = _json_size_kb(metrics)

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

            # Outputs
            st.subheader("Federated Aggregates (Global)")
            st.json(job.get("global_model") or {})

            st.subheader("Execution Metrics")
            st.json(metrics)

            if job.get("last_error"):
                st.error(job.get("last_error"))

            # Performance summary
            if "Performance Summary" in sections:
                st.subheader("Performance Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("UI call duration (sec)", metrics.get("_ui_call_duration_sec"))
                col2.metric("Response size (KB)", metrics.get("_ui_response_size_kb"))
                col3.metric("Metrics size (KB)", metrics.get("_ui_metrics_size_kb"))

            # 1) Privacy / Governance
            privacy = metrics.get("privacy") or {}
            if "Privacy / Governance" in sections and privacy:
                st.subheader("Privacy / Governance")
                st.json(privacy)

            # 2) Feature metrics (single section: dataframe + download + plot)
            feature_metrics = metrics.get("feature_metrics") or metrics.get("feature_stats") or {}
            if "Feature Metrics (Distribution & Data Quality)" in sections and feature_metrics:
                st.subheader("Feature Metrics (Distribution & Data Quality)")
                try:
                    fm_df = (
                        pd.DataFrame.from_dict(feature_metrics, orient="index")
                        .reset_index()
                        .rename(columns={"index": "feature"})
                    )
                    st.dataframe(fm_df, width="stretch")

                    st.download_button(
                        "Download feature metrics (CSV)",
                        data=fm_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"feature_metrics_{job_id.strip()}.csv",
                        mime="text/csv",
                    )

                    if "missing_rate" in fm_df.columns:
                        tmp = fm_df[["feature", "missing_rate"]].copy()
                        tmp["missing_rate"] = pd.to_numeric(tmp["missing_rate"], errors="coerce").fillna(0.0)
                        tmp = tmp.sort_values("missing_rate", ascending=False).head(15)
                        st.bar_chart(tmp.set_index("feature")["missing_rate"])

                except Exception:
                    st.json(feature_metrics)

            # 3) Normalized importance
            norm_imp = metrics.get("normalized_importance")
            if "Normalized Importance" in sections and norm_imp:
                st.subheader("Normalized Feature Importance")
                st.json(norm_imp)

            # 4) Round trends
            round_trends = metrics.get("round_trends")
            if "Round Trends" in sections and round_trends:
                st.subheader("Round Trends (plots)")
                if isinstance(round_trends, dict):
                    for name, series in round_trends.items():
                        try:
                            s = pd.Series(series, name=name)
                            st.line_chart(s)
                        except Exception:
                            pass
                    with st.expander("Raw round_trends JSON"):
                        st.json(round_trends)
                else:
                    st.json(round_trends)

            # 5) Correlation matrix
            corr = metrics.get("correlation_matrix")
            if "Correlation Matrix" in sections and corr:
                st.subheader("Federated Correlation Matrix (Pearson)")
                corr_df = pd.DataFrame(corr).astype(float)
                corr_df = corr_df.reindex(index=corr_df.columns, columns=corr_df.columns)

                n = len(corr_df.columns)
                fig_w = min(8.0, max(4.5, 0.55 * n))
                fig_h = min(6.0, max(3.8, 0.55 * n))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

                im = ax.imshow(corr_df.values, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="equal")

                ax.set_xticks(np.arange(n))
                ax.set_yticks(np.arange(n))
                ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(corr_df.index, fontsize=8)

                vals2 = corr_df.values
                for i in range(n):
                    for j in range(n):
                        vv = float(vals2[i, j])
                        txt_color = "white" if abs(vv) >= 0.6 else "black"
                        ax.text(j, i, f"{vv:.2f}", ha="center", va="center", fontsize=7, color=txt_color)

                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

                ax.set_title("Pearson correlation heatmap", fontsize=10, loc="left", pad=10)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=False)

        except Exception as e:
            st.error(str(e))


def page_runs_history() -> None:
    st.header("Runs / History")
    require_login()
    flash_show()

    st.caption("Keeps a record of actions run by the current user. Includes a download option per entry.")
    try:
        runs = api_get("/runs", params={"mine": "1"})
    except Exception as e:
        st.error(str(e))
        st.info("If this page errors, verify your backend includes GET/POST /runs endpoints.")
        return

    if not runs:
        st.info("No runs saved yet.")
        return

    # Εμφανίζουμε runs με πιο πρόσφατο expanded (idx == 0)
    for idx, r in enumerate(runs):
        rid = r.get("run_id")
        created_at = r.get("created_at")
        run_type = r.get("run_type")
        payload = r.get("payload", {})

        with st.expander(f"{created_at} | {run_type} | {rid}", expanded=(idx == 0)):
            st.json(payload)

            # Download button: εξαγωγή του run ως JSON αρχείο
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

    # Apply role-based filtering
    visible = [r for r in receipts if _receipt_visible_to_user(r)]

    if not visible:
        st.info("No blockchain receipts visible for your role/user yet.")
        return

    # Build dataframe
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
                "chain_id": r.get("chain_id"),
                "block_number": payload.get("block_number"),
                "block_timestamp": payload.get("block_timestamp"),
                "actor_username": actor.get("username"),
                "actor_org": actor.get("org"),
                "mode": payload.get("mode"),
            }
        )

    df = pd.DataFrame(rows)

    # Filters
    st.subheader("Filters")
    ev_types = sorted([x for x in df["event_type"].dropna().unique().tolist()])
    pick = st.multiselect("event_type", options=ev_types, default=ev_types)

    fdf = df[df["event_type"].isin(pick)].copy()

    st.subheader("Receipts (filtered)")
    st.dataframe(fdf, width="stretch")

    st.download_button(
        "Download receipts (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="blockchain_receipts_filtered.csv",
        mime="text/csv",
    )

    # 5) Simple stats (gas/latency placeholders)
    st.subheader("Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Receipts count", int(len(fdf)))
    col2.metric("Unique event_types", int(fdf["event_type"].nunique()))
    col3.metric("Mode", ", ".join(sorted(set([str(x) for x in fdf["mode"].dropna().unique().tolist()]))) or "-")

    st.subheader("Execution Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total On-chain Events",
        int(len(fdf))
    )

    col2.metric(
        "Distinct Event Types",
        int(fdf["event_type"].nunique())
    )

    # Χρονική κάλυψη events (block-based)
    if "block_timestamp" in fdf.columns and fdf["block_timestamp"].notna().any():
        t_min = int(fdf["block_timestamp"].min())
        t_max = int(fdf["block_timestamp"].max())
        col3.metric(
            "Blockchain Time Span (sec)",
            int(t_max - t_min)
        )
    else:
        col3.metric("Blockchain Time Span (sec)", "—")


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