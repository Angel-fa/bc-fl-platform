# streamlit_app.py
# --------------------------------------------------------------------------------------
# Streamlit UI (Frontend) για την πλατφόρμα BC-FL.
#
# Ρόλος του αρχείου:
# - Παρέχει Web UI για login/register, διαχείριση nodes, datasets, consents, access requests,
#   federated jobs και runs/history.
# - Κάνει HTTP calls στο Backend API (FastAPI) στο DEFAULT_BASE_URL.
# - Δεν αποθηκεύει “δεδομένα ασθενών” τοπικά στο UI· το UI είναι client που καλεί endpoints.
#
# Σημαντικό:
# - Τα endpoints που καλεί το UI (π.χ. /auth/login, /datasets, /public/consent)
#   πρέπει να υπάρχουν στο backend κάτω από /api/v1/... (λόγω DEFAULT_BASE_URL).
# - Η ασφάλεια/έλεγχος πρόσβασης (RBAC) πρέπει να επιβάλλεται στο backend.
#   Το UI απλώς “κρύβει” σελίδες ανά ρόλο για ευχρηστία.
# --------------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, List

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

import requests
import streamlit as st
from PIL import Image


# ----------------------------
# Configuration
# ----------------------------
# Base URL του backend API.
# - Σε Docker περιβάλλον, το service name "backend" λύνει μέσω Docker DNS.
# - Το /api/v1 είναι το prefix του κεντρικού router του backend.
DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000/api/v1")

# Timeout για HTTP requests προς backend/agent.
# Αν backend ή agent αργεί/δεν απαντά, το UI θα κάνει timeout αντί να “κολλήσει”.
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "12"))

# Επιτρεπτοί ρόλοι χρηστών (πρέπει να ταιριάζουν με το backend auth/RBAC).
ROLES = ["Admin", "Hospital", "Biobank", "Researcher"]

# Σταθερές για dropdowns/forms.
SENSITIVITY_LEVELS = ["low", "medium", "high"]
CONSENT_STATUS = ["draft", "active", "retired"]
REQUEST_STATUSES = ["submitted", "approved", "denied"]
EXPORT_METHODS = ["federated", "aggregated", "synthetic"]

# Ποιες σελίδες εμφανίζονται σε κάθε ρόλο.
# (Frontend “navigation filter” — ο πραγματικός έλεγχος πρόσβασης είναι στο backend.)
PAGES_BY_ROLE = {
    "Admin": ["Dashboard", "Nodes", "Datasets", "Consents", "Access Requests", "Federated Jobs", "Runs / History", "Settings"],
    "Hospital": ["Dashboard", "Nodes", "Datasets", "Consents", "Access Requests", "Federated Jobs", "Runs / History", "Settings"],
    "Biobank": ["Dashboard", "Nodes", "Datasets", "Access Requests", "Federated Jobs", "Runs / History", "Settings"],
    "Researcher": ["Dashboard", "Nodes", "Datasets", "Access Requests", "Federated Jobs", "Runs / History", "Settings"],
}


def role_norm() -> str:
    """
    Επιστρέφει ρόλο σε “κανονικοποιημένη” μορφή (π.χ. 'hospital' -> 'Hospital').

    Γιατί το χρειαζόμαστε:
    - Το backend μπορεί να επιστρέψει role με διαφορετικό casing.
    - Το UI χρησιμοποιεί συγκεκριμένα strings στα PAGES_BY_ROLE και σε checks.
    """
    r = (st.session_state.get("role") or "").strip()
    mapping = {
        "hospital": "Hospital",
        "biobank": "Biobank",
        "researcher": "Researcher",
        "admin": "Admin",
    }
    return mapping.get(r.lower(), r)


def role() -> str:
    """
    Επιστρέφει το raw role όπως έχει αποθηκευτεί στο session_state.
    (Χρησιμοποιείται όταν δεν χρειάζεται κανονικοποίηση.)
    """
    return st.session_state.get("role", "")


def org() -> str:
    """
    Επιστρέφει το org του logged-in χρήστη από session_state.
    Χρησιμοποιείται σε payloads (π.χ. requester_org) και σε εμφανίσεις.
    """
    return st.session_state.get("org", "")


# ----------------------------
# Helpers
# ----------------------------
def _assets_path(filename: str) -> str:
    """
    Χτίζει absolute path για files μέσα στο /assets του UI container.

    Γιατί:
    - Τα logos (π.χ. logo_1.png) αντιγράφονται στο container μέσω ui/Dockerfile
      (COPY assets /app/assets).
    - Θέλουμε ένα σταθερό τρόπο να τα βρίσκουμε ανεξάρτητα από working directory.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "assets", filename)


def _load_image_optional(path: str) -> Optional[Image.Image]:
    """
    Προσπαθεί να φορτώσει μία εικόνα και επιστρέφει None αν αποτύχει.

    Χρήσιμο όταν:
    - Τα assets μπορεί να λείπουν σε κάποιο build, χωρίς να “σπάσει” όλο το UI.
    """
    try:
        return Image.open(path)
    except Exception:
        return None


def _auth_headers() -> Dict[str, str]:
    """
    Δημιουργεί Authorization header αν υπάρχει token στο session.
    Το backend περιμένει Bearer token (JWT) στο header.

    Αν δεν υπάρχει token:
    - επιστρέφει empty dict
    - άρα τα endpoints που απαιτούν auth θα αποτύχουν (όπως πρέπει).
    """
    token = st.session_state.get("token")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def api_get(path: str, params: Optional[dict] = None) -> Any:
    """
    Wrapper για GET requests προς το backend API.

    - Συνθέτει URL: DEFAULT_BASE_URL + path (π.χ. /datasets).
    - Περνά auth headers (Bearer token) αν υπάρχει.
    - Αν backend επιστρέψει error status, σηκώνει RuntimeError ώστε να το δείξουμε στο UI.
    """
    url = f"{DEFAULT_BASE_URL}{path}"
    r = requests.get(url, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else None


def api_post(path: str, payload: Optional[dict] = None, params: Optional[dict] = None) -> Any:
    """
    Wrapper για POST requests προς το backend API.

    - payload: στέλνεται ως JSON body
    - params: query parameters (π.χ. ?decision=approved)
    - Σηκώνει error αν status >= 400.
    """
    url = f"{DEFAULT_BASE_URL}{path}"
    r = requests.post(url, json=payload or {}, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"POST {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else None


def api_patch(path: str, params: Optional[dict] = None, payload: Optional[dict] = None) -> Any:
    """
    Wrapper για PATCH requests προς το backend API.

    Χρησιμοποιείται για updates, π.χ.:
    - /datasets/{id}/features
    - /access-requests/{id}/decision
    - /consents/{id}/status
    """
    url = f"{DEFAULT_BASE_URL}{path}"
    r = requests.patch(url, json=payload, params=params or {}, headers=_auth_headers(), timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"PATCH {path} failed: {r.status_code} {r.text}")
    return r.json() if r.text else None


def log_run(run_type: str, payload: Dict[str, Any]) -> None:
    """
    Καταγράφει “ιστορικό ενεργειών” (Runs / History) στο backend.

    Design:
    - Best-effort: Δεν θέλουμε να αποτύχει όλο το UI αν λείπει το endpoint.
    - Endpoint που απαιτείται στο backend: POST /api/v1/runs (δηλ. UI call: /runs).

    Γιατί useful:
    - Σου επιτρέπει να δείχνεις στον καθηγητή “τι έτρεξε” κάθε χρήστης.
    - Το backend αποθηκεύει run logs σε sqlite (runs table).
    """
    try:
        api_post("/runs", payload={"run_type": run_type, "payload": payload})
    except Exception:
        # Δεν μπλοκάρουμε το UI αν αποτύχει το history logging.
        pass


def require_login() -> None:
    """
    Guard για σελίδες που απαιτούν authentication.

    Αν δεν υπάρχει token:
    - δείχνει μήνυμα
    - σταματά την εκτέλεση της σελίδας (st.stop()).
    """
    if not st.session_state.get("token"):
        st.info("Please login to continue.")
        st.stop()


def _dataset_columns(ds: Dict[str, Any]) -> List[str]:
    """
    Επιστρέφει τα columns που έχει επιστρέψει το backend μετά από validate.

    Σημείωση:
    - Για Hospital, columns υπάρχουν μετά το /validate.
    - Για Researcher/Biobank, το backend μπορεί να τα “sanitize” (columns=None).
    """
    cols = ds.get("columns") or []
    return [str(x) for x in cols if str(x).strip()]


def _dataset_exposed_features(ds: Dict[str, Any]) -> List[str]:
    """
    Επιστρέφει τις exposed_features (λίστα features που ο Hospital έχει επιτρέψει).

    Αυτό αποτελεί governance/minimization mechanism:
    - external roles (Researcher/Biobank) πρέπει να βλέπουν μόνο exposed_features.
    """
    feats = ds.get("exposed_features")
    if feats is None:
        return []
    return [str(x) for x in (feats or []) if str(x).strip()]


def _features_available_to_requester(ds: Dict[str, Any]) -> List[str]:
    """
    Ορίζει ποια features μπορεί να επιλέξει ο χρήστης όταν δημιουργεί Federated Job.

    Rules:
    - Για Biobank/Researcher: ΜΟΝΟ exposed_features (privacy-by-design).
    - Για Hospital/Admin: exposed_features αν υπάρχουν, αλλιώς columns (πιο permissive).
    """
    exposed = _dataset_exposed_features(ds)
    cols = _dataset_columns(ds)
    if role_norm() in ("Researcher", "Biobank"):
        return exposed
    return exposed or cols


def suggest_actions(metrics: dict) -> list[str]:
    """
    Προαιρετική (rule-based) συνάρτηση για “recommendations” μετά από job.

    - Αναλύει metrics dictionary που γυρνάει το backend (/fl/jobs/{id}/start).
    - Παράγει λίστα με ενέργειες (actions) π.χ.:
      - data quality (missing/outliers)
      - privacy (suppression / min threshold)
      - trends/convergence
      - feature importance

    Note:
    - Το UI έχει το section “Recommended next actions” σχολιασμένο (disabled).
    - Μπορεί να ενεργοποιηθεί εύκολα, αν το θες στην παρουσίαση.
    """
    actions: list[str] = []

    # 1) Data quality: missing rates, outliers, constant features
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

    # 2) Privacy / governance: suppressed features, threshold rules
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

    # 3) Convergence / trends: ύπαρξη round_trends
    trends = metrics.get("round_trends") or {}
    if isinstance(trends, dict) and trends:
        actions.append("[Trends] Υπάρχουν round_trends: δείξε convergence plot ή μείωσε rounds αν συγκλίνει γρήγορα.")

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

    # Αν δεν βγει τίποτα (π.χ. missing metrics), βάζουμε generic μήνυμα.
    if not actions:
        actions.append(
            "Δεν υπάρχουν αρκετά enriched metrics ακόμη για recommendations. "
            "Τρέξε job με feature_metrics / privacy / trends ενεργά."
        )

    return actions


# ----------------------------
# Flash messages (persist across reruns)
# ----------------------------
def flash_set(kind: str, text: str) -> None:
    """
    Αποθηκεύει flash message στο session_state.

    Γιατί:
    - Streamlit κάνει rerun συχνά (κάθε interaction).
    - Θέλουμε να “κρατάμε” ένα μήνυμα (success/error) μέχρι να το δείξουμε.
    """
    st.session_state["flash_msg"] = {"kind": kind, "text": text}


def flash_clear() -> None:
    """Καθαρίζει το flash message."""
    st.session_state.pop("flash_msg", None)


def flash_show() -> None:
    """
    Εμφανίζει (αν υπάρχει) το flash message.

    kind:
    - success -> st.success
    - warning -> st.warning
    - error -> st.error
    - else -> st.info
    """
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


# ----------------------------
# UI: Auth
# ----------------------------
def ui_login_register() -> None:
    """
    Landing UI όταν ο χρήστης ΔΕΝ είναι logged in.

    Περιλαμβάνει:
    - Login form (POST /auth/login)
    - Register form (POST /auth/register)
    - Public Patient Consent portal (POST /public/consent)

    Σημαντικό για Public Consent:
    - Το UI καλεί api_post("/public/consent")
    - Άρα backend endpoint πρέπει να είναι: /api/v1/public/consent
      (δηλ. mounted μέσα στο api router prefix="/api/v1").
    """
    st.title("BC-FL Platform")
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

    # -------------------------
    # Login
    # -------------------------
    with c1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        # Το κουμπί Login κάνει call στο backend για token.
        if st.button("Login", type="primary"):
            try:
                # POST /auth/login (backend επιστρέφει access_token + user info)
                resp = api_post("/auth/login", {"username": username, "password": password})
                st.session_state["token"] = resp["access_token"]

                # Αποθηκεύουμε user claims στο session_state για να αλλάξει UI συμπεριφορά.
                user = resp["user"]
                st.session_state["username"] = user["username"]
                st.session_state["role"] = user["role"]
                st.session_state["org"] = user["org"]

                st.success("Logged in.")

                # Προαιρετική καταγραφή run history
                log_run("auth.login", {"username": username, "role": user.get("role"), "org": user.get("org")})

                # rerun για να αλλάξει σε “logged in” mode και να ανοίξει navigation.
                st.rerun()

            except Exception as e:
                # Σε αποτυχία login, καθαρίζουμε ευαίσθητα session keys.
                for k in ("token", "username", "role", "org"):
                    st.session_state.pop(k, None)
                st.error(str(e))

    # -------------------------
    # Register
    # -------------------------
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

        # Για Hospital/Biobank απαιτείται invite_code (enforced στο backend register endpoint)
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

                # POST /auth/register
                api_post("/auth/register", payload)

                st.success("Registered. You can now login.")
                log_run("auth.register", {"username": r_username, "role": r_role, "org": r_org})

            except Exception as e:
                st.error(str(e))

    # Οπτικός διαχωρισμός πριν το public patient portal
    st.divider()

    # -------------------------
    # Public Patient Consent
    # -------------------------
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

    # Όταν πατηθεί Submit:
    # - καλεί POST /public/consent στο backend (μέσα στο /api/v1)
    # - backend κάνει validation secret και καταγράφει consent (σύμφωνα με patient_consent_routes.py)
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


def ui_topbar() -> None:
    """
    Top bar που εμφανίζεται ΠΑΝΤΑ (στην αρχή του main).

    Περιέχει:
    - ένδειξη backend URL
    - στοιχεία χρήστη (username/role/org) όταν είναι logged in
    - κουμπί logout που καθαρίζει το session_state
    """
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        st.caption(f"Backend: {DEFAULT_BASE_URL}")
    with c2:
        if st.session_state.get("token"):
            st.write(
                f"User: **{st.session_state.get('username')}** | "
                f"Role: **{role_norm()}** | Org: **{org()}**"
            )
    with c3:
        if st.session_state.get("token"):
            if st.button("Logout"):
                # Καθαρίζει όλα τα keys (token, role, org, etc.)
                st.session_state.clear()
                st.rerun()


# ----------------------------
# Pages
# ----------------------------
def page_dashboard() -> None:
    """
    Dashboard page:
    - εμφανίζει logos
    - επιβεβαιώνει ότι backend είναι “alive” (GET /health)
    - δείχνει συνοπτική περιγραφή του PoC flow
    """
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

    # Health check προς backend API (/api/v1/health)
    try:
        health = api_get("/health")
        st.success(f"Backend status: {health.get('status', 'unknown')}")
    except Exception as e:
        st.error(str(e))

    # Περιγραφή του concept: federated-only descriptors (no raw data transfer)
    st.write(
        """
This platform supports federated-only descriptors:
- Hospitals register dataset descriptors (pointers) to local files hosted by their agents.
- Biobanks and Researchers request access.
- Hospitals approve/deny.
- Approved parties can run federated jobs (no raw data transfer).
        """
    )


def page_nodes() -> None:
    """
    Nodes page:
    - εμφανίζει registered Hospital Agents (GET /nodes)
    - χρήσιμο για να δεις ότι ο agent έκανε registration επιτυχώς.
    """
    st.header("Nodes (Hospital Agents)")
    require_login()
    flash_show()

    st.caption("Nodes are registered once and persisted in the platform database. You do NOT need to re-register unless you reset the DB.")
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

    # Hint για Docker DNS / base_url σωστό format
    if role_norm() == "Hospital":
        st.caption("If you changed Docker compose service names, ensure node.base_url matches the internal Docker DNS name (e.g., http://bc-fl-hospital-a-agent:9001).")


def page_datasets() -> None:
    """
    Datasets page:
    - Για Hospital:
      1) Upload dataset file στο agent (POST {agent}/upload)
      2) Create dataset descriptor στο backend (POST /datasets)
      3) Validate descriptor (POST /datasets/{id}/validate) για να πάρεις columns/row_count
      4) Expose features (PATCH /datasets/{id}/features) για external visibility
    - Για άλλους ρόλους:
      - βλέπουν list descriptors (GET /datasets) με τα αντίστοιχα restrictions.
    """
    st.header("Datasets (Descriptors)")
    require_login()
    flash_show()

    # -------------------------
    # Hospital-only: Create dataset descriptor
    # -------------------------
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
            # Dropdown επιλογής node
            node_opts = {f"{n['name']} ({n['org']})": n["node_id"] for n in nodes}
            selected_node_label = st.selectbox("Hosting node", list(node_opts.keys()), key="ds_host_node_sel")
            selected_node_id = node_opts[selected_node_label]

            # Φόρμα metadata του descriptor (όχι το actual file)
            name = st.text_input("Name", value="Admissions Dataset", key="ds_name")
            description = st.text_area(
                "Description",
                value="Federated descriptor pointing to local hospital dataset.",
                key="ds_desc",
            )
            sensitivity = st.selectbox("Sensitivity", SENSITIVITY_LEVELS, index=0, key="ds_sens")
            schema_id = st.text_input("Schema ID", value="admissions_v1", key="ds_schema")

            # File uploader: το αρχείο θα σταλεί στο agent (όχι στο backend)
            uploaded = st.file_uploader("Attach dataset file (CSV or Excel)", type=["csv", "xlsx"], key="ds_uploader")

            if uploaded is not None:
                st.info(f"Selected file: {uploaded.name}")

                # 1) Upload file στο agent
                if st.button("Upload to hosting node", type="secondary", key="ds_upload_btn"):
                    try:
                        # Βρίσκουμε το selected node info
                        node = next(n for n in nodes if str(n["node_id"]) == str(selected_node_id))
                        agent_base = node["base_url"].rstrip("/")

                        # multipart/form-data upload προς agent /upload
                        files = {"file": (uploaded.name, uploaded.getvalue())}
                        r = requests.post(
                            f"{agent_base}/upload",
                            files=files,
                            headers={"X-Agent-Secret": os.getenv("AGENT_REG_SECRET", "dev-secret")},
                            timeout=REQUEST_TIMEOUT,
                        )
                        if r.status_code >= 400:
                            raise RuntimeError(f"Upload failed: {r.status_code} {r.text}")

                        # Ο agent επιστρέφει local_uri (path μέσα στο agent container)
                        st.session_state["uploaded_local_uri"] = r.json().get("local_uri", "")
                        st.success(f"Uploaded to: {st.session_state['uploaded_local_uri']}")
                    except Exception as e:
                        st.error(str(e))

            # 2) Local URI (read-only) μετά το upload
            local_uri = st.session_state.get("uploaded_local_uri", "")
            st.text_input("Local URI (inside agent container)", value=local_uri, disabled=True, key="ds_local_uri")

            # 3) Create descriptor στο backend (/datasets)
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

                    # Καταγραφή run (history)
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

    # -------------------------
    # List datasets (όλοι οι ρόλοι)
    # -------------------------
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

    # -------------------------
    # Hospital-only: Validate + Expose features
    # -------------------------
    if role_norm() == "Hospital":
        st.subheader("Validate dataset descriptor")
        ds_map = {f"{d['name']} | {d['dataset_id']}": d["dataset_id"] for d in datasets}
        selected_label = st.selectbox("Select dataset", list(ds_map.keys()), key="ds_validate_sel")
        selected_id = ds_map[selected_label]

        # Validate call στο backend: backend καλεί agent /validate και αποθηκεύει columns/row_count
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

        # Παίρνουμε το τρέχον dataset record από τη λίστα
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
    """
    Consent Policies page:
    - Hospital-only
    - Δημιουργεί policy (POST /consents)
    - Λιστάρει policies (GET /consents)
    """
    st.header("Consent Policies")
    require_login()
    flash_show()

    if role_norm() != "Hospital":
        st.info("Consent management is Hospital-only in this PoC.")
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
    """
    Access Requests page:
    - Researcher/Biobank: υποβάλλουν request (POST /access-requests)
    - Όλοι: βλέπουν requests (GET /access-requests)
    - Hospital: εγκρίνει/απορρίπτει (PATCH /access-requests/{id}/decision)
    """
    st.header("Access Requests")
    require_login()
    flash_show()

    # -------------------------
    # External roles: Submit access request
    # -------------------------
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

    # -------------------------
    # List access requests
    # -------------------------
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

    # -------------------------
    # Hospital-only: Approve / Deny
    # -------------------------
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

        # PATCH decision στο backend
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
    """
    Client-side check (PoC) για να εμποδίσει Researcher/Biobank να τρέξει job
    αν δεν έχει approved access request.

    Πώς δουλεύει:
    - Κάνει GET /access-requests
    - Βρίσκει request με:
      - ίδιο dataset_id
      - status == approved
      - requested_by == current username
    - Αν υπάρχει, επιστρέφει True.

    Σημαντικό:
    - Αυτό είναι “UI safeguard”. Η πραγματική ασφάλεια πρέπει να μπει στο backend.
    """
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
    """
    Σελίδα "Federated Jobs" (Streamlit UI).

    Τι κάνει συνολικά:
    1) Εμφανίζει φόρμα δημιουργίας Federated Job (create job)
    2) Εμφανίζει φόρμα εκκίνησης Federated Job (run/start job)
    3) Μετά το run, εμφανίζει αποτελέσματα (global_model + metrics)
       και επιπλέον sections (privacy, feature metrics, trends, correlation matrix)
       ανάλογα με τις επιλογές του χρήστη.

    Σημαντικό (αρχιτεκτονική):
    - Η εκτέλεση (train_round, aggregates, metrics) γίνεται στον Hospital Agent.
    - Το Streamlit καλεί ΜΟΝΟ backend endpoints:
        - GET /datasets
        - POST /fl/jobs
        - POST /fl/jobs/{job_id}/start
    - Κανένα raw dataset δεν φεύγει από το νοσοκομειακό περιβάλλον (PoC αρχή).
    """
    # Τίτλος σελίδας
    st.header("Federated Jobs")

    # Guard: αν δεν υπάρχει token, σταματάει τη σελίδα και ζητά login
    require_login()

    # Εμφανίζουμε τυχόν flash message από προηγούμενη ενέργεια
    flash_show()

    # Περιγραφικό κείμενο για τον χρήστη (UX)
    st.caption(
        "This Platform runs federated computation against the hosting Hospital Agent. "
        "No raw dataset leaves the hospital boundary."
    )

    # Φορτώνουμε datasets από το backend
    # Αν αποτύχει (π.χ. backend down / auth failure), εμφανίζουμε error και σταματάμε
    try:
        datasets = api_get("/datasets")
    except Exception as e:
        st.error(str(e))
        return

    # Αν δεν υπάρχουν datasets, δεν μπορούμε να δημιουργήσουμε job
    if not datasets:
        st.warning("No datasets available.")
        return

    # -------------------------
    # Create job
    # -------------------------
    # UI section για δημιουργία νέου Federated Job
    st.subheader("Create job")

    # Δημιουργούμε mapping από "φιλικό label" -> dataset_id
    # ώστε ο χρήστης να επιλέγει εύκολα dataset από dropdown,
    # αλλά εμείς να κρατάμε ως πραγματική τιμή το dataset_id.
    ds_map = {
        f"{d['name']} | owner={d['owner_org']} | {d['dataset_id']}": d["dataset_id"]
        for d in datasets
    }

    # Dropdown επιλογής dataset
    ds_sel = st.selectbox("Dataset", list(ds_map.keys()))

    # Πραγματική τιμή dataset_id που θα στείλουμε στο backend
    dataset_id = ds_map[ds_sel]

    # Παίρνουμε ολόκληρο το dataset record (dict) που αντιστοιχεί στο dataset_id
    # Χρήσιμο για να βρούμε columns/exposed_features και να κάνουμε feature selection.
    ds_current = next(
        (d for d in datasets if str(d.get("dataset_id")) == str(dataset_id)),
        None,
    )

    # Ο χρήστης επιλέγει αριθμό rounds (iteration steps) για το PoC job
    rounds = st.number_input("Rounds", min_value=1, max_value=50, value=3, step=1)

    # Frontend-level guard: μόνο συγκεκριμένοι ρόλοι μπορούν να δημιουργούν jobs.
    # (Το backend έχει επίσης RBAC — αυτό είναι απλώς UX safety.)
    if role_norm() not in ("Researcher", "Biobank", "Hospital"):
        st.error("Only Hospital, Researcher and Biobank can create FL jobs.")
        st.stop()

    # Επιλογή διαθέσιμων features:
    # - Researcher/Biobank βλέπουν μόνο exposed_features
    # - Hospital/Admin βλέπουν exposed_features αν υπάρχουν, αλλιώς columns
    available_features = _features_available_to_requester(ds_current or {})

    # Multiselect για επιλογή χαρακτηριστικών (features)
    # default: τα 2 πρώτα αν υπάρχουν, αλλιώς όσα υπάρχουν
    selected_features = st.multiselect(
        "Features (select one or more)",
        options=available_features,
        default=available_features[:2] if len(available_features) >= 2 else available_features,
    )

    # (Προαιρετικό) label field (π.χ. target variable)
    label = st.text_input("Label (optional)", value="")

    # (Σημειώσεις) για καταγραφή σκοπού job
    notes = st.text_area("Notes", value="Compute federated statistics (PoC).")

    # Κουμπί δημιουργίας job
    if st.button("Create FL Job", type="primary"):
        try:
            # Για Researcher/Biobank απαιτούμε προηγούμενο APPROVED access request
            # Αυτό δεν είναι το "real" security layer (backend κάνει RBAC),
            # αλλά είναι χρήσιμο UI check ώστε να μην τρέχει άδικα.
            if not _is_request_approved_for_user(dataset_id):
                st.error(
                    "You need an APPROVED access request for this dataset "
                    "before creating/running a federated job."
                )
                st.stop()

            # Validation: πρέπει να έχει επιλεγεί τουλάχιστον 1 feature
            if not selected_features:
                st.error("Select at least one feature.")
                st.stop()

            # Payload που θα σταλεί στο backend endpoint POST /fl/jobs
            payload = {
                "dataset_id": dataset_id,
                "rounds": int(rounds),
                "features": selected_features,
                "label": label.strip() or None,
                "notes": notes,
            }

            # Δημιουργία job στο backend (επιστρέφει FLJob record)
            job = api_post("/fl/jobs", payload)

            # UX feedback: επιτυχία + εμφανίζουμε job_id
            st.success(f"Job created: {job.get('job_id')}")

            # Logging στο history σύστημα (backend /runs)
            # Best-effort: αν δεν υπάρχει endpoint, δεν "σπάει" το UI.
            log_run(
                "fl_jobs.create",
                {
                    "job_id": job.get("job_id"),
                    "dataset_id": dataset_id,
                    "rounds": int(rounds),
                    "features": selected_features,
                },
            )

            # Κρατάμε το τελευταίο job_id στο session για να το προτείνουμε στο "Run job"
            st.session_state["last_job_id"] = job.get("job_id")

            # st.rerun(): επαναφόρτωση Streamlit script ώστε να ανανεωθεί η σελίδα
            # (π.χ. να εμφανιστεί έτοιμο το job_id input με τη νέα τιμή)
            st.rerun()

        except Exception as e:
            # Αν backend επέστρεψε error ή υπάρχει network issue
            st.error(str(e))

    # -------------------------
    # Run job
    # -------------------------
    # UI section για εκκίνηση job που έχει ήδη δημιουργηθεί
    st.divider()
    st.subheader("Run job")

    # Επιλογές εμφάνισης (display controls):
    # Ο χρήστης επιλέγει ποια sections θα δει στα αποτελέσματα
    # πριν πατήσει Start job, ώστε να μην χρειάζεται να τρέχει ξανά.
    st.subheader("Display options")
    sections = st.multiselect(
        "Select which analytics sections to display",
        options=[
            "Privacy / Governance",
            "Feature Metrics (Distribution & Data Quality)",
            "Normalized Importance",
            "Round Trends",
            "Correlation Matrix",
            # "Recommended next actions",
        ],
        default=[
            "Feature Metrics (Distribution & Data Quality)",
            "Normalized Importance",
            "Round Trends",
            # "Recommended next actions",
        ],
        key="fl_display_sections",
    )

    # Input για job_id:
    # - default: τελευταίο job_id που δημιουργήσαμε (session_state["last_job_id"])
    job_id = st.text_input(
        "Job ID",
        value=st.session_state.get("last_job_id", "") or ""
    )

    # Κουμπί Start job: καλεί backend endpoint POST /fl/jobs/{job_id}/start
    if st.button("Start job", type="secondary"):
        try:
            # Validation: δεν πρέπει να είναι κενό
            if not job_id.strip():
                st.error("Provide a Job ID.")
                st.stop()

            # Εκκίνηση job στο backend
            # Το backend θα orchestrate τα rounds και θα καλέσει agent /train_round
            job = api_post(f"/fl/jobs/{job_id.strip()}/start", payload={})

            # UX feedback: επιτυχής ολοκλήρωση + status και round
            st.success(
                f"Job finished: status={job.get('status')}, "
                f"round={job.get('current_round')}"
            )

            # -------------------------
            # Core outputs
            # -------------------------
            # Global model / aggregated outputs (PoC: συνήθως federated statistics)
            st.subheader("Federated Aggregates (Global)")
            st.json(job.get("global_model") or {})

            # Execution metrics (dictionary): privacy, feature_metrics, round_trends, correlation_matrix, κ.λπ.
            st.subheader("Execution Metrics")
            metrics = job.get("metrics") or {}
            st.json(metrics)

            # Αν το backend κατέγραψε last_error, το εμφανίζουμε στο UI
            if job.get("last_error"):
                st.error(job.get("last_error"))

            # -------------------------
            # Enriched analytics (display controls)
            # -------------------------

            # 1) Privacy / Governance
            # Περιλαμβάνει policy thresholds (π.χ. min_row_threshold) και suppressed_features
            privacy = metrics.get("privacy") or {}
            if "Privacy / Governance" in sections and privacy:
                st.subheader("Privacy / Governance")
                st.json(privacy)

            # 2) Feature metrics (distribution + data quality)
            # Περιμένουμε dict: feature -> stats (missing_rate, outlier_rate, mean, std, κ.λπ.)
            feature_metrics = metrics.get("feature_metrics") or metrics.get("feature_stats") or {}
            if "Feature Metrics (Distribution & Data Quality)" in sections and feature_metrics:
                st.subheader("Feature Metrics (Distribution & Data Quality)")
                try:
                    # Μετατρέπουμε σε DataFrame για καλύτερη ανάγνωση
                    fm_df = (
                        pd.DataFrame.from_dict(feature_metrics, orient="index")
                        .reset_index()
                        .rename(columns={"index": "feature"})
                    )
                    st.dataframe(fm_df, width="stretch")
                except Exception:
                    # Fallback: αν κάτι δεν πάει καλά, εμφανίζουμε raw json
                    st.json(feature_metrics)

            # 3) Normalized importance (variance-based proxy)
            # PoC: δείχνει ποια features έχουν μεγαλύτερη συμβολή/διακύμανση
            norm_imp = metrics.get("normalized_importance")
            if "Normalized Importance" in sections and norm_imp:
                st.subheader("Normalized Feature Importance (Variance-based Proxy)")
                st.json(norm_imp)

            # 4) Round trends (convergence)
            # Συνήθως dict: feature_mean -> [values per round]
            round_trends = metrics.get("round_trends")
            if "Round Trends" in sections and round_trends:
                st.subheader("Round Trends (Convergence)")
                st.json(round_trends)

            # 5) Correlation matrix (Pearson)
            # metrics["correlation_matrix"] πρέπει να είναι matrix-like dict/list
            corr = metrics.get("correlation_matrix")
            if "Correlation Matrix" in sections and corr:
                st.subheader("Federated Correlation Matrix (Pearson)")

                # Μετατρέπουμε correlation matrix σε pandas DataFrame
                corr_df = pd.DataFrame(corr).astype(float)

                # Διασφαλίζουμε ότι τα rows/cols έχουν την ίδια σειρά
                corr_df = corr_df.reindex(index=corr_df.columns, columns=corr_df.columns)

                # ---- Size control: προσαρμόζουμε μέγεθος figure ανάλογα με αριθμό features ----
                n = len(corr_df.columns)
                fig_w = min(8.0, max(4.5, 0.55 * n))
                fig_h = min(6.0, max(3.8, 0.55 * n))
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)

                # ---- Auto color scaling ----
                # Στόχος: να φαίνεται σωστά το gradient ακόμα και αν οι συσχετίσεις είναι μικρές.
                vals = corr_df.values.copy()
                np.fill_diagonal(vals, np.nan)  # αγνοούμε diagonal=1.0 για scaling
                max_abs = np.nanmax(np.abs(vals))
                if not np.isfinite(max_abs) or max_abs == 0:
                    max_abs = 1.0

                # Καθορίζουμε color range [-v, v]
                # Το floor 0.10 βοηθά να μη βγαίνει "άχρωμο" όταν οι τιμές είναι πολύ μικρές
                v = max(0.10, float(max_abs))

                # Heatmap με colormap coolwarm (αρνητικά -> μπλε, θετικά -> κόκκινο)
                im = ax.imshow(corr_df.values, vmin=-v, vmax=v, cmap="coolwarm", aspect="equal")

                # Ticks + labels
                ax.set_xticks(np.arange(n))
                ax.set_yticks(np.arange(n))
                ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
                ax.set_yticklabels(corr_df.index, fontsize=8)
                ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

                # ---- Annotations: γράφουμε τις τιμές μέσα στα κελιά ----
                vals2 = corr_df.values
                for i in range(n):
                    for j in range(n):
                        vv = float(vals2[i, j])
                        # Επιλέγουμε χρώμα γραμματοσειράς για contrast
                        txt_color = "white" if abs(vv) >= (0.6 * v) else "black"
                        ax.text(j, i, f"{vv:.2f}", ha="center", va="center", fontsize=7, color=txt_color)

                # Colorbar δίπλα στο heatmap
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=8)

                ax.set_title("Pearson correlation heatmap", fontsize=10, loc="left", pad=10)

                fig.tight_layout()

                # Προβολή του matplotlib figure στο Streamlit
                st.pyplot(fig, use_container_width=False)

            """
             6) Recommended next actions
            if "Recommended next actions" in sections:
                #st.subheader("Recommended next actions (rule-based)")
                #try:
                    #for a in suggest_actions(metrics):
                        #st.write(f"- {a}")
                #except Exception as e:
                    #st.write(f"- Δεν μπόρεσα να υπολογίσω recommendations. ({e})")
            """
        except Exception as e:
            # Αν backend/agent έβγαλε error, το εμφανίζουμε
            st.error(str(e))


def page_runs_history() -> None:
    """
    Σελίδα Runs / History.

    Τι κάνει:
    - Ζητά από backend τα runs του τρέχοντος χρήστη (mine=1)
    - Εμφανίζει κάθε run σε expander
    - Παρέχει download JSON για τεκμηρίωση/traceability (audit-like χρήση)
    """
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


def page_settings() -> None:
    """
    Σελίδα Settings.

    Τι κάνει:
    - Εμφανίζει πληροφορίες για backend URL (runtime config)
    - Εμφανίζει debug info για session (username/role/org/token_present)
    """
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
    """
    Main entrypoint της Streamlit εφαρμογής.

    Ροή:
    1) Θέτουμε page config και topbar
    2) Αν δεν υπάρχει token -> εμφανίζουμε login/register UI
    3) Αν υπάρχει token -> sidebar navigation βάσει ρόλου
    4) Route σελίδας (Dashboard, Nodes, Datasets, κ.λπ.)
    """
    # Βασικό configuration της σελίδας (τίτλος + layout)
    st.set_page_config(page_title="BC-FL Platform", layout="wide")

    # Top bar: backend URL + user info + logout
    ui_topbar()

    # Αν δεν έχει γίνει login, δείχνουμε μόνο την auth οθόνη
    if not st.session_state.get("token"):
        ui_login_register()
        return

    # Σελίδες που επιτρέπονται βάσει ρόλου
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
    elif page == "Settings":
        page_settings()
    else:
        st.write("Unknown page.")


# Standard Python entrypoint guard
# Όταν τρέχουμε το αρχείο ως script (streamlit run streamlit_app.py),
# εκτελείται το main().
if __name__ == "__main__":
    main()