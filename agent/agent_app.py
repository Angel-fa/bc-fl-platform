# agent_app.py
from __future__ import annotations

# ----------------------------
# Standard libraries
# ----------------------------
import csv               # Διαβάζουμε CSV αρχεία με DictReader (header -> dict ανά γραμμή)
import json              # JSON encoding/decoding (π.χ. payloads προς orchestrator, tokens)
import os                # Πρόσβαση σε env vars + filesystem paths
import time              # timestamps (π.χ. ονόματα αρχείων upload, token payload time)
import urllib.request    # Lightweight HTTP client χωρίς extra dependency (requests) στο agent
from typing import Any, Dict, List, Optional  # Type hints για καθαρότητα/τεκμηρίωση

# ----------------------------
# Third-party libraries
# ----------------------------
import numpy as np        # αριθμητικοί υπολογισμοί (std, quantiles, z-score outliers)
import pandas as pd       # DataFrame για υπολογισμό enriched metrics & correlation

from fastapi import FastAPI, Header, HTTPException, UploadFile, File  # API server + request primitives
from pydantic import BaseModel, Field                                 # request schemas/validation
from openpyxl import load_workbook                                    # read-only ανάγνωση Excel χωρίς pandas overhead


# ----------------------------
# Env (Configuration μέσω environment variables)
# ----------------------------
# Secret shared με orchestrator/UI για να προστατεύονται τα endpoints του agent (upload/train_round κ.λπ.)
APP_SECRET = os.getenv("AGENT_REG_SECRET", "dev-secret")

# Base URL του orchestrator/backend (όπως φαίνεται *μέσα* στο Docker network)
ORCHESTRATOR_BASE_URL = os.getenv("ORCHESTRATOR_BASE_URL", "http://backend:8000/api/v1")

# Φιλικό όνομα του node (για UI/monitoring)
NODE_NAME = os.getenv("NODE_NAME", "Hospital Agent")

# Ο οργανισμός/νοσοκομείο στο οποίο ανήκει αυτός ο agent
HOSPITAL_ORG = os.getenv("HOSPITAL_ORG", "HospitalA")

# Base URL με το οποίο *οι άλλες υπηρεσίες* (backend) θα καλούν τον agent
# Συνήθως είναι το service name στο docker compose (όχι localhost).
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://hospital_a_agent:9001")

# Directory μέσα στο container όπου αποθηκεύονται τα uploads (π.χ. /data volume-mounted)
DATA_DIR = os.getenv("DATA_DIR", "/data")

# Μέγιστο μέγεθος upload για προστασία (default 200MB)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# Outlier detection threshold (z-score)
OUTLIER_Z = float(os.getenv("OUTLIER_Z", "3.0"))

# Privacy threshold: κάτω από αυτό, κάνουμε suppression (δεν επιστρέφουμε aggregates)
MIN_ROWS_THRESHOLD = int(os.getenv("MIN_ROWS_THRESHOLD", "50"))

# ----------------------------
# Consent filtering (PoC)
# ----------------------------
# Αν είναι ενεργό, ο agent φιλτράρει γραμμές χωρίς consent.
# Σημ.: για PoC το consent check γίνεται με call προς orchestrator endpoint /consents/has.
CONSENT_FILTER_ENABLED = str(os.getenv("CONSENT_FILTER_ENABLED", "0")).strip().lower() in ("1", "true", "yes", "on")

# Όνομα στήλης που περιέχει patient pseudonymous id, ώστε να γίνει per-row consent check
PATIENT_ID_COLUMN = os.getenv("PATIENT_ID_COLUMN", "patient_id")

# Shared secret που πρέπει να ταιριάζει με το backend PATIENT_PORTAL_SECRET
# Χρησιμοποιείται στο /patient/consent-link PoC
PATIENT_PORTAL_SECRET = os.getenv("PATIENT_PORTAL_SECRET", "")

# Ενεργοποίηση "enriched metrics" (feature stats, importance, correlation)
ENRICHED_METRICS_ENABLED = str(os.getenv("ENRICHED_METRICS_ENABLED", "1")).strip().lower() in ("1", "true", "yes", "on")

# Cap ώστε να μην διαβάζουμε άπειρες γραμμές για metrics (προστασία χρόνου/μνήμης)
ENRICHED_SAMPLE_CAP = int(os.getenv("ENRICHED_SAMPLE_CAP", "20000"))

# Δημιουργία FastAPI app για τον agent service
app = FastAPI(title="BCFL Hospital Agent")


# ----------------------------
# Models (Pydantic request bodies)
# ----------------------------
class ValidateRequest(BaseModel):
    # local_uri: path μέσα στο agent container όπου βρίσκεται το αρχείο
    local_uri: str = Field(..., min_length=1)
    # schema_id: απλό id για versioning του schema (PoC)
    schema_id: str = Field(..., min_length=1)


class TrainRoundRequest(BaseModel):
    # job_id/round: έννοιες FL orchestration (job instance + τρέχον γύρος)
    job_id: str
    round: int

    # dataset descriptor fields (έρχονται από orchestrator μέσω backend)
    dataset_id: str
    local_uri: str
    schema_id: str

    # Features που θα υπολογιστούν (π.χ. mean, metrics κ.λπ.)
    features: List[str] = Field(default_factory=list)

    # label: PoC (π.χ. target column) - εδώ δεν χρησιμοποιείται εκτενώς
    label: Optional[str] = None

    # stratify_by: προαιρετικό field για stratified metrics (π.χ. Gender)
    stratify_by: Optional[str] = None


class PatientConsentLinkRequest(BaseModel):
    # dataset και patient ids για να παραχθεί consent-link token (PoC helper)
    dataset_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)


# ----------------------------
# Helpers (utility functions)
# ----------------------------
def _check_secret(x_agent_secret: Optional[str]) -> None:
    """
    Επιβεβαιώνει ότι το request προς τον agent έχει σωστό shared secret.

    - Το UI (Streamlit) και το backend (orchestrator) πρέπει να στέλνουν header:
        X-Agent-Secret: <AGENT_REG_SECRET>
    - Αν δεν ταιριάζει, επιστρέφουμε 403.
    """
    if x_agent_secret != APP_SECRET:
        raise HTTPException(status_code=403, detail="Invalid agent secret")


def _ensure_data_dir() -> None:
    """
    Δημιουργεί το DATA_DIR αν δεν υπάρχει.
    Χρήσιμο σε container που ξεκινάει "καθαρό" και θέλει να γράψει uploads.
    """
    os.makedirs(DATA_DIR, exist_ok=True)


def _safe_float(x: Any) -> Optional[float]:
    """
    Μετατρέπει με ασφάλεια μια τιμή σε float.
    Επιστρέφει None αν:
      - είναι None
      - είναι κενό string
      - δεν γίνεται parse σε float
    """
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _read_csv_head(path: str, max_rows: int = 5000) -> Dict[str, Any]:
    """
    Διαβάζει "κεφάλι" CSV:
      - columns (fieldnames)
      - row_count έως max_rows (γρήγορος έλεγχος)
    Δεν φορτώνει όλο το dataset (performance/PoC).
    """
    if not os.path.exists(path):
        return {"ok": False, "error": f"File not found: {path}"}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        n = 0
        for _ in reader:
            n += 1
            if n >= max_rows:
                break
    return {"ok": True, "columns": cols, "row_count": n}


def _read_xlsx_head(path: str, max_rows: int = 5000) -> Dict[str, Any]:
    """
    Διαβάζει "κεφάλι" Excel:
      - header (1η γραμμή) ως columns
      - row_count έως max_rows (PoC)
    Χρησιμοποιεί openpyxl read_only=True για λιγότερη μνήμη.
    """
    if not os.path.exists(path):
        return {"ok": False, "error": f"File not found: {path}"}

    wb = load_workbook(filename=path, read_only=True, data_only=True)
    ws = wb.active

    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header_row:
        return {"ok": False, "error": "Empty Excel file"}

    cols = [str(c).strip() for c in header_row if c is not None and str(c).strip() != ""]

    n = 0
    for _ in ws.iter_rows(min_row=2, max_row=1 + max_rows, values_only=True):
        n += 1

    return {"ok": True, "columns": cols, "row_count": n}


def _detect_columns(path: str) -> Dict[str, Any]:
    """
    Detects file type από extension και διαβάζει columns/row_count (head only).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _read_csv_head(path)
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return _read_xlsx_head(path)
    return {"ok": False, "error": f"Unsupported file type: {ext}. Allowed: .csv, .xlsx"}


def _http_post(url: str, payload: dict, timeout: int = 8) -> dict:
    """
    Minimal HTTP POST helper με urllib.
    Χρησιμοποιείται κυρίως για agent->orchestrator επικοινωνία (π.χ. register node, consent check).
    """
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


# ----------------------------
# Consent (optional) + caching
# ----------------------------
# Cache για consent results ώστε να μην κάνουμε POST /consents/has για κάθε γραμμή συνέχεια.
# Structure: dataset_id -> patient_id -> bool
_cons_cache: Dict[str, Dict[str, bool]] = {}


def _orchestrator_has_consent(dataset_id: str, patient_id: str) -> bool:
    """
    Κάνει call στο orchestrator/backend για να ελέγξει consent.

    Expected backend endpoint:
      POST /api/v1/consents/has  { "dataset_id": "...", "patient_id": "..." }
        -> { "has_consent": true/false }

    Αν αποτύχει (network error / endpoint missing), επιστρέφει False για safety.
    """
    try:
        url = f"{ORCHESTRATOR_BASE_URL.rstrip('/')}/consents/has"
        payload = {"dataset_id": dataset_id, "patient_id": patient_id}
        data = _http_post(url, payload, timeout=8)
        return bool(data.get("has_consent"))
    except Exception:
        return False


def _has_consent_cached(dataset_id: str, patient_id: str) -> bool:
    """
    Wrapper που χρησιμοποιεί in-memory cache για consent lookups.
    - Αν υπάρχει στο cache, επιστρέφει κατευθείαν.
    - Αλλιώς καλεί orchestrator και αποθηκεύει το αποτέλεσμα.
    """
    ds = (dataset_id or "").strip()
    pid = (patient_id or "").strip()
    if not ds or not pid:
        return False

    ds_cache = _cons_cache.setdefault(ds, {})
    if pid in ds_cache:
        return ds_cache[pid]

    ok = _orchestrator_has_consent(ds, pid)
    ds_cache[pid] = ok
    return ok


# ----------------------------
# Training computation (enriched metrics helpers)
# ----------------------------
def _iqr_stats(series: pd.Series) -> Dict[str, float]:
    """
    Υπολογίζει IQR (Inter-Quartile Range): Q3-Q1.
    Χρησιμοποιείται σαν robust distribution statistic.
    """
    q1 = float(series.quantile(0.25)) if series.notna().any() else 0.0
    q3 = float(series.quantile(0.75)) if series.notna().any() else 0.0
    return {"q1": q1, "q3": q3, "iqr": float(q3 - q1)}


def _outlier_rate_zscore(series: pd.Series, z: float = 3.0) -> float:
    """
    Υπολογίζει outlier rate με z-score:
      outlier αν |(x-mean)/std| > z
    Επιστρέφει ποσοστό (0..1).
    """
    s = series.dropna()
    if len(s) < 5:
        return 0.0
    std = float(s.std(ddof=0))
    if std == 0.0:
        return 0.0
    mean = float(s.mean())
    zz = ((s - mean) / std).abs()
    return float((zz > z).mean())

def _compute_enriched_feature_metrics_df(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """
    Returns:
      feature_metrics: per feature stats (distribution + quality)
      normalized_importance: variance normalized to sum=1 (proxy)
      correlation_matrix: Pearson for numeric features (optional)

    Στόχος:
    - Να επιστρέψουμε "πλούσια" metrics (χωρίς raw data)
    - Για να μπορεί ο orchestrator/UI να δείξει:
      * data quality (missing/outliers)
      * feature importance proxy (variance)
      * correlation heatmap
    """
    feature_metrics: Dict[str, Dict[str, Any]] = {}
    variances: Dict[str, float] = {}

    for feat in features:
        # Αν feature δεν υπάρχει στο dataset, το αγνοούμε
        if feat not in df.columns:
            continue

        col = df[feat]
        # missing_rate: ποσοστό NaN/empty (pandas NaN)
        missing_rate = float(col.isna().mean())

        # Μετατροπή σε numeric όπου γίνεται (non-numeric -> NaN)
        num = pd.to_numeric(col, errors="coerce")
        non_na = num.dropna()

        # Εκτίμηση "σταθερότητας" feature
        unique_values = int(col.nunique(dropna=True))
        is_constant = bool(unique_values <= 1)

        # Distribution stats (αν δεν υπάρχουν numeric τιμές, defaults 0)
        mean = float(non_na.mean()) if len(non_na) else 0.0
        std = float(non_na.std(ddof=0)) if len(non_na) else 0.0
        vmin = float(non_na.min()) if len(non_na) else 0.0
        vmax = float(non_na.max()) if len(non_na) else 0.0
        median = float(non_na.median()) if len(non_na) else 0.0

        # Robust stats
        iqr = _iqr_stats(non_na) if len(non_na) else {"q1": 0.0, "q3": 0.0, "iqr": 0.0}
        outlier_rate = _outlier_rate_zscore(non_na, z=3.0) if len(non_na) else 0.0

        # Variance ως proxy “importance”
        var = float(non_na.var(ddof=0)) if len(non_na) else 0.0
        variances[feat] = var

        # Συγκεντρώνουμε metrics για το συγκεκριμένο feature
        feature_metrics[feat] = {
            "mean": mean,
            "std": std,
            "min": vmin,
            "max": vmax,
            "median": median,
            "q1": float(iqr["q1"]),
            "q3": float(iqr["q3"]),
            "iqr": float(iqr["iqr"]),
            "missing_rate": missing_rate,
            "outlier_rate": float(outlier_rate),
            "unique_values": unique_values,
            "is_constant": is_constant,
            "variance": var,
        }

    # Normalized importance: variance / sum(variances)
    total_var = float(sum(max(v, 0.0) for v in variances.values()))
    normalized_importance = {
        k: (float(max(v, 0.0)) / total_var if total_var > 0 else 0.0)
        for k, v in variances.items()
    }

    # Pearson correlation matrix (μόνο numeric, και μόνο αν έχουμε >=2 numeric cols)
    corr_matrix: Dict[str, Dict[str, float]] = {}
    numeric_df = df[features].apply(pd.to_numeric, errors="coerce")
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(method="pearson")
        corr_matrix = {c: {r: float(corr.loc[r, c]) for r in corr.index} for c in corr.columns}

    return {
        "feature_metrics": feature_metrics,
        "normalized_importance": normalized_importance,
        "correlation_matrix": corr_matrix,
    }


def _compute_feature_means_csv(
    path: str,
    features: List[str],
    dataset_id: str,
    max_rows: int = 200000,
) -> Dict[str, Any]:
    """
    Υπολογίζει MEANS ανά feature για CSV χωρίς να φορτώνει όλο το αρχείο σε RAM.
    - Διαβάζει γραμμή-γραμμή (streaming).
    - Optionally κάνει consent filtering ανά γραμμή.
    - Παράγει:
        update: {feature: mean}
        metrics: info για processing + enriched metrics (best-effort με pandas)
    """
    if not os.path.exists(path):
        return {"ok": False, "error": f"File not found: {path}"}

    # streaming aggregates (sum/count ανά feature)
    sums: Dict[str, float] = {f: 0.0 for f in features}
    counts: Dict[str, int] = {f: 0 for f in features}

    rows_total = 0
    rows_used = 0
    rows_skipped_no_consent = 0

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1

            # Consent filter: αν ON, κρατάμε μόνο γραμμές που έχουν consent για τον patient_id
            if CONSENT_FILTER_ENABLED:
                pid = (row.get(PATIENT_ID_COLUMN) or "").strip()
                if not pid or not _has_consent_cached(dataset_id, pid):
                    rows_skipped_no_consent += 1
                    if rows_total >= max_rows:
                        break
                    continue

            rows_used += 1

            # Προσθήκη στους sums/counts μόνο αν η τιμή γίνεται float
            for feat in features:
                v = _safe_float(row.get(feat))
                if v is not None:
                    sums[feat] += v
                    counts[feat] += 1

            # Safety cap στο processing
            if rows_total >= max_rows:
                break

    # Τελικοί means
    means: Dict[str, float] = {}
    for feat in features:
        means[feat] = (sums[feat] / float(counts[feat])) if counts[feat] > 0 else 0.0

    # Enriched metrics via pandas (best-effort).
    # Προσοχή: εδώ ΔΕΝ εφαρμόζουμε consent filtering στο df.
    # Γι’ αυτό και στο train_round παρακάτω, όταν CONSENT_FILTER_ENABLED=1,
    # κρατάμε minimal output και δεν βασιζόμαστε σε αυτά.
    try:
        df = pd.read_csv(path, nrows=max_rows)
        enriched = _compute_enriched_feature_metrics_df(df, features)
    except Exception:
        enriched = {"feature_metrics": {}, "normalized_importance": {}, "correlation_matrix": {}}

    return {
        "ok": True,
        "row_count": rows_total,
        "update": means,
        "metrics": {
            "rows_processed": rows_total,
            "rows_used": rows_used,
            "rows_skipped_no_consent": rows_skipped_no_consent,
            "consent_filter_enabled": CONSENT_FILTER_ENABLED,
            "patient_id_column": PATIENT_ID_COLUMN if CONSENT_FILTER_ENABLED else None,

            # Enriched (optional)
            "feature_metrics": enriched.get("feature_metrics", {}),
            "normalized_importance": enriched.get("normalized_importance", {}),
            "correlation_matrix": enriched.get("correlation_matrix", {}),
        },
    }


def _compute_feature_means_xlsx(
    path: str,
    features: List[str],
    dataset_id: str,
    max_rows: int = 200000,
) -> Dict[str, Any]:
    """
    Αντίστοιχο του CSV αλλά για Excel (openpyxl streaming).
    Υπολογίζει means ανά feature.
    """
    if not os.path.exists(path):
        return {"ok": False, "error": f"File not found: {path}"}

    wb = load_workbook(filename=path, read_only=True, data_only=True)
    ws = wb.active

    header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header_row:
        return {"ok": False, "error": "Empty Excel file"}

    # Map: column name -> index
    cols = [str(c).strip() if c is not None else "" for c in header_row]
    col_index = {name: idx for idx, name in enumerate(cols) if name}

    # Αν consent filter είναι ON, βρίσκουμε index της patient_id στήλης
    pid_idx = col_index.get(PATIENT_ID_COLUMN) if CONSENT_FILTER_ENABLED else None

    sums: Dict[str, float] = {f: 0.0 for f in features}
    counts: Dict[str, int] = {f: 0 for f in features}

    rows_total = 0
    rows_used = 0
    rows_skipped_no_consent = 0

    # iterate rows (data starts at row 2)
    for row in ws.iter_rows(min_row=2, max_row=1 + max_rows, values_only=True):
        rows_total += 1

        if CONSENT_FILTER_ENABLED:
            pid = ""
            if pid_idx is not None and pid_idx < len(row):
                pid = str(row[pid_idx] or "").strip()
            if not pid or not _has_consent_cached(dataset_id, pid):
                rows_skipped_no_consent += 1
                continue

        rows_used += 1

        for feat in features:
            idx = col_index.get(feat)
            if idx is None or idx >= len(row):
                continue
            v = _safe_float(row[idx])
            if v is not None:
                sums[feat] += v
                counts[feat] += 1

    means: Dict[str, float] = {}
    for feat in features:
        means[feat] = (sums[feat] / float(counts[feat])) if counts[feat] > 0 else 0.0

    # Enriched metrics via pandas (best-effort, ίδιο caveat με consent filter)
    try:
        df = pd.read_excel(path, nrows=max_rows)
        enriched = _compute_enriched_feature_metrics_df(df, features)
    except Exception:
        enriched = {"feature_metrics": {}, "normalized_importance": {}, "correlation_matrix": {}}

    return {
        "ok": True,
        "row_count": rows_total,
        "update": means,
        "metrics": {
            "rows_processed": rows_total,
            "rows_used": rows_used,
            "rows_skipped_no_consent": rows_skipped_no_consent,
            "consent_filter_enabled": CONSENT_FILTER_ENABLED,
            "patient_id_column": PATIENT_ID_COLUMN if CONSENT_FILTER_ENABLED else None,

            # Enriched (optional)
            "feature_metrics": enriched.get("feature_metrics", {}),
            "normalized_importance": enriched.get("normalized_importance", {}),
            "correlation_matrix": enriched.get("correlation_matrix", {}),
        },
    }


def _compute_feature_means(path: str, features: List[str], dataset_id: str, max_rows: int = 200000) -> Dict[str, Any]:
    """
    Dispatcher: ανάλογα με extension επιλέγει CSV ή Excel μέθοδο.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return _compute_feature_means_csv(path, features, dataset_id=dataset_id, max_rows=max_rows)
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return _compute_feature_means_xlsx(path, features, dataset_id=dataset_id, max_rows=max_rows)
    return {"ok": False, "error": f"Unsupported file type for training: {ext}"}


def _read_df_for_metrics(path: str, columns: List[str], max_rows: int) -> pd.DataFrame:
    """
    Minimal loader για metrics:
    - Διαβάζει ΜΟΝΟ τις στήλες που χρειάζονται (features + stratify_by).
    - Περιορίζει γραμμές σε max_rows.
    """
    ext = os.path.splitext(path)[1].lower()
    usecols = [c for c in (columns or []) if c]

    if ext == ".csv":
        return pd.read_csv(path, usecols=usecols, nrows=max_rows, low_memory=False)
    if ext in (".xlsx", ".xlsm", ".xltx", ".xltm"):
        return pd.read_excel(path, usecols=usecols, nrows=max_rows)
    raise ValueError(f"Unsupported file type for metrics: {ext}")


def _compute_feature_metrics_df(df: pd.DataFrame, features: List[str], outlier_z: float) -> Dict[str, Any]:
    """
    Πιο "γενικό" metrics calculator (numeric + categorical).

    Numeric:
      mean,std,min,max,median,q1,q3,iqr + missing_rate,outlier_rate,unique,is_constant

    Categorical:
      missing_rate,unique,unique_ratio,is_constant

    Επιστρέφει dictionary per feature ώστε ο orchestrator/UI να τα εμφανίσει χωρίς raw data.
    """
    res: Dict[str, Any] = {}

    for f in features:
        if f not in df.columns:
            continue

        s_raw = df[f]
        total = int(len(s_raw))
        missing = int(s_raw.isna().sum())

        # Δοκιμή numeric conversion
        s_num = pd.to_numeric(s_raw, errors="coerce")
        non_na = s_num.dropna()

        if len(non_na) > 0:
            # numeric case
            x = non_na.to_numpy(dtype=float)
            n = int(x.size)

            mean = float(np.mean(x))
            std = float(np.std(x, ddof=0))
            vmin = float(np.min(x))
            vmax = float(np.max(x))

            q1 = float(np.quantile(x, 0.25))
            med = float(np.quantile(x, 0.50))
            q3 = float(np.quantile(x, 0.75))
            iqr = float(q3 - q1)

            outliers = 0
            if std > 0:
                z = np.abs((x - mean) / std)
                outliers = int((z > outlier_z).sum())

            missing_rate = float(missing / max(total, 1))
            outlier_rate = float(outliers / max(n, 1))

            uniq = int(pd.Series(x).nunique(dropna=True))
            is_constant = bool(uniq <= 1)

            res[f] = {
                "type": "numeric",
                "total": total,
                "missing": missing,
                "n": n,
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
                "q1": q1,
                "median": med,
                "q3": q3,
                "iqr": iqr,
                "missing_rate": missing_rate,
                "outliers": outliers,
                "outlier_rate": outlier_rate,
                "unique": uniq,
                "is_constant": is_constant,
            }
        else:
            # categorical / non-numeric case
            non_na_raw = s_raw.dropna()
            uniq = int(non_na_raw.nunique(dropna=True))
            denom = max(int(len(non_na_raw)), 1)

            res[f] = {
                "type": "categorical",
                "total": total,
                "missing": missing,
                "missing_rate": float(missing / max(total, 1)),
                "unique": uniq,
                "unique_ratio": float(uniq / denom) if len(non_na_raw) else 0.0,
                "is_constant": bool(uniq <= 1),
            }

    return res


def _compute_stratified_metrics(df: pd.DataFrame, stratify_by: str, features: List[str], outlier_z: float) -> Dict[str, Any]:
    """
    Stratified metrics:
    - Ομαδοποιεί το dataset με βάση τη στήλη stratify_by (π.χ. Gender)
    - Για κάθε ομάδα, υπολογίζει feature metrics
    """
    if not stratify_by or stratify_by not in df.columns:
        return {}

    out: Dict[str, Any] = {}

    # dropna ώστε να μην έχουμε NaN group
    for gval, gdf in df.dropna(subset=[stratify_by]).groupby(df[stratify_by].astype(str)):
        out[str(gval)] = _compute_feature_metrics_df(gdf, features, outlier_z)

    return out


# ----------------------------
# Routes (FastAPI endpoints)
# ----------------------------
@app.get("/health")
def health():
    """
    Health endpoint για monitoring/debugging.
    Επιστρέφει:
      - status
      - org/name ώστε να ξέρουμε ποιο agent απαντάει
    """
    return {
        "status": "ok",
        "org": HOSPITAL_ORG,
        "name": NODE_NAME
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...), x_agent_secret: Optional[str] = Header(default=None)):
    """
    Upload endpoint:
    - Προστατευμένο με X-Agent-Secret
    - Αποθηκεύει το αρχείο στο DATA_DIR
    - Επιστρέφει local_uri (path στο container), columns και row_count (head)
    """
    _check_secret(x_agent_secret)
    _ensure_data_dir()

    # Διαβάζουμε όλο το αρχείο στη μνήμη (PoC).
    # Για πολύ μεγάλα αρχεία, θες streaming, αλλά εδώ έχουμε MAX_UPLOAD_BYTES για προστασία.
    contents = await file.read()
    if contents is None:
        raise HTTPException(status_code=400, detail="Empty upload")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max is {MAX_UPLOAD_BYTES} bytes")

    filename = (file.filename or "upload").strip()
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".csv", ".xlsx", ".xlsm", ".xltx", ".xltm"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: CSV, XLSX")

    # Σχηματίζουμε ασφαλές όνομα αρχείου (απλή αντικατάσταση κενών)
    safe_name = filename.replace(" ", "_")

    # local_uri: path μέσα στο container (DATA_DIR) με timestamp prefix ώστε να μην συγκρούονται ονόματα
    local_uri = os.path.join(DATA_DIR, f"{int(time.time())}_{safe_name}")

    # Γράφουμε στο δίσκο/volume
    with open(local_uri, "wb") as f:
        f.write(contents)

    # Διαβάζουμε columns + row_count (head) για να ενημερώσουμε UI/backend
    info = _detect_columns(local_uri)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail=info.get("error", "Failed to read columns"))

    return {
        "ok": True,
        "local_uri": local_uri,
        "columns": info.get("columns", []),
        "row_count": info.get("row_count", 0),
        "filename": filename,
    }


@app.post("/validate")
def validate(req: ValidateRequest, x_agent_secret: Optional[str] = Header(default=None)):
    """
    Validate endpoint:
    - Δίνει στο backend έναν τρόπο να ελέγξει ότι το local_uri είναι valid και να πάρει columns.
    - Προστατευμένο με X-Agent-Secret.
    """
    _check_secret(x_agent_secret)
    info = _detect_columns(req.local_uri)
    # κρατάμε και το schema_id που έστειλε ο orchestrator
    info["schema_id"] = req.schema_id
    return info


@app.post("/train_round")
def train_round(req: TrainRoundRequest, x_agent_secret: Optional[str] = Header(default=None)):
    """
    Core FL endpoint (PoC):
    - Καλείται από τον orchestrator όταν ξεκινά/τρέχει ένα job round.
    - Δεν επιστρέφει raw data.
    - Επιστρέφει "updates" (aggregates) + optional metrics.
    """
    _check_secret(x_agent_secret)

    # Features που θέλει ο orchestrator να υπολογιστούν
    features = req.features or []
    if not features:
        # Αν δεν έρθουν features, δεν έχει νόημα job. Επιστρέφουμε structured error payload.
        return {
            "ok": False,
            "error": "Provide features list",
            "row_count": 0,
            "update": {},
            "feature_metrics": {},
            "stratified_metrics": {},
            "suppressed": False,
            "min_row_threshold": MIN_ROWS_THRESHOLD,
            "metrics": {},
        }

    # 1) Υπολογισμός means (streaming) + optional consent filtering
    base = _compute_feature_means(req.local_uri, features=features, dataset_id=req.dataset_id)
    if not base.get("ok"):
        # Αν αποτύχει ο υπολογισμός (π.χ. file missing), επιστρέφουμε error
        return {
            "ok": False,
            "error": base.get("error", "compute failed"),
            "row_count": int(base.get("row_count") or 0),
            "update": {},
            "feature_metrics": {},
            "stratified_metrics": {},
            "suppressed": False,
            "min_row_threshold": MIN_ROWS_THRESHOLD,
            "metrics": base.get("metrics") or {},
        }

    row_count = int(base.get("row_count") or 0)

    # 2) Privacy / governance suppression
    # Αν το dataset είναι μικρό (N < threshold), δεν επιστρέφουμε aggregates ώστε να μειωθεί re-identification risk.
    if row_count < MIN_ROWS_THRESHOLD:
        return {
            "ok": True,
            "row_count": row_count,
            "update": {},  # suppressed: κρύβουμε aggregates
            "feature_metrics": {},
            "stratified_metrics": {},
            "suppressed": True,
            "min_row_threshold": MIN_ROWS_THRESHOLD,
            "metrics": {
                **(base.get("metrics") or {}),
                "privacy": {"min_row_threshold": MIN_ROWS_THRESHOLD, "suppressed": True},
            },
        }

    # 3) Feature metrics (distribution + quality) + optional stratification
    # Σημ.: Αν CONSENT_FILTER_ENABLED=1, εδώ επίτηδες ΔΕΝ κάνουμε pandas metrics (θα ήθελε consent-filtered df).
    feature_metrics: Dict[str, Any] = {}
    stratified_metrics: Dict[str, Any] = {}

    if CONSENT_FILTER_ENABLED:
        # Minimal output όταν consent filter είναι ON
        feature_metrics = {}
        stratified_metrics = {}
    else:
        try:
            # Διαβάζουμε μόνο τις απαραίτητες στήλες για metrics
            cols_needed = list(features)
            if req.stratify_by:
                cols_needed.append(req.stratify_by)

            df = _read_df_for_metrics(req.local_uri, cols_needed, max_rows=200000)

            # Metrics per feature
            feature_metrics = _compute_feature_metrics_df(df, features, OUTLIER_Z)

            # Stratified metrics αν έχει ζητηθεί
            if req.stratify_by:
                stratified_metrics = _compute_stratified_metrics(df, req.stratify_by, features, OUTLIER_Z)
        except Exception:
            # best-effort: δεν θέλουμε να αποτύχει όλο το job επειδή απέτυχαν τα metrics
            feature_metrics = {}
            stratified_metrics = {}

    # 4) Τελικό payload προς orchestrator/backend:
    # - update: means (backward compatible)
    # - feature_metrics/stratified_metrics: εμπλουτισμένα analytics για UI
    # - metrics: privacy info + outlier_z + base metrics
    return {
        "ok": True,
        "row_count": row_count,
        "update": base.get("update") or {},
        "feature_metrics": feature_metrics,
        "stratified_metrics": stratified_metrics,
        "suppressed": False,
        "min_row_threshold": MIN_ROWS_THRESHOLD,
        "metrics": {
            **(base.get("metrics") or {}),
            "privacy": {"min_row_threshold": MIN_ROWS_THRESHOLD, "suppressed": False},
            "outlier_z": OUTLIER_Z,
        },
    }


@app.post("/patient/consent-link")
def patient_consent_link(req: PatientConsentLinkRequest, x_agent_secret: Optional[str] = Header(default=None)):
    """
    PoC helper endpoint:
    - Δημιουργεί ένα deterministic token (hash) για future patient-portal flow.
    - Δεν γράφει consent on-chain.
    - Το backend/portal μπορεί να επαληθεύσει το token με το shared PATIENT_PORTAL_SECRET.

    Χρήση:
    - Hospital θέλει να δώσει στον ασθενή ένα link/token για να πάει στο portal και να δηλώσει consent.
    """
    _check_secret(x_agent_secret)

    if not PATIENT_PORTAL_SECRET:
        raise HTTPException(status_code=500, detail="PATIENT_PORTAL_SECRET is not set on agent")

    token_payload = {
        "dataset_id": req.dataset_id,
        "patient_id": req.patient_id,
        "org": HOSPITAL_ORG,
        "ts": int(time.time()),
    }

    # Πολύ απλό PoC signing (NOT JWT):
    # sig = sha256(secret + payload_json)
    raw = json.dumps(token_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

    import hashlib
    sig = hashlib.sha256((PATIENT_PORTAL_SECRET.encode("utf-8") + raw)).hexdigest()

    return {"ok": True, "token": sig, "payload": token_payload}


@app.on_event("startup")
def startup_register():
    """
    Κατά το startup του agent:
    - προσπαθεί να κάνει register τον εαυτό του στον orchestrator (/nodes/register).
    - best-effort: αν αποτύχει, δεν σταματάει ο agent (pass).

    Το backend μετά μπορεί να εμφανίσει τον node στη σελίδα Nodes του UI.
    """
    try:
        url = f"{ORCHESTRATOR_BASE_URL.rstrip('/')}/nodes/register?secret={APP_SECRET}"
        payload = {"org": HOSPITAL_ORG, "base_url": PUBLIC_BASE_URL, "name": NODE_NAME}
        _http_post(url, payload, timeout=8)
    except Exception:
        pass