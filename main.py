"""
Fix Before Fail (FBF) – Gasifier Analytics Backend
FastAPI + Uvicorn  |  Swagger UI at /docs
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import base64
import glob
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.special import expit

matplotlib.use("Agg")   # headless – no display needed


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = r"C:\Eminds pros\E-minds projects\fbf-backend"

GASIFIERS_DIR   = os.path.join(BASE_DIR, "Gasifiers")
BMAT_DIR        = os.path.join(BASE_DIR, "B_Matrices")
RUL_HISTORY_DIR = os.path.join(BASE_DIR, "rul_history")
LIVE_DATA_DIR   = os.path.join(BASE_DIR, "Live_data")
GRAPHS_DIR      = os.path.join(BASE_DIR, "predicted_rul_output_graphs")

# /simulation endpoint outputs (kept)
SIM_OUTPUT_DIR  = os.path.join(BASE_DIR, "simulated_outputs")
SIM_META_DIR    = os.path.join(BASE_DIR, "simulation_metadata")

# /{gasifier}/fbf/run-montecarlo outputs
FBF_MC_DIR      = os.path.join(BASE_DIR, "fbf_mc_results")     # JSON summaries
FBF_MC_CSV_DIR  = os.path.join(BASE_DIR, "fbf_mc_results", "success_cases")  # success CSVs

# /{gasifier}/fbf/custom outputs
FBF_CUSTOM_DIR  = os.path.join(BASE_DIR, "fbf_custom_results")
FBF_CUSTOM_CSV  = os.path.join(FBF_CUSTOM_DIR, "outputs")
FBF_CUSTOM_META = os.path.join(FBF_CUSTOM_DIR, "metadata")

# Alerts
ALERTS_DIR      = os.path.join(BASE_DIR, "alerts")
RUL_ALERTS_DIR  = os.path.join(ALERTS_DIR, "rul_alerts")
HEALTH_ALERTS_DIR = os.path.join(ALERTS_DIR, "health_alerts")

BUNDLE_PATH = os.path.join(BASE_DIR, "rul_bundle_v5_excluding_g5r19.joblib")

TIMESTAMP_COL  = "timestamp"
BUCKET_CHOICES = ["early", "mid", "late"]
SIM_ITERATIONS = 10    # matches simulation.ipynb

SENSORS = [
    "SlurryPDI",
    "OxygenPDI",
    "SlurryFlow",
    "Temperature",
    "SlurryPressure",
    "OxygenFlow",
    "OxygenSetpoint",
]
FEATURE_COLS = SENSORS   # alias – same 7 columns used for simulation

KNOWN_TOTAL_DAYS: dict[str, float] = {
    "g1r20":  120,
    "g4r26":  133,
    "g5r19":  120,
    "g5r21":  103,
    "g6r17":  112,
    "g9r24":  103,
    "g10r23": 107,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Fix Before Fail – Gasifier Analytics API",
    description=(
        "Backend for the FBF predictive maintenance platform.\n\n"
        "---\n\n"
        "## Overview & Assets\n"
        "| Method | Endpoint | Description |\n"
        "|--------|----------|-------------|\n"
        "| GET | `/assets` | All gasifiers with live RUL + health score |\n"
        "| GET | `/overview/plant` | Plant-level KPIs: asset counts, avg RUL & health |\n"
        "| GET | `/overview/gasifiers` | Static gasifier metadata from Gasifiers/ folder |\n"
        "| GET | `/overview/health` | Phase-wise health report **with critical indicators** (current value, % deviation, 7-day trend) |\n"
        "| GET | `/overview/collective-live-rul` | Live RUL summary for all gasifiers |\n\n"
        "## Alerts\n"
        "| Method | Endpoint | Description |\n"
        "|--------|----------|-------------|\n"
        "| GET | `/alerts` | View all RUL and health alerts (filterable by gasifier) |\n\n"
        "## Root Cause\n"
        "| Method | Endpoint | Description |\n"
        "|--------|----------|-------------|\n"
        "| GET | `/rootcause/fetch-bmat` | Fetch B-matrix for a gasifier + phase bucket with feedback-loop detection |\n\n"
        "## Prediction\n"
        "| Method | Endpoint | Description |\n"
        "|--------|----------|-------------|\n"
        "| POST | `/predict/rul` | Predict current RUL from a specific live CSV (auto-logs alerts) |\n\n"
        "## Simulation\n"
        "| Method | Endpoint | Description |\n"
        "|--------|----------|-------------|\n"
        "| POST | `/simulation` | Simulate sensor behaviour under a modified B-matrix, returns before/after RUL + plot |\n\n"
        "## FBF Interventions (per-gasifier)\n"
        "| Method | Endpoint | Description |\n"
        "|--------|----------|-------------|\n"
        "| POST | `/{gasifier}/fbf/run-montecarlo` | Physics-gated MC search. Saves success CSV **with per-sensor baseline & simulated means** per trial. |\n"
        "| POST | `/{gasifier}/fbf/custom` | Apply specific B-matrix edge values, check physics gate, compute RUL before/after |\n"
        "| GET  | `/{gasifier}/fbf/top-interventions` | Retrieve latest MC run's success-case CSV path + summary |\n"
        "| GET  | `/{gasifier}/fbf/recommend-intervention` | **Operator-ready recommendation**: reads the best MC trial and returns sensor before→after operating targets with plain-English guidance |\n\n"
        "---\n\n"
        "## Typical Intervention Workflow\n"
        "```\n"
        "1. POST /{gasifier}/fbf/run-montecarlo   → runs MC, saves success cases\n"
        "2. GET  /{gasifier}/fbf/recommend-intervention  → get best sensor targets\n"
        "3. POST /{gasifier}/fbf/custom           → Change the cell values manually (1 Iteration)\n"
        "4. GET  /overview/health                 → verify critical indicators after adjustment\n"
        "```\n"
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Overview / Assets ─────────────────────────────────────────────────────────

class GasifierMeta(BaseModel):
    gasifier_id:   str
    filename:      str
    num_rows:      int
    num_columns:   int
    timestamp_min: Optional[str]
    timestamp_max: Optional[str]
    num_days:      Optional[float]
    columns:       list[str]

class GasifierListResponse(BaseModel):
    count:     int
    gasifiers: list[GasifierMeta]

class AssetInfo(BaseModel):
    gasifier_id:      str
    filename:         str
    num_rows:         int
    num_columns:      int
    timestamp_min:    Optional[str]
    timestamp_max:    Optional[str]
    num_days:         Optional[float]
    columns:          List[str]
    current_rul_days: Optional[float]
    health_score:     Optional[float]
    health_label:     Optional[str]

class AssetsResponse(BaseModel):
    count:  int
    assets: List[AssetInfo]

class GasifierLiveStatus(BaseModel):
    gasifier_id:      str
    file_processed:   str
    current_rul_days: float
    health_score:     float
    health_label:     str   # "Healthy" | "Moderate" | "Failing"
    rul_status:       str   # "Healthy" | "Risky" | "Critical"

class PlantOverviewResponse(BaseModel):
    total_assets:      int
    healthy_count:     int
    risky_count:       int
    critical_count:    int
    avg_health_score:  float
    avg_rul_days:      float
    gasifier_statuses: List[GasifierLiveStatus]

class PhaseStats(BaseModel):
    mean: float
    std:  float

class ColumnHealthStats(BaseModel):
    column:             str
    early:              PhaseStats
    mid:                PhaseStats
    late:               PhaseStats
    early_to_mid_drift: float
    mid_to_late_drift:  float

class CriticalIndicator(BaseModel):
    """
    Snapshot of a single sensor's current operating state.

    Fields
    ------
    parameter   : Human-readable sensor name (e.g. "Temperature (°C)").
    current     : Mean value in the most-recent (late) phase of the run.
    deviation   : Percentage drift of the late-phase mean relative to the
                  early-phase mean.  Positive = rose, negative = fell.
    trend_7d    : Direction of travel over the last ~7 days of data:
                  "rising" | "falling" | "stable".
    status      : "normal" | "warning" | "critical" – based on absolute
                  deviation magnitude (>10 % warning, >25 % critical).
    """
    parameter : str
    current   : float          # late-phase mean sensor value
    deviation : float          # % drift vs early phase  (e.g. +18.0 = +18 %)
    trend_7d  : str            # "rising" | "falling" | "stable"
    status    : str            # "normal" | "warning" | "critical"

class HealthReportResponse(BaseModel):
    gasifier_id:         str
    asset_name:          str
    total_rows:          int
    phase_sizes:         dict[str, int]
    timestamp_range:     dict[str, Optional[str]]
    num_days:            Optional[float]
    health_score:        float
    health_label:        str            # "Healthy" | "Moderate" | "Failing"
    critical_indicators: list[CriticalIndicator]   # one entry per SENSOR
    columns_health:      list[ColumnHealthStats]

class LiveRulSummary(BaseModel):
    gasifier_id:      str
    file_processed:   str
    current_rul_days: float
    critical_status:  bool

class CollectiveRulResponse(BaseModel):
    total_processed: int
    summary:         list[LiveRulSummary]


# ── Alerts ────────────────────────────────────────────────────────────────────

class AlertEntry(BaseModel):
    alert_id:    str
    timestamp:   str
    gasifier_id: str
    alert_type:  str   # "rul" | "health"
    severity:    str   # "critical" | "warning"
    message:     str
    details:     dict

class AlertsResponse(BaseModel):
    total_rul_alerts:    int
    total_health_alerts: int
    rul_alerts:          List[AlertEntry]
    health_alerts:       List[AlertEntry]


# ── Root Cause ────────────────────────────────────────────────────────────────

class FeedbackLoop(BaseModel):
    node_a:        str
    node_b:        str
    weight_a_to_b: float
    weight_b_to_a: float

class RootCauseConnection(BaseModel):
    """
    A single node-to-node connection highlighted as a top root cause.

    Fields
    ------
    node_from      : Source node label (e.g. "N1").
    node_to        : Target node label (e.g. "N2").
    sensor_from    : Sensor name for the source node.
    sensor_to      : Sensor name for the target node.
    weight         : B-matrix edge weight (influence strength) from node_from → node_to.
    description    : Plain-English explanation of what this connection means.
    """
    node_from   : str
    node_to     : str
    sensor_from : str
    sensor_to   : str
    weight      : Optional[float]
    description : str


class BMatResponse(BaseModel):
    gasifier_id:          str
    bucket:               str
    filename:             str
    shape:                dict[str, int]
    lambda_val:           Optional[str]
    lmin:                 Optional[str]
    lmax:                 Optional[str]
    matrix:               list[list[Optional[float]]]
    row_labels:           list[str]
    col_labels:           list[str]
    total_feedback_loops: int
    feedback_loops:       list[FeedbackLoop]
    # ── Root cause highlights ─────────────────────────────────────────────────
    # HARDCODED for now: always returns the connections between Node 1↔Node 2
    # and Node 2↔Node 5, as these are the known primary influence pathways in
    # the gasifier B-matrix. Will be made dynamic in a future iteration once
    # we have domain-confirmed causal ranking logic.
    top_root_causes:      list[RootCauseConnection]


# ── Prediction ────────────────────────────────────────────────────────────────

class RULPredictionRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Absolute path to the live data CSV file",
        example=r"C:\Eminds pros\E-minds projects\fbf-backend\Live_data\g5r19_live_data.csv",
    )

class CurrentRULResponse(BaseModel):
    """
    Response returned by POST /predict/rul.

    Fields
    ------
    gasifier_id      : Gasifier identifier parsed from the CSV filename.
    file_processed   : Name of the CSV that was processed.
    current_rul_days : Smoothed RUL estimate at the most recent timestamp (days).

    -- Confidence (how much to trust the number) --
    confidence_score : 0–1 score blended from regime certainty, data completeness,
                       and prediction stability. Higher = more trustworthy.
    confidence_label : Human-readable bucket: "high" (≥0.75) | "medium" (≥0.50) | "low".

    -- Failure probability (when might it actually fail) --
    p_fail_7d        : Probability the gasifier fails within the next  7 days.
    p_fail_14d       : Probability the gasifier fails within the next 14 days.
    p_fail_30d       : Probability the gasifier fails within the next 30 days.
    failure_risk_label : Risk bucket derived from p_fail_30d:
                         "critical" (≥70%) | "high" (≥40%) | "moderate" (≥15%) | "low".
    """
    gasifier_id:         str
    file_processed:      str
    current_rul_days:    float
    confidence_score:    float
    confidence_label:    str
    p_fail_7d:           float
    p_fail_14d:          float
    p_fail_30d:          float
    failure_risk_label:  str

class RULStats(BaseModel):
    gasifier_id:         str
    asset_name:          str
    total_rows_raw:      int
    rows_after_clean:    int
    glitch_rows_removed: int
    run_duration_days:   float
    total_days_known:    Optional[float]
    current_rul_days:    float
    current_actual_rul:  Optional[float]
    mae_on_run:          Optional[float]
    model_used:          str
    rul_at_10pct:        Optional[float]
    rul_at_25pct:        Optional[float]
    rul_at_50pct:        Optional[float]
    rul_at_75pct:        Optional[float]
    rul_at_90pct:        Optional[float]
    mae_cv:              Optional[float]
    std_cv:              Optional[float]
    graph_path:          str
    graph_base64:        str


# ── Simulation ────────────────────────────────────────────────────────────────

class EdgeModification(BaseModel):
    row:   int    # 0-based row index in the 7×7 B-matrix
    col:   int    # 0-based col index
    value: float

    class Config:
        json_schema_extra = {"example": {"row": 1, "col": 0, "value": 0.733}}

class SimulationRequest(BaseModel):
    file_path:          str = Field(
        ...,
        description="Absolute path to the data CSV file",
        example=r"C:\Eminds pros\E-minds projects\fbf-backend\Live_data\g5r19_live_data.csv",
    )
    bucket:             str = "late"
    edge_modifications: list[EdgeModification]

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": r"C:\...\g5r19_live_data.csv",
                "bucket": "late",
                "edge_modifications": [
                    {"row": 1, "col": 0, "value": 0.733},
                    {"row": 0, "col": 1, "value": 0.492},
                ],
            }
        }

class SensorSimStats(BaseModel):
    sensor:      str
    node:        str
    mean_before: float
    mean_after:  float
    pct_change:  float
    direction:   str   # "increased" | "decreased" | "no change"

class RULComparison(BaseModel):
    elapsed_total_days: float
    rul_before:         float
    rul_after:          float
    rul_delta:          float
    improved:           bool
    pct_improvement:    float

class SimulationResponse(BaseModel):
    gasifier_id:        str
    bucket:             str
    bmat_file:          str
    total_rows:         int
    edge_modifications: list[dict]
    output_csv_path:    str
    metadata_path:      str
    rul_comparison:     RULComparison
    sensor_stats:       list[SensorSimStats]
    plot_base64:        str


# ── FBF Monte Carlo ───────────────────────────────────────────────────────────

class MCCell(BaseModel):
    row: int
    col: int

class SensorBound(BaseModel):
    lower: float
    upper: float

class MonteCarloBestWorst(BaseModel):
    pct_improvement:  float
    days_improvement: float
    simulated_rul:    float
    trial_number:     int
    cell_values:      list[float]
    physics_detail:   dict

class ThresholdBucket(BaseModel):
    label:      str    # "Conservative" | "Moderate" | "Aggressive"
    min_pct:    float
    count:      int
    best_trial: Optional[MonteCarloBestWorst]

class FBFMonteCarloRequest(BaseModel):
    file_path:      str   = Field(..., description="Absolute path to the live CSV file")
    bucket:         str   = "late"
    perturb_cells:  List[MCCell] = [
        MCCell(row=1, col=0), MCCell(row=0, col=1),
        MCCell(row=4, col=0), MCCell(row=0, col=4),
        MCCell(row=4, col=1), MCCell(row=1, col=4),
    ]
    perturb_range:  List[float] = [-1.0, 1.0]
    max_trials:     int   = 1000
    n_success:      int   = 50
    sensor_bounds:  Optional[Dict[str, SensorBound]] = Field(
        None,
        description=(
            "Per-sensor hardcoded physics bounds. Key = sensor name (e.g. 'SlurryPDI'). "
            "Simulated means must fall within [lower, upper] to pass the gate. "
            "Omit a sensor to skip its bound check."
        ),
        example={"SlurryPDI": {"lower": -1000, "upper": 1000}},
    )
    ref_rows:   int = 1000
    window_rows: int = 60

class FBFMonteCarloResponse(BaseModel):
    gasifier_id:         str
    file_processed:      str
    baseline_rul:        float
    total_trials:        int
    physics_passed:      int
    infeasible:          int
    success_cases:       int
    buckets:             List[ThresholdBucket]
    best_overall:        Optional[MonteCarloBestWorst]
    best_risk_reduction: Optional[MonteCarloBestWorst]
    results_json_path:   Optional[str]
    results_csv_path:    Optional[str]   # success-cases CSV
    time_taken_sec:      float


# ── FBF Custom ────────────────────────────────────────────────────────────────

class FBFCustomRequest(BaseModel):
    file_path:          str = Field(..., description="Absolute path to the live CSV file")
    bucket:             str = "late"
    edge_modifications: List[EdgeModification]
    sensor_bounds:      Optional[Dict[str, SensorBound]] = Field(
        None,
        description="Per-sensor hardcoded physics bounds.",
    )

class FBFCustomResponse(BaseModel):
    gasifier_id:     str
    physics_passed:  bool
    physics_detail:  Dict[str, dict]
    rul_before:      Optional[float]
    rul_after:       Optional[float]
    rul_delta:       Optional[float]
    pct_improvement: Optional[float]
    sensor_stats:    List[SensorSimStats]
    output_csv_path: Optional[str]   # simulated sensor data saved here
    metadata_path:   Optional[str]   # run metadata JSON
    message:         str


# ── FBF Top Interventions ─────────────────────────────────────────────────────

class TopInterventionsResponse(BaseModel):
    gasifier_id:      str
    run_id:           str
    timestamp:        str
    baseline_rul:     float
    success_cases:    int
    results_csv_path: str
    results_json_path: str


# ── FBF Recommend Intervention ────────────────────────────────────────────────

class SensorTarget(BaseModel):
    """
    Before/after operating target for a single sensor derived from the best
    Monte Carlo intervention.

    Fields
    ------
    sensor       : Sensor name (e.g. "Temperature").
    node         : B-matrix node label (e.g. "N4").
    current_mean : Observed mean in the baseline (last ref_rows rows).
    target_mean  : Simulated mean when the best cell values are applied.
    delta        : Absolute change (target_mean − current_mean).
    pct_change   : Percentage change relative to current_mean.
    direction    : "increase" | "decrease" | "no change" – what the operator
                   should do to this sensor's operating point.
    """
    sensor       : str
    node         : str
    current_mean : float
    target_mean  : float
    delta        : float
    pct_change   : float
    direction    : str      # "increase" | "decrease" | "no change"

class InterventionCellDetail(BaseModel):
    """
    Human-readable description of one B-matrix edge modification in the
    recommended intervention.
    """
    cell_key      : str   # e.g. "cell_r1_c0"
    row           : int
    col           : int
    node_from     : str   # e.g. "N2"   (row index + 1)
    node_to       : str   # e.g. "N1"   (col index + 1)
    sensor_from   : str   # sensor name for that row
    sensor_to     : str   # sensor name for that col
    original_value: float # value currently in the B-matrix
    recommended   : float # value the MC search found best

class RecommendedInterventionResponse(BaseModel):
    """
    Full operator-facing recommendation derived from the best Monte Carlo trial.

    How to read this response
    -------------------------
    1. `recommended_cells` – the exact B-matrix edge values the MC found best.
       A plant engineer would pass these directly into `/{gasifier}/fbf/custom`
       to validate against the physics gate before committing.

    2. `sensor_targets` – for every sensor, the *current* operating mean and the
       *simulated target* mean that results from applying those cell values.
       The operator should adjust each sensor's setpoint towards `target_mean`.

    3. `rul_improvement` – expected gain in Remaining Useful Life (days + %).

    4. `bucket` / `trial_number` – which MC trial this recommendation came from
       and which threshold bucket it belonged to.
    """
    gasifier_id         : str
    run_id              : str
    trial_number        : int
    bucket              : str           # "Conservative" | "Moderate" | "Aggressive"
    baseline_rul        : float
    simulated_rul       : float
    rul_improvement_days: float
    rul_improvement_pct : float
    recommended_cells   : list[InterventionCellDetail]
    sensor_targets      : list[SensorTarget]
    physics_detail      : dict
    operator_summary    : str           # plain-English action list for the operator


# ═══════════════════════════════════════════════════════════════════════════════
#  GENERAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _csv_files() -> list[Path]:
    return [Path(p) for p in glob.glob(os.path.join(GASIFIERS_DIR, "*.csv"))]

def _gasifier_name_from_file(path: Path) -> str:
    """Extract e.g. 'g5r19' from 'g5r19_combined_agg_7cols.csv'."""
    return path.stem.split("_")[0]

def _find_csv(gasifier: str) -> Path:
    for p in _csv_files():
        if _gasifier_name_from_file(p).lower() == gasifier.lower():
            return p
    raise HTTPException(status_code=404, detail=f"Gasifier '{gasifier}' not found in {GASIFIERS_DIR}")

def _load_df(gasifier: str) -> pd.DataFrame:
    path = _find_csv(gasifier)
    df = pd.read_csv(path)
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_candidates:
        df[ts_candidates[0]] = pd.to_datetime(df[ts_candidates[0]], errors="coerce")
        df = df.rename(columns={ts_candidates[0]: TIMESTAMP_COL})
    return df

def _split_phases(df: pd.DataFrame):
    """Split df into early (0–33%), mid (33–66%), late (66–100%)."""
    n = len(df)
    return df.iloc[: n // 3], df.iloc[n // 3 : 2 * n // 3], df.iloc[2 * n // 3 :]

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _log_alert(alert_type: str, gasifier_id: str, severity: str, message: str, details: dict) -> str:
    """Persist an alert JSON to the appropriate sub-folder. Returns the alert_id."""
    folder = RUL_ALERTS_DIR if alert_type == "rul" else HEALTH_ALERTS_DIR
    os.makedirs(folder, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    alert_id = f"{gasifier_id}_{alert_type}_{ts_str}"
    payload  = {
        "alert_id":    alert_id,
        "timestamp":   datetime.now().isoformat(),
        "gasifier_id": gasifier_id,
        "alert_type":  alert_type,
        "severity":    severity,
        "message":     message,
        "details":     details,
    }
    path = os.path.join(folder, f"{alert_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return alert_id


def _scan_rul_alerts(gasifier_id: str, current_rul: float) -> list[str]:
    """Evaluate RUL thresholds and log any triggered alerts. Returns list of alert_ids."""
    logged = []
    if current_rul < 7:
        logged.append(_log_alert(
            "rul", gasifier_id, "critical",
            f"RUL critically low: {current_rul:.1f} days (<7 d)",
            {"current_rul_days": current_rul, "threshold_days": 7},
        ))
    elif current_rul < 14:
        logged.append(_log_alert(
            "rul", gasifier_id, "critical",
            f"RUL below critical threshold: {current_rul:.1f} days (<14 d)",
            {"current_rul_days": current_rul, "threshold_days": 14},
        ))
    elif current_rul < 30:
        logged.append(_log_alert(
            "rul", gasifier_id, "warning",
            f"RUL approaching critical zone: {current_rul:.1f} days (<30 d)",
            {"current_rul_days": current_rul, "threshold_days": 30},
        ))
    return logged


def _scan_health_alerts(gasifier_id: str, score: float, label: str) -> list[str]:
    """Evaluate health score thresholds and log any triggered alerts."""
    logged = []
    if label == "Failing":
        logged.append(_log_alert(
            "health", gasifier_id, "critical",
            f"Health score in Failing zone: {score:.1f}/100",
            {"health_score": score, "health_label": label},
        ))
    elif label == "Moderate":
        logged.append(_log_alert(
            "health", gasifier_id, "warning",
            f"Health score in Moderate zone: {score:.1f}/100",
            {"health_score": score, "health_label": label},
        ))
    return logged


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH SCORE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_health_score(df: pd.DataFrame) -> tuple[float, str]:
    """
    Derive a 0–100 health score from sensor drift across phases.
    High late-phase drift → lower score.
    Returns (score, label) where label is 'Healthy' | 'Moderate' | 'Failing'.
    """
    sensor_cols = [c for c in SENSORS if c in df.columns]
    if not sensor_cols:
        return 50.0, "Moderate"

    early, mid, late = _split_phases(df)
    drift_scores = []
    for col in sensor_cols:
        e_mean = float(early[col].mean()) if len(early) else 0.0
        l_mean = float(late[col].mean())  if len(late)  else 0.0
        denom  = abs(e_mean) + 1e-9
        drift  = abs(l_mean - e_mean) / denom * 100
        drift_scores.append(drift)

    avg_drift = float(np.mean(drift_scores))
    score = max(0.0, min(100.0, 100.0 - avg_drift * 2.0))
    score = round(score, 2)

    if score >= 70:
        label = "Healthy"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Failing"

    return score, label


# ═══════════════════════════════════════════════════════════════════════════════
#  CRITICAL INDICATORS HELPER
# ═══════════════════════════════════════════════════════════════════════════════

# Sensor display labels used in the critical indicators table (UI-friendly names)
SENSOR_DISPLAY_LABELS: dict[str, str] = {
    "SlurryPDI":       "Slurry PDI",
    "OxygenPDI":       "Oxygen PDI",
    "SlurryFlow":      "Slurry Flow",
    "Temperature":     "Temperature (°C)",
    "SlurryPressure":  "Slurry Pressure (barg)",
    "OxygenFlow":      "Oxygen Flow",
    "OxygenSetpoint":  "Oxygen Setpoint",
}

def _compute_critical_indicators(df: pd.DataFrame) -> list[CriticalIndicator]:
    """
    For each sensor in SENSORS, compute:
      - current   : mean value in the late phase (last 33% of rows)
      - deviation : % drift of late-phase mean vs early-phase mean
      - trend_7d  : direction of change in the final 7-day window
      - status    : "normal" | "warning" | "critical"  (by deviation magnitude)

    Returns one CriticalIndicator per sensor that is present in df.

    Deviation thresholds
    --------------------
      |deviation| < 10 %  →  normal
      |deviation| 10–25 % →  warning
      |deviation| > 25 %  →  critical

    Trend (7-day window)
    --------------------
    The last 7 days of rows (approx. 10 080 1-min rows, capped to available
    rows) are split in half.  If the second-half mean exceeds the first-half
    mean by more than 1 % of the early-phase mean it is "rising"; if lower by
    more than 1 %, "falling"; otherwise "stable".
    """
    sensor_cols = [c for c in SENSORS if c in df.columns]
    early, _, late = _split_phases(df)
    indicators   = []

    # Approximate 7-day window in rows (1-min resampled data = 1440 rows/day)
    rows_per_day   = 1440
    trend_window   = min(len(df), 7 * rows_per_day)
    trend_df       = df.iloc[-trend_window:]
    mid_trend      = len(trend_df) // 2

    for col in sensor_cols:
        e_vals = early[col].dropna()
        l_vals = late[col].dropna()

        e_mean = float(e_vals.mean()) if len(e_vals) else 0.0
        l_mean = float(l_vals.mean()) if len(l_vals) else 0.0
        denom  = abs(e_mean) + 1e-9
        dev    = (l_mean - e_mean) / denom * 100          # signed %

        abs_dev = abs(dev)
        if abs_dev >= 25.0:
            status = "critical"
        elif abs_dev >= 10.0:
            status = "warning"
        else:
            status = "normal"

        # 7-day trend
        first_half  = trend_df.iloc[:mid_trend][col].dropna()
        second_half = trend_df.iloc[mid_trend:][col].dropna()
        if len(first_half) and len(second_half):
            fh_mean = float(first_half.mean())
            sh_mean = float(second_half.mean())
            noise   = abs(e_mean) * 0.01 + 1e-9
            if sh_mean - fh_mean > noise:
                trend = "rising"
            elif fh_mean - sh_mean > noise:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        indicators.append(CriticalIndicator(
            parameter = SENSOR_DISPLAY_LABELS.get(col, col),
            current   = round(l_mean, 4),
            deviation = round(dev, 2),
            trend_7d  = trend,
            status    = status,
        ))

    return indicators


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSICS GATE HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _hardcoded_physics_gate(
    transformed: np.ndarray,
    sensor_bounds: Optional[dict],
) -> tuple[bool, dict]:
    """
    Checks that the mean of each simulated sensor column falls within
    the caller-supplied [lower, upper] bounds.
    If sensor_bounds is None or empty, every sensor passes automatically.
    Returns (passed, detail_dict).
    """
    detail: dict = {}
    if not sensor_bounds:
        return True, detail

    for i, sensor in enumerate(SENSORS):
        if sensor not in sensor_bounds:
            continue
        col_vals = transformed[:, i]
        col_mean = float(np.mean(col_vals))
        col_min  = float(np.min(col_vals))
        col_max  = float(np.max(col_vals))
        bound    = sensor_bounds[sensor]
        passed   = bound.lower <= col_mean <= bound.upper
        detail[sensor] = {
            "simulated_mean": round(col_mean, 4),
            "simulated_min":  round(col_min,  4),
            "simulated_max":  round(col_max,  4),
            "bound_lower":    bound.lower,
            "bound_upper":    bound.upper,
            "passed":         passed,
        }
        if not passed:
            return False, detail

    return True, detail


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION HELPERS  (feature engineering, inference, graphing)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Confidence scoring ────────────────────────────────────────────────────────
#
# Industry rationale
# ------------------
# Point predictions (a single RUL number) are never perfectly reliable.
# Operators need to know *how much* to trust the number before acting on it.
# This function produces a 0–1 confidence score by blending three independent
# signals that are all available at inference time — no retraining required.
#
# The three signals
# -----------------
#   1. Regime certainty  (weight 0.45)
#      Our model has two regimes: Model A (early life) and Model B (late life).
#      Confidence is highest when the gasifier is cleanly in one regime.
#      It drops in the transition zone where both models are partially active.
#
#   2. Data completeness  (weight 0.35)
#      What fraction of engineered feature columns have real values?
#      Sensor dropout or a very short CSV will leave NaN features, which
#      makes predictions less reliable.
#
#   3. Prediction stability  (weight 0.20)
#      How much does the RUL bounce around across the current data window?
#      A stable gasifier produces smooth, consistent predictions.
#      Wild swings signal sensor noise or operating instability.
#
# Label thresholds
# ----------------
#   score ≥ 0.75  →  "high"
#   score ≥ 0.50  →  "medium"
#   score < 0.50  →  "low"
#
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(
    df_feat: pd.DataFrame,
    bundle: dict,
    predictions: np.ndarray,
) -> dict:
    """
    Compute a 0–1 confidence score for the current RUL prediction.

    Parameters
    ----------
    df_feat     : Feature-engineered DataFrame (must contain 'elapsed_days').
    bundle      : Loaded model bundle (must contain 'transition_days', 'features_A').
    predictions : Raw (unsmoothed) per-row RUL predictions from _predict_rul().
                  Pass the raw array — smoothed predictions would hide real instability.

    Returns
    -------
    dict with keys:
        confidence_score  : float  0–1
        confidence_label  : str    "high" | "medium" | "low"
        _debug            : dict   per-signal sub-scores for diagnostics
    """
    elapsed              = df_feat["elapsed_days"].values
    estimated_transition = bundle.get("transition_days", 80.0)
    blend_width          = 5.0

    # Signal 1: Regime certainty
    # w_B is close to 0 (pure Model A) or 1 (pure Model B) at the extremes,
    # and 0.5 right in the blend zone. We want certainty = 1 at extremes and
    # 0 in the middle, so we measure distance from 0.5 and scale to [0, 1].
    w_B = expit((elapsed - estimated_transition) / blend_width)
    regime_certainty = float(np.mean(np.abs(w_B - 0.5) * 2))

    # Signal 2: Data completeness
    # Check what fraction of the model's feature columns actually have values.
    # If sensors dropped out or rolling windows haven't filled yet, this drops.
    feat_cols    = bundle["features_A"]
    completeness = float(df_feat[feat_cols].notna().mean().mean())

    # Signal 3: Prediction stability
    # A low coefficient of variation (std / mean) means the model is producing
    # consistent estimates — we treat that as high stability / high confidence.
    if len(predictions) > 1:
        rel_std   = float(np.std(predictions) / (np.mean(predictions) + 1e-6))
        stability = float(np.clip(1.0 - rel_std, 0.0, 1.0))
    else:
        stability = 0.6  # single-point prediction — moderate confidence by default

    # Weighted blend of the three signals
    score = round(
        0.45 * regime_certainty +
        0.35 * completeness     +
        0.20 * stability,
        3,
    )

    # Map numeric score to a human-readable label
    if score >= 0.75:
        label = "high"
    elif score >= 0.50:
        label = "medium"
    else:
        label = "low"

    return {
        "confidence_score": score,
        "confidence_label": label,
        # Sub-scores exposed for debugging / dashboard display
        "_debug": {
            "regime_certainty":     round(regime_certainty, 3),
            "data_completeness":    round(completeness,     3),
            "prediction_stability": round(stability,        3),
        },
    }

# ── Failure probability scoring ───────────────────────────────────────────────
#
# Industry rationale
# ------------------
# A single RUL number ("42 days left") doesn't tell the operator *how confident*
# to be about that number. This function answers: given the prediction AND its
# known uncertainty (from cross-validation), what is the probability of actual
# failure within 7, 14, or 30 days?
#
# Why lognormal?
# --------------
# RUL prediction errors are NOT symmetric. When a gasifier has 5 days left, the
# model can't predict negative RUL — there's a hard floor at zero. But it can
# overestimate remaining life by 30+ days. This creates a right-skewed error
# distribution, and the lognormal distribution is the standard fit for that shape
# in reliability engineering (referenced in IEC 61508, NASA CMAPSS benchmarks).
#
# How sigma is derived
# --------------------
# We use the cross-validation MAE and std from the model bundle — these represent
# real observed prediction error on held-out gasifier runs, making them an honest
# estimate of how wrong the model can be at inference time.
#
# Risk label thresholds (based on p_fail_30d)
# -------------------------------------------
#   p_fail_30d ≥ 0.70  →  "critical"   (act immediately)
#   p_fail_30d ≥ 0.40  →  "high"       (plan intervention soon)
#   p_fail_30d ≥ 0.15  →  "moderate"   (monitor closely)
#   p_fail_30d < 0.15  →  "low"        (operating normally)
#
# ─────────────────────────────────────────────────────────────────────────────

def _compute_failure_probability(
    current_rul: float,
    cv_mae: float,
    cv_std: float,
    horizons: list[int] = [7, 14, 30],
) -> dict:
    """
    Estimate P(failure within N days) using a lognormal model of RUL uncertainty.

    Parameters
    ----------
    current_rul : Smoothed RUL estimate at the most recent timestamp (days).
                  This comes from smooth[-1] in the prediction pipeline.
    cv_mae      : Mean absolute error from leave-one-run-out cross-validation.
                  Stored in bundle["cv_mean_mae"]. Represents typical prediction error.
    cv_std      : Standard deviation of MAE across CV folds.
                  Stored in bundle["cv_std_mae"]. Represents variability of that error.
    horizons    : Time windows (days) to compute failure probability for.
                  Default: [7, 14, 30] — matches our RUL alert thresholds.

    Returns
    -------
    dict with keys:
        p_fail_7d          : float  P(failure within  7 days)
        p_fail_14d         : float  P(failure within 14 days)
        p_fail_30d         : float  P(failure within 30 days)
        failure_risk_label : str    "critical" | "high" | "moderate" | "low"
    """
    from scipy.stats import norm

    # Edge case: RUL already at or below zero — failure is imminent
    if current_rul <= 0:
        return {f"p_fail_{h}d": 1.0 for h in horizons} | {"failure_risk_label": "critical"}

    # Derive sigma (spread of the lognormal) from cross-val error.
    # Relative error = how large the uncertainty is compared to the prediction.
    # Example: current_rul=42, cv_mae=8, cv_std=3 → relative_error = 11/42 ≈ 0.26
    # Clipped to [0.05, 1.5] to avoid degenerate distributions.
    relative_error = (cv_mae + cv_std) / max(current_rul, 1.0)
    sigma = float(np.clip(relative_error, 0.05, 1.5))

    # mu is the log of our point estimate (centre of the lognormal distribution)
    mu = float(np.log(current_rul))

    result = {}
    for h in horizons:
        if h <= 0:
            continue
        # P(true RUL ≤ h) is the CDF of a lognormal evaluated at h.
        # Equivalent to a standard normal CDF evaluated at (log(h) - mu) / sigma.
        p = float(norm.cdf(np.log(h), loc=mu, scale=sigma))
        result[f"p_fail_{h}d"] = round(float(np.clip(p, 0.0, 1.0)), 4)

    # Risk label is driven by the 30-day horizon — most actionable for scheduling
    p30 = result.get("p_fail_30d", 0.0)
    if p30 >= 0.70:
        label = "critical"
    elif p30 >= 0.40:
        label = "high"
    elif p30 >= 0.15:
        label = "moderate"
    else:
        label = "low"

    result["failure_risk_label"] = label
    return result



def _clean_run(df: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, int]:
    """Resample → ffill → glitch mask → elapsed_days."""
    sensors = bundle["sensors"]
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    df = df[sensors].resample("1min").mean().ffill(limit=5)
    df = df.reset_index()

    o2_median       = df["OxygenFlow"].median()
    glitch_mask     = df["OxygenFlow"] < (o2_median * 0.05)
    glitch_expanded = (
        glitch_mask.rolling(window=61, center=True, min_periods=1).max().astype(bool)
    )
    glitch_count = int(glitch_expanded.sum())
    df = df[~glitch_expanded].copy().reset_index(drop=True)

    df["elapsed_days"] = (
        df["timestamp"] - df["timestamp"].iloc[0]
    ).dt.total_seconds() / 86400

    return df, glitch_count


def _engineer_features(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Rolling features, dev z-score, o2_error, age_pct."""
    sensors      = bundle["sensors"]
    window_short = bundle["window_short"]   # 60
    window_long  = bundle["window_long"]    # 360

    g = df.copy().sort_values("timestamp").reset_index(drop=True)
    n   = len(g)
    mid = g.iloc[n // 4 : 3 * n // 4]
    stable_med = mid[sensors].median()
    stable_std = mid[sensors].std().replace(0, 1)

    for s in sensors:
        col = g[s]
        g[f"{s}_rmean60"]  = col.rolling(window_short, min_periods=10).mean()
        g[f"{s}_rstd60"]   = col.rolling(window_short, min_periods=10).std()
        g[f"{s}_roc60"]    = col.diff(window_short)
        g[f"{s}_rmean360"] = col.rolling(window_long,  min_periods=30).mean()
        g[f"{s}_rstd360"]  = col.rolling(window_long,  min_periods=30).std()
        g[f"{s}_dev"]      = (col - stable_med[s]) / stable_std[s]
        g[f"{s}_mean_div"] = g[f"{s}_rmean60"] - g[f"{s}_rmean360"]

    g["o2_error"]          = g["OxygenFlow"] - g["OxygenSetpoint"]
    g["o2_error_rmean60"]  = g["o2_error"].rolling(window_short, min_periods=10).mean()
    g["o2_error_rstd60"]   = g["o2_error"].rolling(window_short, min_periods=10).std()
    g["o2_error_rmean360"] = g["o2_error"].rolling(window_long,  min_periods=30).mean()

    g["rows_elapsed"]     = np.arange(len(g))
    g["rows_elapsed_log"] = np.log1p(g["rows_elapsed"])

    # The target campaign life provided by plant management
    CAMPAIGN_LIFE_DAYS = 120.0
    g["age_pct"]          = g["elapsed_days"] / CAMPAIGN_LIFE_DAYS # Changed from g['elapsed_days'].max() as it always shows current as 100% which is not correct.

    return g


def _predict_rul(df_feat: pd.DataFrame, bundle: dict) -> np.ndarray:
    """
    Dual-model inference mirroring the cheat-free CV in the notebook.
    Blend weight is a sigmoid over elapsed_days relative to transition_days.
    """
    model_A = bundle["model_A"]
    model_B = bundle["model_B"]
    feat_A  = bundle["features_A"]
    feat_B  = bundle["features_B"]

    raw_A = np.clip(model_A.predict(df_feat[feat_A]), 0, None)
    raw_B = np.clip(model_B.predict(df_feat[feat_B]), 0, None)

    estimated_transition = bundle.get("transition_days", 80.0)
    blend_width          = 5.0

    elapsed = df_feat["elapsed_days"].values
    w_B     = expit((elapsed - estimated_transition) / blend_width)

    return (1.0 - w_B) * raw_A + w_B * raw_B


def _smoothed(series: np.ndarray, window: int = 120) -> np.ndarray:
    return pd.Series(series).rolling(window=window, min_periods=1, center=True).mean().values


def _rul_at_pct(df: pd.DataFrame, predictions: np.ndarray, pct: float) -> Optional[float]:
    cutoff = df["elapsed_days"].max() * pct
    mask   = df["elapsed_days"] <= cutoff
    if mask.sum() == 0:
        return None
    idx = int(mask.values.nonzero()[0][-1])
    return round(float(predictions[idx]), 2)


def _build_graph(
    gasifier:    str,
    df:          pd.DataFrame,
    predictions: np.ndarray,
    bundle:      dict,
    total_days:  Optional[float],
    graphs_dir:  str,
) -> tuple[str, str, Optional[float]]:
    """4-panel chart: RUL curve, sensor trends, stats card, phase-avg bar."""
    os.makedirs(graphs_dir, exist_ok=True)

    rul_split   = bundle["rul_split"]
    elapsed     = df["elapsed_days"].values
    smooth_pred = _smoothed(predictions)
    current_rul = round(float(smooth_pred[-1]), 2)
    mae         = bundle.get("cv_mean_mae")

    actual_rul: Optional[np.ndarray] = None
    mae_on_run: Optional[float]      = None
    if total_days is not None:
        actual_rul = np.clip(total_days - elapsed, 0, None)
        mae_on_run = float(np.mean(np.abs(actual_rul - smooth_pred)))

    fig = plt.figure(figsize=(16, 15))
    fig.suptitle(f"RUL Prediction  —  {gasifier.upper()}", fontsize=16, fontweight="bold", y=0.985)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.32, height_ratios=[2.8, 1.4, 1.4])

    ax_rul  = fig.add_subplot(gs[0, :])
    ax_sen1 = fig.add_subplot(gs[1, 0])
    ax_sen2 = fig.add_subplot(gs[1, 1])
    ax_stat = fig.add_subplot(gs[2, 0])
    ax_bar  = fig.add_subplot(gs[2, 1])

    # Panel 1 – Actual vs Predicted RUL
    if actual_rul is not None:
        ax_rul.plot(elapsed, actual_rul, color="steelblue", linewidth=2.0,
                    linestyle="--", label="Actual RUL", zorder=4)
        ax_rul.fill_between(elapsed, actual_rul, smooth_pred,
                            alpha=0.13, color="tomato", label="Error region")
    ax_rul.plot(elapsed, smooth_pred, color="tomato", linewidth=2.5, label="Predicted RUL", zorder=5)
    ax_rul.axvline(elapsed[-1], color="gray", linestyle=":", linewidth=1.4, alpha=0.7, label="Current position")
    ax_rul.axhline(14, color="#E05C5C", linestyle="--", linewidth=1.2, alpha=0.7, label="Critical (<14 d)")
    ax_rul.axhline(30, color="#E0A050", linestyle="--", linewidth=1.2, alpha=0.7, label="Warning (<30 d)")
    ax_rul.set_xlabel("Elapsed Days", fontsize=11)
    ax_rul.set_ylabel("RUL (days)",   fontsize=11)
    ax_rul.set_title(f"Remaining Useful Life  |  Current: {current_rul:.1f} d", fontsize=13)
    ax_rul.legend(fontsize=9, loc="upper right")
    ax_rul.grid(True, alpha=0.3)

    # Panel 2 & 3 – Sensor trends
    sensors = bundle["sensors"]
    for ax, s_list in [(ax_sen1, sensors[:4]), (ax_sen2, sensors[4:])]:
        for s in s_list:
            if s in df.columns:
                ax.plot(elapsed, df[s].values, linewidth=0.9, label=s, alpha=0.85)
        ax.set_xlabel("Elapsed Days", fontsize=9)
        ax.set_ylabel("Sensor value", fontsize=9)
        ax.set_title("Sensor Trends", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Panel 4 – Stats card
    ax_stat.axis("off")
    info_lines = [
        f"Gasifier:   {gasifier.upper()}",
        f"Run days:   {elapsed[-1]:.1f} d",
        f"Known total: {total_days or 'N/A'} d",
        f"Current RUL: {current_rul:.1f} d",
        f"MAE on run:  {mae_on_run:.2f} d" if mae_on_run else "MAE on run: N/A",
        f"CV MAE:      {mae:.2f} d" if mae else "CV MAE: N/A",
    ]
    ax_stat.text(0.05, 0.95, "\n".join(info_lines), transform=ax_stat.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#F0F0F0", alpha=0.8))
    ax_stat.set_title("Summary", fontsize=11)

    # Panel 5 – Phase-avg bar
    n = len(elapsed)
    slices = {
        "Early (0–33%)":  (slice(0, n // 3), ),
        "Mid (33–66%)":   (slice(n // 3, 2 * n // 3), ),
        "Late (66–100%)": (slice(2 * n // 3, n), ),
    }
    phase_labels = list(slices.keys())
    pred_means   = [float(np.nanmean(smooth_pred[s[0]])) for s in slices.values()]
    x     = np.arange(len(phase_labels))
    width = 0.35

    if actual_rul is not None:
        act_means = [float(np.nanmean(actual_rul[s[0]])) for s in slices.values()]
        bars_act  = ax_bar.bar(x - width / 2, act_means, width, color="steelblue",
                               edgecolor="white", linewidth=1.1, label="Actual RUL")
        bars_pred = ax_bar.bar(x + width / 2, pred_means, width, color="tomato",
                               edgecolor="white", linewidth=1.1, label="Predicted RUL")
        for bar, val in list(zip(bars_act, act_means)) + list(zip(bars_pred, pred_means)):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(max(act_means), max(pred_means)) * 0.015,
                        f"{val:.0f}d", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color="#333")
        ax_bar.legend(fontsize=9)
        ax_bar.set_ylim(0, max(pred_means + act_means) * 1.20)
    else:
        bars_pred = ax_bar.bar(x, pred_means, width * 1.6,
                               color=["steelblue", "darkorange", "tomato"],
                               edgecolor="white", linewidth=1.1)
        for bar, val in zip(bars_pred, pred_means):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(pred_means) * 0.015,
                        f"{val:.0f}d", ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#333")
        ax_bar.set_ylim(0, max(pred_means) * 1.20)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(phase_labels, fontsize=9)
    ax_bar.set_ylabel("Mean RUL (days)", fontsize=10)
    ax_bar.set_title("Phase-Avg RUL: Actual vs Predicted", fontsize=11)
    ax_bar.grid(True, axis="y", alpha=0.3)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"rul_{gasifier.lower()}_{ts}.png"
    save_path = os.path.join(graphs_dir, filename)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    with open(save_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")

    return save_path, b64, mae_on_run


# ═══════════════════════════════════════════════════════════════════════════════
#  B-MATRIX HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_bmat_filename(name: str) -> dict:
    lam  = re.search(r"lam([\d.]+)", name)
    lmin = re.search(r"lmin([\d]+)", name)
    lmax = re.search(r"lmax([\d]+)", name)
    return {
        "lambda_val": lam.group(1)  if lam  else None,
        "lmin":       lmin.group(1) if lmin else None,
        "lmax":       lmax.group(1) if lmax else None,
    }


def _load_bmat_array(gasifier: str, bucket: str) -> tuple[np.ndarray, str]:
    """Load B-matrix Excel and return (7×7 array, filename)."""
    pattern = os.path.join(BMAT_DIR, f"{gasifier.lower()}_buc-{bucket.lower()}_BM_*.xlsx")
    matches = glob.glob(pattern)
    if not matches:
        all_xlsx = glob.glob(os.path.join(BMAT_DIR, "*.xlsx"))
        matches  = [
            p for p in all_xlsx
            if gasifier.lower() in Path(p).name.lower()
            and f"buc-{bucket.lower()}" in Path(p).name.lower()
        ]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No B-matrix found for gasifier='{gasifier}', bucket='{bucket}' in {BMAT_DIR}",
        )
    chosen = sorted(matches)[-1]
    fname  = Path(chosen).name
    try:
        raw = pd.read_excel(chosen, header=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read B-matrix: {exc}")
    B_matrix = raw.iloc[1:].reset_index(drop=True).values.astype(float)
    return B_matrix, fname


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_row(X: np.ndarray, B_orig: np.ndarray, B_sim: np.ndarray, iterations: int) -> np.ndarray:
    """Encode X under B_orig, then decode under B_sim with iterative refinement."""
    I         = np.eye(len(X))
    Y         = (I - B_orig.T) @ X
    X_current = np.linalg.inv(I - B_sim.T) @ Y
    for _ in range(iterations):
        X_current = B_sim.T @ X_current + Y
    return X_current


def _build_sim_plot(
    gasifier:           str,
    df_original:        pd.DataFrame,
    df_simulated:       pd.DataFrame,
    edge_modifications: list[EdgeModification],
    B_original:         np.ndarray,
) -> str:
    """Before/after stacked comparison plot (one panel per sensor). Returns base64 PNG."""
    import matplotlib.dates as mdates

    COLOR_BEFORE = "#3B6FA0"
    COLOR_AFTER  = "#E08214"
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    n_nodes = len(FEATURE_COLS)
    fig, axes = plt.subplots(n_nodes, 1, figsize=(14, 3.4 * n_nodes), sharex=True)

    has_ts = "timestamp" in df_original.columns
    x = df_original["timestamp"] if has_ts else np.arange(len(df_original))
    mod_str = ", ".join(
        f"N{e.row+1}→N{e.col+1}: {B_original[e.row, e.col]:.4f} → {e.value}"
        for e in edge_modifications
    )

    for i, col in enumerate(FEATURE_COLS):
        ax     = axes[i]
        before = df_original[col].values
        after  = df_simulated[col].values

        mean_before = float(np.nanmean(before))
        mean_after  = float(np.nanmean(after))
        pct_change  = (mean_after - mean_before) / mean_before * 100 if mean_before != 0 else float("nan")
        direction   = "increased" if mean_after > mean_before else "decreased" if mean_after < mean_before else "stayed flat"

        ax.plot(x, before, color=COLOR_BEFORE, linewidth=1.1, label="Before", alpha=0.9)
        ax.plot(x, after,  color=COLOR_AFTER,  linewidth=1.1, label="After",  alpha=0.9)
        ax.axhline(mean_before, color=COLOR_BEFORE, linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(mean_after,  color=COLOR_AFTER,  linestyle="--", linewidth=1, alpha=0.6)
        ax.set_title(f"Node {i+1} ({col})")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.text(0.01, -0.20,
                f"Avg {direction} by {abs(pct_change):.2f}%  ({mean_before:.3f} → {mean_after:.3f})",
                transform=ax.transAxes, fontsize=9, color="#444", ha="left", va="top")

    if has_ts:
        axes[-1].set_xlabel("Timestamp")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=20)
    else:
        axes[-1].set_xlabel("Row index")

    fig.suptitle(
        f"Before vs After Simulation — {gasifier.upper()} — All Nodes\n{mod_str}",
        fontsize=12, fontweight="bold", y=0.999,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=120, bbox_inches="tight", format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  MONTE CARLO HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _mc_predict_tail(slice_feat: pd.DataFrame, bundle: dict) -> float:
    """Predict RUL using the tail-averaging logic (fast path for Monte Carlo)."""
    feat_a = slice_feat.reindex(columns=bundle["features_A"], fill_value=0)
    feat_b = slice_feat.reindex(columns=bundle["features_B"], fill_value=0)

    est_a = float(np.clip(bundle["model_A"].predict(feat_a), 0, None).mean())
    est_b = float(np.clip(bundle["model_B"].predict(feat_b), 0, None).mean())

    blend_high, blend_low = 60.0, 30.0
    if est_a >= blend_high:
        w_b = 0.0
    elif est_a <= blend_low:
        w_b = 1.0
    else:
        w_b = 1.0 - (est_a - blend_low) / (blend_high - blend_low)

    return (1 - w_b) * est_a + w_b * est_b


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPER – compute live RUL + health from one CSV (with alert logging)
# ═══════════════════════════════════════════════════════════════════════════════

def _live_rul_and_health(
    file_path: Path,
    bundle: dict,
    log_alerts: bool = True,
) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Load a live CSV, run the full pipeline, return (rul, health_score, health_label).
    Automatically logs RUL and health alerts unless log_alerts=False.
    Returns (None, None, None) on any error.
    """
    try:
        df_raw = pd.read_csv(file_path)
        ts_candidates = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
        if ts_candidates:
            df_raw["timestamp"] = pd.to_datetime(df_raw[ts_candidates[0]], errors="coerce")

        df_clean, _ = _clean_run(df_raw, bundle)
        if len(df_clean) < 400:
            return None, None, None

        df_feat = _engineer_features(df_clean, bundle)
        df_feat = df_feat.dropna(subset=bundle["features_B"]).reset_index(drop=True)
        if len(df_feat) == 0:
            return None, None, None

        predictions = _predict_rul(df_feat, bundle)
        smooth      = _smoothed(predictions)
        current_rul = round(float(smooth[-1]), 2)

        health_score, health_label = _compute_health_score(df_clean)

        gasifier_id = file_path.stem.split("_")[0].lower()
        if log_alerts:
            _scan_rul_alerts(gasifier_id, current_rul)
            _scan_health_alerts(gasifier_id, health_score, health_label)

        return current_rul, health_score, health_label
    except Exception:
        return None, None, None


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 – GET /overview/gasifiers
#  Static metadata from Gasifiers/ folder
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/overview/gasifiers",
    response_model=GasifierListResponse,
    summary="List all gasifiers with metadata (from Gasifiers folder)",
    tags=["Overview"],
)
def list_gasifiers():
    """Scans the Gasifiers directory and returns metadata for every CSV found."""
    files = _csv_files()
    if not files:
        raise HTTPException(status_code=404, detail=f"No CSV files found in {GASIFIERS_DIR}")

    results = []
    for path in sorted(files):
        gid = _gasifier_name_from_file(path)
        try:
            df_full = pd.read_csv(path)
            n_rows  = len(df_full)
            cols_list = list(df_full.columns)

            ts_candidates = [c for c in df_full.columns if "time" in c.lower() or "date" in c.lower()]
            ts_min = ts_max = num_days = None
            if ts_candidates:
                ts_series = pd.to_datetime(df_full[ts_candidates[0]], errors="coerce").dropna()
                if not ts_series.empty:
                    ts_min   = str(ts_series.min())
                    ts_max   = str(ts_series.max())
                    num_days = round((ts_series.max() - ts_series.min()).total_seconds() / 86400, 2)

            results.append(GasifierMeta(
                gasifier_id=gid, filename=path.name,
                num_rows=n_rows, num_columns=len(cols_list),
                timestamp_min=ts_min, timestamp_max=ts_max,
                num_days=num_days, columns=cols_list,
            ))
        except Exception:
            results.append(GasifierMeta(
                gasifier_id=gid, filename=path.name,
                num_rows=-1, num_columns=-1,
                timestamp_min=None, timestamp_max=None,
                num_days=None, columns=[],
            ))

    return GasifierListResponse(count=len(results), gasifiers=results)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 – GET /assets
#  Live data: metadata + current RUL + health (with alert logging)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/assets",
    response_model=AssetsResponse,
    summary="All gasifiers with live RUL and health score",
    tags=["Assets"],
)
def list_assets():
    """
    Reads the Gasifiers folder for static metadata, then enriches each entry
    with live RUL and health score from the corresponding file in Live_data/.
    Automatically logs RUL and health alerts for every gasifier scanned.
    """
    files = _csv_files()
    if not files:
        raise HTTPException(status_code=404, detail=f"No CSV files found in {GASIFIERS_DIR}")

    bundle = None
    if os.path.exists(BUNDLE_PATH):
        try:
            bundle = joblib.load(BUNDLE_PATH)
        except Exception:
            bundle = None

    results = []
    for path in sorted(files):
        gid = _gasifier_name_from_file(path)
        try:
            df_full   = pd.read_csv(path)
            n_rows    = len(df_full)
            cols_list = list(df_full.columns)

            ts_candidates = [c for c in df_full.columns if "time" in c.lower() or "date" in c.lower()]
            ts_min = ts_max = num_days = None
            if ts_candidates:
                ts_series = pd.to_datetime(df_full[ts_candidates[0]], errors="coerce").dropna()
                if not ts_series.empty:
                    ts_min   = str(ts_series.min())
                    ts_max   = str(ts_series.max())
                    num_days = round((ts_series.max() - ts_series.min()).total_seconds() / 86400, 2)
        except Exception:
            n_rows = -1; cols_list = []; ts_min = ts_max = num_days = None

        current_rul = health_score = health_label = None
        if bundle and os.path.exists(LIVE_DATA_DIR):
            live_candidates = glob.glob(os.path.join(LIVE_DATA_DIR, f"{gid}*.csv"))
            if live_candidates:
                live_path = Path(sorted(live_candidates)[-1])
                current_rul, health_score, health_label = _live_rul_and_health(live_path, bundle, log_alerts=True)

        results.append(AssetInfo(
            gasifier_id=gid, filename=path.name,
            num_rows=n_rows, num_columns=len(cols_list),
            timestamp_min=ts_min, timestamp_max=ts_max,
            num_days=num_days, columns=cols_list,
            current_rul_days=current_rul,
            health_score=health_score,
            health_label=health_label,
        ))

    return AssetsResponse(count=len(results), assets=results)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 – GET /overview/plant
#  Plant-level KPIs from Live_data (with alert logging)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/overview/plant",
    response_model=PlantOverviewResponse,
    summary="Plant-level KPIs: asset counts, avg RUL, avg health",
    tags=["Overview"],
)
def plant_overview():
    """
    Scans Live_data/ for the latest CSV per gasifier, runs the full pipeline,
    and returns plant-level summary metrics. Automatically logs alerts.

    RUL status thresholds:
      Critical  : RUL < 14 days
      Risky     : 14 ≤ RUL < 30 days
      Healthy   : RUL ≥ 30 days
    """
    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(status_code=500, detail=f"Model bundle not found at {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    if not os.path.exists(LIVE_DATA_DIR):
        raise HTTPException(status_code=404, detail=f"Live data directory not found: {LIVE_DATA_DIR}")

    statuses: list[GasifierLiveStatus] = []
    for file_path in sorted(glob.glob(os.path.join(LIVE_DATA_DIR, "*.csv"))):
        target_path = Path(file_path)
        gid         = target_path.stem.split("_")[0].lower()
        rul, h_score, h_label = _live_rul_and_health(target_path, bundle, log_alerts=True)
        if rul is None:
            continue

        rul_status = "Critical" if rul < 14 else "Risky" if rul < 30 else "Healthy"

        statuses.append(GasifierLiveStatus(
            gasifier_id=gid,
            file_processed=target_path.name,
            current_rul_days=rul,
            health_score=h_score if h_score is not None else 50.0,
            health_label=h_label if h_label is not None else "Moderate",
            rul_status=rul_status,
        ))

    total          = len(statuses)
    healthy_count  = sum(1 for s in statuses if s.health_label == "Healthy")
    risky_count    = sum(1 for s in statuses if s.rul_status   == "Risky")
    critical_count = sum(1 for s in statuses if s.rul_status   == "Critical")
    avg_health     = round(float(np.mean([s.health_score     for s in statuses])), 2) if statuses else 0.0
    avg_rul        = round(float(np.mean([s.current_rul_days for s in statuses])), 2) if statuses else 0.0

    return PlantOverviewResponse(
        total_assets=total,
        healthy_count=healthy_count,
        risky_count=risky_count,
        critical_count=critical_count,
        avg_health_score=avg_health,
        avg_rul_days=avg_rul,
        gasifier_statuses=statuses,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 – GET /overview/health
#  Phase-wise health breakdown (with alert logging)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/overview/health",
    response_model=HealthReportResponse,
    summary="Phase-wise health report for a gasifier (includes critical indicators)",
    tags=["Overview"],
)
def health_report(
    gasifier: str = Query(..., description="Gasifier ID, e.g. g5r19", example="g5r19"),
):
    """
    Returns a three-phase health breakdown (early / mid / late) for the given gasifier.
    Automatically logs health alerts if the score is in the Moderate or Failing zone.

    **New: critical_indicators block**
    Each sensor now includes:
    - `current`   – observed mean in the most-recent (late) phase
    - `deviation` – % drift vs the early phase (positive = risen, negative = fallen)
    - `trend_7d`  – direction of travel over the last 7 days: "rising" | "falling" | "stable"
    - `status`    – "normal" | "warning" | "critical" based on deviation magnitude

    Deviation thresholds:
      < 10%  → normal
      10–25% → warning
      > 25%  → critical
    """
    df = _load_df(gasifier)
    early, mid, late = _split_phases(df)
    num_cols = _numeric_cols(df)

    ts_min = ts_max = num_days = None
    if TIMESTAMP_COL in df.columns:
        ts_series = df[TIMESTAMP_COL].dropna()
        if not ts_series.empty:
            ts_min   = str(ts_series.min())
            ts_max   = str(ts_series.max())
            num_days = round((ts_series.max() - ts_series.min()).total_seconds() / 86400, 2)

    health_stats = []
    for col in num_cols:
        def _stats(phase_df):
            s = phase_df[col].dropna()
            return PhaseStats(
                mean=round(float(s.mean()), 4) if len(s) else 0.0,
                std=round(float(s.std()),  4) if len(s) else 0.0,
            )
        e_s, m_s, l_s = _stats(early), _stats(mid), _stats(late)
        health_stats.append(ColumnHealthStats(
            column=col,
            early=e_s, mid=m_s, late=l_s,
            early_to_mid_drift=round((m_s.mean - e_s.mean) / (abs(e_s.mean) + 1e-9) * 100, 2),
            mid_to_late_drift =round((l_s.mean - m_s.mean) / (abs(m_s.mean) + 1e-9) * 100, 2),
        ))

    health_score, health_label = _compute_health_score(df)
    critical_indicators        = _compute_critical_indicators(df)
    _scan_health_alerts(gasifier.lower(), health_score, health_label)

    return HealthReportResponse(
        gasifier_id=gasifier.lower(),
        asset_name=f"Gasifier {gasifier.upper()}",
        total_rows=len(df),
        phase_sizes={"early": len(early), "mid": len(mid), "late": len(late)},
        timestamp_range={"min": ts_min, "max": ts_max},
        num_days=num_days,
        health_score=health_score,
        health_label=health_label,
        critical_indicators=critical_indicators,
        columns_health=health_stats,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 5 – GET /overview/collective-live-rul
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/overview/collective-live-rul",
    response_model=CollectiveRulResponse,
    summary="Calculate live RUL for all gasifiers in the Live_data directory",
    tags=["Overview"],
)
def collective_live_rul():
    """Scans Live_data/, runs the prediction pipeline for each CSV, returns a collective RUL overview."""
    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(status_code=500, detail=f"Model bundle not found at {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    if not os.path.exists(LIVE_DATA_DIR):
        raise HTTPException(status_code=404, detail=f"Live data directory not found: {LIVE_DATA_DIR}")

    results = []
    for file_path in glob.glob(os.path.join(LIVE_DATA_DIR, "*.csv")):
        target_path = Path(file_path)
        gasifier_id = target_path.stem.split("_")[0].lower()
        try:
            df_raw = pd.read_csv(target_path)
            ts_candidates = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
            if ts_candidates:
                df_raw["timestamp"] = pd.to_datetime(df_raw[ts_candidates[0]], errors="coerce")

            df_clean, _ = _clean_run(df_raw, bundle)
            if len(df_clean) < 400:
                continue

            df_feat = _engineer_features(df_clean, bundle)
            df_feat = df_feat.dropna(subset=bundle["features_B"]).reset_index(drop=True)
            if len(df_feat) == 0:
                continue

            predictions = _predict_rul(df_feat, bundle)
            smooth      = _smoothed(predictions)
            current_rul = round(float(smooth[-1]), 2)

            results.append(LiveRulSummary(
                gasifier_id=gasifier_id,
                file_processed=target_path.name,
                current_rul_days=current_rul,
                critical_status=(current_rul < 14.0),
            ))
        except Exception:
            continue

    return CollectiveRulResponse(total_processed=len(results), summary=results)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 6 – GET /alerts
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/alerts",
    response_model=AlertsResponse,
    summary="View all logged RUL and health alerts",
    tags=["Alerts"],
)
def get_alerts(
    gasifier: Optional[str] = Query(None, description="Filter by gasifier ID, e.g. g5r19"),
    limit:    int            = Query(50,   description="Max alerts per type", ge=1, le=500),
):
    """
    Returns all persisted RUL and health alerts (newest-first).
    Alerts are auto-created when any endpoint that runs live RUL or health inference is hit:
    /predict/rul, /overview/plant, /assets, /overview/health.
    """
    def _load_alerts(folder: str) -> list[AlertEntry]:
        if not os.path.exists(folder):
            return []
        files = sorted(glob.glob(os.path.join(folder, "*.json")), key=os.path.getmtime, reverse=True)
        if gasifier:
            files = [f for f in files if Path(f).name.startswith(gasifier.lower())]
        entries = []
        for fp in files[:limit]:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                entries.append(AlertEntry(**data))
            except Exception:
                continue
        return entries

    rul_alerts    = _load_alerts(RUL_ALERTS_DIR)
    health_alerts = _load_alerts(HEALTH_ALERTS_DIR)

    return AlertsResponse(
        total_rul_alerts=len(rul_alerts),
        total_health_alerts=len(health_alerts),
        rul_alerts=rul_alerts,
        health_alerts=health_alerts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 7 – GET /rootcause/fetch-bmat
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/rootcause/fetch-bmat",
    response_model=BMatResponse,
    summary="Fetch B-matrix for a gasifier and phase bucket",
    tags=["Root Cause"],
)
def fetch_bmat(
    gasifier:  str   = Query(..., description="Gasifier ID, e.g. g1r20", example="g1r20"),
    bucket:    str   = Query(..., description="Phase bucket: early | mid | late", example="early"),
    threshold: float = Query(0.0, description="Min absolute weight for feedback loop detection", example=0.1),
):
    """Returns the B-matrix as a JSON matrix with feedback-loop detection."""
    if bucket.lower() not in BUCKET_CHOICES:
        raise HTTPException(status_code=400, detail=f"bucket must be one of {BUCKET_CHOICES}")

    pattern = os.path.join(BMAT_DIR, f"{gasifier.lower()}_buc-{bucket.lower()}_BM_*.xlsx")
    matches = glob.glob(pattern)
    if not matches:
        all_xlsx = glob.glob(os.path.join(BMAT_DIR, "*.xlsx"))
        matches  = [
            p for p in all_xlsx
            if gasifier.lower() in Path(p).name.lower()
            and f"buc-{bucket.lower()}" in Path(p).name.lower()
        ]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No B-matrix found for gasifier='{gasifier}', bucket='{bucket}' in {BMAT_DIR}",
        )

    chosen = sorted(matches)[-1]
    fname  = Path(chosen).name
    meta   = _parse_bmat_filename(fname)

    try:
        raw = pd.read_excel(chosen, header=None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read B-matrix: {exc}")

    col_labels  = [str(v) for v in raw.iloc[0].tolist()]
    value_block = raw.iloc[1:].reset_index(drop=True)
    matrix = [
        [
            None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 6)
            for v in row
        ]
        for _, row in value_block.iterrows()
    ]

    n_rows, n_cols = len(matrix), len(col_labels)
    feedback_loops = []
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            val_ij, val_ji = matrix[i][j], matrix[j][i]
            if (val_ij is not None and abs(val_ij) >= threshold
                    and val_ji is not None and abs(val_ji) >= threshold):
                feedback_loops.append({
                    "node_a": col_labels[i], "node_b": col_labels[j],
                    "weight_a_to_b": val_ij, "weight_b_to_a": val_ji,
                })

    # ── Top root cause connections ────────────────────────────────────────────
    # HARDCODED for now: we always surface the Node 1↔Node 2 and Node 2↔Node 5
    # connections as the primary root cause pathways. These two pairs were
    # identified by domain experts as the dominant influence channels in the
    # gasifier B-matrix (SlurryPDI↔OxygenPDI and OxygenPDI↔SlurryPressure).
    #
    # TODO: replace with dynamic causal ranking (e.g. by absolute edge weight
    # or by contribution to feedback loop strength) once that logic is validated.
    #
    # Node index mapping (0-based): N1=idx0, N2=idx1, N5=idx4
    # Pairs to highlight: (0,1) = N1↔N2  and  (1,4) = N2↔N5
    HARDCODED_ROOT_CAUSE_PAIRS = [
        (0, 1),   # Node 1 ↔ Node 2
        (1, 4),   # Node 2 ↔ Node 5
    ]

    top_root_causes: list[dict] = []
    for (r_idx, c_idx) in HARDCODED_ROOT_CAUSE_PAIRS:
        # Guard against matrices smaller than expected (shouldn't happen, but safe)
        if r_idx >= n_rows or c_idx >= n_cols:
            continue

        node_from   = col_labels[r_idx] if r_idx < len(col_labels) else f"N{r_idx + 1}"
        node_to     = col_labels[c_idx] if c_idx < len(col_labels) else f"N{c_idx + 1}"
        sensor_from = SENSORS[r_idx]    if r_idx < len(SENSORS)    else node_from
        sensor_to   = SENSORS[c_idx]    if c_idx < len(SENSORS)    else node_to
        weight      = matrix[r_idx][c_idx]  # directional weight: node_from → node_to

        top_root_causes.append(RootCauseConnection(
            node_from   = node_from,
            node_to     = node_to,
            sensor_from = sensor_from,
            sensor_to   = sensor_to,
            weight      = weight,
            description = (
                f"{sensor_from} directly influences {sensor_to} "
                f"(B-matrix edge {node_from}→{node_to}, weight={weight})."
            ),
        ))

    return BMatResponse(
        gasifier_id=gasifier.lower(),
        bucket=bucket.lower(),
        filename=fname,
        shape={"rows": n_rows, "cols": n_cols},
        lambda_val=meta.get("lambda_val"),
        lmin=meta.get("lmin"),
        lmax=meta.get("lmax"),
        matrix=matrix,
        row_labels=[],
        col_labels=col_labels,
        total_feedback_loops=len(feedback_loops),
        feedback_loops=feedback_loops,
        top_root_causes=top_root_causes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 8 – POST /predict/rul
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/predict/rul",
    response_model=CurrentRULResponse,
    summary="Predict the current RUL from a specific CSV file path",
    tags=["Prediction"],
)
def predict_rul(req: RULPredictionRequest):
    """
    Reads a specific CSV via absolute path, runs the full FE pipeline,
    and returns the current predicted RUL. Automatically logs RUL alerts.
    """
    target_path = Path(req.file_path)
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    gasifier_id = target_path.stem.split("_")[0].lower()

    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(status_code=500, detail=f"Model bundle not found at {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    try:
        df_raw = pd.read_csv(target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

    ts_candidates = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
    if not ts_candidates:
        raise HTTPException(status_code=422, detail="No timestamp column found.")
    df_raw["timestamp"] = pd.to_datetime(df_raw[ts_candidates[0]], errors="coerce")

    missing = [s for s in SENSORS if s not in df_raw.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing sensor columns: {missing}")

    df_clean, _ = _clean_run(df_raw, bundle)
    if len(df_clean) < 400:
        raise HTTPException(status_code=422, detail=f"Only {len(df_clean)} clean rows after glitch masking (need ≥400).")

    df_feat = _engineer_features(df_clean, bundle)
    df_feat = df_feat.dropna(subset=bundle["features_B"]).reset_index(drop=True)
    if len(df_feat) == 0:
        raise HTTPException(status_code=422, detail="All rows dropped after feature engineering (too many NaNs).")

    predictions = _predict_rul(df_feat, bundle)
    smooth      = _smoothed(predictions)
    current_rul = round(float(smooth[-1]), 2)

    # NEW — pass raw predictions (not smoothed) to confidence, current_rul to failure prob
    confidence   = _compute_confidence(df_feat, bundle, predictions)
    failure_prob = _compute_failure_probability(
        current_rul=current_rul,
        cv_mae=bundle.get("cv_mean_mae", 8.0),
        cv_std=bundle.get("cv_std_mae", 3.0),
    )

    _scan_rul_alerts(gasifier_id, current_rul)

    print(f"elapsed_days max: {df_feat['elapsed_days'].max():.2f}")
    print(f"total rows after FE: {len(df_feat)}")
    print(f"last 5 predictions: {smooth[-5:]}")

    return CurrentRULResponse(
        gasifier_id=gasifier_id,
        file_processed=target_path.name,
        current_rul_days=current_rul,
        # Confidence: how reliable is this prediction right now
        confidence_score=confidence["confidence_score"],
        confidence_label=confidence["confidence_label"],
        # Failure probability: P(failure within N days) given current RUL + model error
        p_fail_7d=failure_prob["p_fail_7d"],
        p_fail_14d=failure_prob["p_fail_14d"],
        p_fail_30d=failure_prob["p_fail_30d"],
        failure_risk_label=failure_prob["failure_risk_label"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 9 – POST /simulation
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/simulation",
    response_model=SimulationResponse,
    summary="Simulate sensor behaviour under a modified B-matrix",
    tags=["Simulation"],
)
def run_simulation(req: SimulationRequest):
    """
    Reads a CSV via absolute path, loads its B-matrix, applies the supplied edge
    modifications, iteratively refines sensor states, and returns a before/after
    RUL comparison plus a per-sensor comparison plot.
    """
    target_path = Path(req.file_path)
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    gasifier = target_path.stem.split("_")[0].lower()
    bucket   = req.bucket.lower()
    if bucket not in BUCKET_CHOICES:
        raise HTTPException(status_code=400, detail=f"bucket must be one of {BUCKET_CHOICES}")

    try:
        df_raw = pd.read_csv(target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read CSV: {e}")

    ts_candidates = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_candidates:
        df_raw[ts_candidates[0]] = pd.to_datetime(df_raw[ts_candidates[0]], errors="coerce")
        if ts_candidates[0] != "timestamp":
            df_raw = df_raw.rename(columns={ts_candidates[0]: "timestamp"})

    missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV is missing sensor columns: {missing}")

    B_original, bmat_fname = _load_bmat_array(gasifier, bucket)
    expected_size = len(FEATURE_COLS)
    if B_original.shape != (expected_size, expected_size):
        raise HTTPException(
            status_code=422,
            detail=f"B-matrix shape {B_original.shape} ≠ expected ({expected_size}×{expected_size}).",
        )

    B_simulation = B_original.copy()
    applied_mods = []
    for edge in req.edge_modifications:
        if not (0 <= edge.row < expected_size and 0 <= edge.col < expected_size):
            raise HTTPException(
                status_code=400,
                detail=f"Edge ({edge.row}, {edge.col}) out of range for {expected_size}×{expected_size} matrix.",
            )
        original_val = float(B_original[edge.row, edge.col])
        B_simulation[edge.row, edge.col] = edge.value
        applied_mods.append({
            "row": edge.row, "col": edge.col,
            "node_from": f"N{edge.row+1}", "node_to": f"N{edge.col+1}",
            "original_value": round(original_val, 6), "new_value": edge.value,
        })

    feature_matrix = df_raw[FEATURE_COLS].values.astype(float)
    col_means = np.nanmean(feature_matrix, axis=0)
    nan_mask  = np.isnan(feature_matrix)
    feature_matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    transformed = np.apply_along_axis(
        _simulate_row, axis=1, arr=feature_matrix,
        B_orig=B_original, B_sim=B_simulation, iterations=SIM_ITERATIONS,
    )

    df_out = df_raw.copy()
    for i, col in enumerate(FEATURE_COLS):
        df_out[col] = transformed[:, i]
    out_cols = (["timestamp"] if "timestamp" in df_out.columns else []) + FEATURE_COLS
    df_out   = df_out[out_cols]

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{gasifier}_{bucket}_simulated_{ts_str}.csv"
    out_path = os.path.join(SIM_OUTPUT_DIR, out_name)
    df_out.to_csv(out_path, index=False)

    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(status_code=500, detail=f"Model bundle not found at {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    def _compute_rul_at_end(df_input: pd.DataFrame) -> tuple[float, float]:
        df_c, _ = _clean_run(df_input, bundle)
        df_f    = _engineer_features(df_c, bundle)
        df_f    = df_f.dropna(subset=bundle["features_B"]).reset_index(drop=True)
        preds   = _predict_rul(df_f, bundle)
        smooth  = _smoothed(preds)
        return round(float(smooth[-1]), 2), float(df_f["elapsed_days"].max())

    df_orig_for_rul = df_raw.copy()
    if "timestamp" not in df_orig_for_rul.columns:
        ts_cands = [c for c in df_orig_for_rul.columns if "time" in c.lower() or "date" in c.lower()]
        if ts_cands:
            df_orig_for_rul["timestamp"] = pd.to_datetime(df_orig_for_rul[ts_cands[0]], errors="coerce")

    rul_before, elapsed_end = _compute_rul_at_end(df_orig_for_rul)

    df_sim_for_rul = df_out.copy()
    if "timestamp" not in df_sim_for_rul.columns and "timestamp" in df_raw.columns:
        df_sim_for_rul["timestamp"] = df_raw["timestamp"]
    rul_after, _ = _compute_rul_at_end(df_sim_for_rul)

    rul_delta  = round(rul_after - rul_before, 4)
    pct_improv = round((rul_delta / rul_before * 100) if rul_before != 0 else 0.0, 4)

    rul_comp = RULComparison(
        elapsed_total_days=round(elapsed_end, 4),
        rul_before=round(rul_before, 4),
        rul_after=round(rul_after, 4),
        rul_delta=rul_delta,
        improved=(rul_after > rul_before),
        pct_improvement=pct_improv,
    )

    sensor_stats = []
    for i, col in enumerate(FEATURE_COLS):
        mb  = float(np.nanmean(feature_matrix[:, i]))
        ma  = float(np.nanmean(transformed[:, i]))
        pct = (ma - mb) / mb * 100 if mb != 0 else float("nan")
        sensor_stats.append(SensorSimStats(
            sensor=col, node=f"N{i+1}",
            mean_before=round(mb, 4), mean_after=round(ma, 4),
            pct_change=round(pct, 4),
            direction="increased" if ma > mb else "decreased" if ma < mb else "no change",
        ))

    df_plot_before = df_raw[(["timestamp"] if "timestamp" in df_raw.columns else []) + FEATURE_COLS].copy()
    plot_b64 = _build_sim_plot(gasifier, df_plot_before, df_out, req.edge_modifications, B_original)

    os.makedirs(SIM_META_DIR, exist_ok=True)
    meta_name = f"{gasifier}_{bucket}_meta_{ts_str}.json"
    meta_path = os.path.join(SIM_META_DIR, meta_name)
    metadata  = {
        "simulation_id":      meta_name.replace(".json", ""),
        "timestamp":          datetime.now().isoformat(),
        "file_processed":     target_path.name,
        "gasifier_id":        gasifier,
        "bucket":             bucket,
        "bmat_file":          bmat_fname,
        "total_rows":         len(df_raw),
        "output_csv_path":    out_path,
        "edge_modifications": applied_mods,
        "rul_comparison": {
            "elapsed_total_days": rul_comp.elapsed_total_days,
            "rul_before":         rul_comp.rul_before,
            "rul_after":          rul_comp.rul_after,
            "rul_delta":          rul_comp.rul_delta,
            "improved":           rul_comp.improved,
            "pct_improvement":    rul_comp.pct_improvement,
        },
        "sensor_stats": [s.model_dump() for s in sensor_stats],
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    return SimulationResponse(
        gasifier_id=gasifier, bucket=bucket, bmat_file=bmat_fname,
        total_rows=len(df_raw), edge_modifications=applied_mods,
        output_csv_path=out_path, metadata_path=meta_path,
        rul_comparison=rul_comp, sensor_stats=sensor_stats, plot_base64=plot_b64,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 10 – POST /{gasifier}/fbf/run-montecarlo
#
#  Physics-gated MC search. Every physics-passing success case is appended to a
#  per-run CSV so results are easy to inspect and rank later.
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/{gasifier}/fbf/run-montecarlo",
    response_model=FBFMonteCarloResponse,
    summary="FBF Monte Carlo: threshold-bucketed search, saves success-case CSV",
    tags=["FBF"],
)
def fbf_run_montecarlo(gasifier: str, req: FBFMonteCarloRequest):
    """
    Runs a Monte Carlo search over specified B-Matrix cells.

    **Physics gate** uses caller-supplied per-sensor hardcoded bounds (supply
    `sensor_bounds` with `lower`/`upper` per sensor; omit a sensor to skip its check).

    **Threshold buckets in response**
    - Conservative : 5% ≤ improvement < 10%
    - Moderate     : 10% ≤ improvement < 20%
    - Aggressive   : improvement ≥ 20%

    **Saved outputs** (in fbf_mc_results/)
    - `{gasifier}_fbf_mc_{ts}.json` – full run summary
    - `success_cases/{gasifier}_fbf_mc_{ts}_success.csv` – one row per physics-passing
      success case with cell values, simulated RUL, and % improvement
    """
    t_start  = time.time()
    gasifier = gasifier.lower()

    target_path = Path(req.file_path)
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(status_code=500, detail=f"Bundle missing: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    try:
        df_raw = pd.read_csv(target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV read error: {e}")

    ts_candidates = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_candidates:
        df_raw["timestamp"] = pd.to_datetime(df_raw[ts_candidates[0]], errors="coerce")

    df_clean, _ = _clean_run(df_raw, bundle)
    df_feat     = _engineer_features(df_clean, bundle)
    df_feat     = df_feat.dropna(subset=bundle["features_B"]).reset_index(drop=True)
    if len(df_feat) == 0:
        raise HTTPException(status_code=422, detail="Data dropped entirely due to NaNs.")

    ref_end    = len(df_feat)
    ref_start  = ref_end - req.ref_rows
    pred_start = ref_end - req.window_rows
    if ref_start < 0:
        raise HTTPException(
            status_code=422,
            detail=f"Not enough clean rows for a {req.ref_rows}-row reference window (found {ref_end}).",
        )

    ref_sensors    = df_feat.iloc[ref_start:ref_end][SENSORS].values.astype(float)
    baseline_slice = df_feat.iloc[pred_start:ref_end].copy()
    baseline_rul   = _mc_predict_tail(baseline_slice, bundle)
    B_ORIGINAL, _  = _load_bmat_array(gasifier, req.bucket)

    mid_chunk  = df_feat.iloc[len(df_feat) // 4 : 3 * len(df_feat) // 4]
    stable_med = mid_chunk[SENSORS].median()
    stable_std = mid_chunk[SENSORS].std().replace(0, 1)

    # ── Counters & trackers ───────────────────────────────────────────────────
    counters = {"total": 0, "physics_passed": 0}

    BUCKET_DEFS = [
        (5.0,  10.0,          "Conservative"),
        (10.0, 20.0,          "Moderate"),
        (20.0, float("inf"),  "Aggressive"),
    ]
    bucket_best:   dict[str, dict] = {b[2]: {"pct": -np.inf, "trial": None} for b in BUCKET_DEFS}
    bucket_counts: dict[str, int]  = {b[2]: 0 for b in BUCKET_DEFS}

    best_overall  = {"pct": -np.inf, "days": -np.inf, "rul_sim": 0.0, "trial_n": 0, "vals": [], "phys": {}}
    success_total = 0

    # Collect all success cases for CSV export
    success_records: list[dict] = []

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    while counters["total"] < req.max_trials and success_total < req.n_success:
        counters["total"] += 1

        shared_vals = np.random.uniform(
            req.perturb_range[0], req.perturb_range[1], size=len(req.perturb_cells)
        )
        B_SIM = B_ORIGINAL.copy()
        for cell, val in zip(req.perturb_cells, shared_vals):
            B_SIM[cell.row, cell.col] = val

        try:
            transformed = np.apply_along_axis(
                _simulate_row, axis=1, arr=ref_sensors,
                B_orig=B_ORIGINAL, B_sim=B_SIM, iterations=10,
            )
        except np.linalg.LinAlgError:
            continue

        passed, phys_detail = _hardcoded_physics_gate(transformed, req.sensor_bounds)
        if not passed:
            continue
        counters["physics_passed"] += 1

        slice_after      = baseline_slice.copy()
        transformed_tail = transformed[-req.window_rows:]
        for i, s in enumerate(SENSORS):
            slice_after[s]          = transformed_tail[:, i]
            slice_after[f"{s}_dev"] = (slice_after[s] - stable_med[s]) / stable_std[s]

        rul_sim  = _mc_predict_tail(slice_after, bundle)
        days_imp = rul_sim - baseline_rul
        pct_imp  = (days_imp / baseline_rul * 100) if baseline_rul > 0 else 0.0

        # ── Per-sensor simulated means for this trial ─────────────────────────
        # We compute both baseline (current) mean and simulated mean per sensor
        # so the recommend-intervention endpoint can build before/after targets
        # without needing to re-run the simulation.
        sensor_sim_means  = {f"sim_mean_{s}":  round(float(np.nanmean(transformed[:, i])), 6) for i, s in enumerate(SENSORS)}
        sensor_base_means = {f"base_mean_{s}": round(float(np.nanmean(ref_sensors[:, i])),  6) for i, s in enumerate(SENSORS)}

        trial_snapshot = {
            "pct": pct_imp, "days": days_imp, "rul_sim": rul_sim,
            "trial_n": counters["total"], "vals": shared_vals.tolist(), "phys": phys_detail,
            "sensor_sim_means":  sensor_sim_means,
            "sensor_base_means": sensor_base_means,
        }

        if pct_imp > best_overall["pct"]:
            best_overall = trial_snapshot.copy()

        # Assign to threshold bucket
        for bmin, bmax, blabel in BUCKET_DEFS:
            if bmin <= pct_imp < bmax:
                bucket_counts[blabel] += 1
                success_total += 1
                if pct_imp > bucket_best[blabel]["pct"]:
                    bucket_best[blabel] = {"pct": pct_imp, "trial": trial_snapshot}

                # Build a flat row for the success-case CSV.
                # Columns: metadata | cell values | sensor baseline means | sensor sim means
                # This makes the CSV self-contained for the recommend-intervention endpoint.
                row: dict = {
                    "trial_number":      counters["total"],
                    "bucket":            blabel,
                    "simulated_rul":     round(rul_sim, 4),
                    "baseline_rul":      round(baseline_rul, 4),
                    "days_improvement":  round(days_imp, 4),
                    "pct_improvement":   round(pct_imp, 4),
                }
                # Cell values (B-matrix edges that were perturbed)
                for idx, cell in enumerate(req.perturb_cells):
                    row[f"cell_r{cell.row}_c{cell.col}"] = round(float(shared_vals[idx]), 6)
                # Sensor baseline means (current operating state at time of MC run)
                row.update(sensor_base_means)
                # Sensor simulated means (predicted operating state after intervention)
                row.update(sensor_sim_means)
                success_records.append(row)
                break

    # ── Build response objects ────────────────────────────────────────────────
    def _snap_to_best_worst(snap: Optional[dict]) -> Optional[MonteCarloBestWorst]:
        if snap is None:
            return None
        return MonteCarloBestWorst(
            pct_improvement  = round(snap["pct"],     2),
            days_improvement = round(snap["days"],    2),
            simulated_rul    = round(snap["rul_sim"], 2),
            trial_number     = snap["trial_n"],
            cell_values      = [round(v, 4) for v in snap["vals"]],
            physics_detail   = snap["phys"],
        )

    buckets_out = []
    for bmin, _, blabel in BUCKET_DEFS:
        best_trial_snap = bucket_best[blabel]["trial"]
        buckets_out.append(ThresholdBucket(
            label=blabel,
            min_pct=bmin,
            count=bucket_counts[blabel],
            best_trial=_snap_to_best_worst(best_trial_snap),
        ))

    best_overall_out = _snap_to_best_worst(best_overall if counters["physics_passed"] > 0 else None)
    best_risk_out    = best_overall_out   # max RUL improvement = max risk reduction
    infeasible       = counters["total"] - counters["physics_passed"]
    time_taken       = round(time.time() - t_start, 2)
    ts_str           = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    os.makedirs(FBF_MC_DIR, exist_ok=True)
    result_name = f"{gasifier}_fbf_mc_{ts_str}.json"
    result_path = os.path.join(FBF_MC_DIR, result_name)
    # Build best_overall enriched dict (includes sensor means so the
    # recommend-intervention endpoint can read targets without re-simulating)
    best_overall_json = None
    if best_overall_out:
        best_overall_json = best_overall_out.model_dump()
        best_overall_json["sensor_sim_means"]  = best_overall.get("sensor_sim_means",  {})
        best_overall_json["sensor_base_means"] = best_overall.get("sensor_base_means", {})

    mc_summary  = {
        "run_id":         result_name.replace(".json", ""),
        "timestamp":      datetime.now().isoformat(),
        "gasifier_id":    gasifier,
        "file_processed": target_path.name,
        "bucket":         req.bucket,
        "parameters": {
            "perturb_cells":  [{"row": c.row, "col": c.col} for c in req.perturb_cells],
            "perturb_range":  req.perturb_range,
            "max_trials":     req.max_trials,
            "n_success":      req.n_success,
            "ref_rows":       req.ref_rows,
            "window_rows":    req.window_rows,
            "sensor_bounds":  {k: {"lower": v.lower, "upper": v.upper} for k, v in (req.sensor_bounds or {}).items()},
        },
        "summary": {
            "baseline_rul":   round(baseline_rul, 2),
            "total_trials":   counters["total"],
            "physics_passed": counters["physics_passed"],
            "infeasible":     infeasible,
            "success_cases":  success_total,
            "time_taken_sec": time_taken,
        },
        "buckets":      [b.model_dump() for b in buckets_out],
        "best_overall": best_overall_json,
    }
    with open(result_path, "w", encoding="utf-8") as rf:
        json.dump(mc_summary, rf, indent=2, default=str)

    # ── Save success-case CSV ─────────────────────────────────────────────────
    os.makedirs(FBF_MC_CSV_DIR, exist_ok=True)
    csv_name = f"{gasifier}_fbf_mc_{ts_str}_success.csv"
    csv_path = os.path.join(FBF_MC_CSV_DIR, csv_name)
    if success_records:
        df_success = pd.DataFrame(success_records)
        # Sort best to worst by pct_improvement descending
        df_success = df_success.sort_values("pct_improvement", ascending=False).reset_index(drop=True)
        df_success.to_csv(csv_path, index=False)
    else:
        # Write empty CSV with headers so the file always exists
        cell_cols = [f"cell_r{c.row}_c{c.col}" for c in req.perturb_cells]
        pd.DataFrame(columns=[
            "trial_number", "bucket", "simulated_rul", "baseline_rul",
            "days_improvement", "pct_improvement", *cell_cols,
        ]).to_csv(csv_path, index=False)

    return FBFMonteCarloResponse(
        gasifier_id=gasifier,
        file_processed=target_path.name,
        baseline_rul=round(baseline_rul, 2),
        total_trials=counters["total"],
        physics_passed=counters["physics_passed"],
        infeasible=infeasible,
        success_cases=success_total,
        buckets=buckets_out,
        best_overall=best_overall_out,
        best_risk_reduction=best_risk_out,
        results_json_path=result_path,
        results_csv_path=csv_path,
        time_taken_sec=time_taken,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 11 – POST /{gasifier}/fbf/custom
#
#  Apply specific edge values, check physics, compute RUL before/after.
#  Saves simulated sensor CSV + metadata JSON.
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/{gasifier}/fbf/custom",
    response_model=FBFCustomResponse,
    summary="Custom B-matrix simulation: apply specific cell values and check physics",
    tags=["FBF"],
)
def fbf_custom(gasifier: str, req: FBFCustomRequest):
    """
    Applies the provided edge modifications to the B-matrix, simulates sensor data,
    and runs the physics check using caller-supplied per-sensor bounds.

    If the simulated data passes the physics gate, the predicted RUL (before and after)
    is returned along with the saved simulated CSV and metadata paths.
    """
    gasifier = gasifier.lower()

    target_path = Path(req.file_path)
    if not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    bucket = req.bucket.lower()
    if bucket not in BUCKET_CHOICES:
        raise HTTPException(status_code=400, detail=f"bucket must be one of {BUCKET_CHOICES}")

    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(status_code=500, detail=f"Bundle missing: {BUNDLE_PATH}")
    bundle = joblib.load(BUNDLE_PATH)

    try:
        df_raw = pd.read_csv(target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV read error: {e}")

    ts_candidates = [c for c in df_raw.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_candidates:
        df_raw[ts_candidates[0]] = pd.to_datetime(df_raw[ts_candidates[0]], errors="coerce")
        if ts_candidates[0] != "timestamp":
            df_raw = df_raw.rename(columns={ts_candidates[0]: "timestamp"})

    missing = [c for c in FEATURE_COLS if c not in df_raw.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"CSV missing sensor columns: {missing}")

    B_original, bmat_fname = _load_bmat_array(gasifier, bucket)
    expected_size = len(FEATURE_COLS)
    if B_original.shape != (expected_size, expected_size):
        raise HTTPException(
            status_code=422,
            detail=f"B-matrix shape {B_original.shape} ≠ expected ({expected_size}×{expected_size}).",
        )

    B_simulation = B_original.copy()
    applied_mods = []
    for edge in req.edge_modifications:
        if not (0 <= edge.row < expected_size and 0 <= edge.col < expected_size):
            raise HTTPException(
                status_code=400,
                detail=f"Edge ({edge.row}, {edge.col}) out of range for {expected_size}×{expected_size} matrix.",
            )
        B_simulation[edge.row, edge.col] = edge.value
        applied_mods.append({
            "row": edge.row, "col": edge.col,
            "node_from": f"N{edge.row+1}", "node_to": f"N{edge.col+1}",
            "original_value": round(float(B_original[edge.row, edge.col]), 6),
            "new_value": edge.value,
        })

    feature_matrix = df_raw[FEATURE_COLS].values.astype(float)
    col_means      = np.nanmean(feature_matrix, axis=0)
    nan_mask       = np.isnan(feature_matrix)
    feature_matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    try:
        transformed = np.apply_along_axis(
            _simulate_row, axis=1, arr=feature_matrix,
            B_orig=B_original, B_sim=B_simulation, iterations=SIM_ITERATIONS,
        )
    except np.linalg.LinAlgError as e:
        raise HTTPException(status_code=422, detail=f"Singular matrix during simulation: {e}")

    passed, phys_detail = _hardcoded_physics_gate(transformed, req.sensor_bounds)

    # Per-sensor stats (always computed regardless of physics gate result)
    sensor_stats = []
    for i, col in enumerate(FEATURE_COLS):
        mb  = float(np.nanmean(feature_matrix[:, i]))
        ma  = float(np.nanmean(transformed[:, i]))
        pct = (ma - mb) / mb * 100 if mb != 0 else float("nan")
        sensor_stats.append(SensorSimStats(
            sensor=col, node=f"N{i+1}",
            mean_before=round(mb, 4), mean_after=round(ma, 4),
            pct_change=round(pct, 4),
            direction="increased" if ma > mb else "decreased" if ma < mb else "no change",
        ))

    if not passed:
        return FBFCustomResponse(
            gasifier_id=gasifier,
            physics_passed=False,
            physics_detail=phys_detail,
            rul_before=None,
            rul_after=None,
            rul_delta=None,
            pct_improvement=None,
            sensor_stats=sensor_stats,
            output_csv_path=None,
            metadata_path=None,
            message="Simulation rejected: one or more sensors exceeded the physics bounds. RUL not computed.",
        )

    # ── Physics passed: save simulated CSV ───────────────────────────────────
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(FBF_CUSTOM_CSV, exist_ok=True)
    out_name = f"{gasifier}_{bucket}_custom_{ts_str}.csv"
    out_path = os.path.join(FBF_CUSTOM_CSV, out_name)

    df_out = df_raw.copy()
    for i, col in enumerate(FEATURE_COLS):
        df_out[col] = transformed[:, i]
    out_cols = (["timestamp"] if "timestamp" in df_out.columns else []) + FEATURE_COLS
    df_out[out_cols].to_csv(out_path, index=False)

    # ── RUL comparison ────────────────────────────────────────────────────────
    def _compute_rul_at_end(df_input: pd.DataFrame) -> float:
        df_c, _ = _clean_run(df_input, bundle)
        df_f    = _engineer_features(df_c, bundle)
        df_f    = df_f.dropna(subset=bundle["features_B"]).reset_index(drop=True)
        preds   = _predict_rul(df_f, bundle)
        smooth  = _smoothed(preds)
        return round(float(smooth[-1]), 2)

    df_orig_for_rul = df_raw.copy()
    if "timestamp" not in df_orig_for_rul.columns:
        ts_cands = [c for c in df_orig_for_rul.columns if "time" in c.lower() or "date" in c.lower()]
        if ts_cands:
            df_orig_for_rul["timestamp"] = pd.to_datetime(df_orig_for_rul[ts_cands[0]], errors="coerce")

    rul_before = _compute_rul_at_end(df_orig_for_rul)

    df_sim_for_rul = df_out[out_cols].copy()
    if "timestamp" not in df_sim_for_rul.columns and "timestamp" in df_raw.columns:
        df_sim_for_rul["timestamp"] = df_raw["timestamp"]
    rul_after  = _compute_rul_at_end(df_sim_for_rul)

    rul_delta  = round(rul_after - rul_before, 4)
    pct_improv = round((rul_delta / rul_before * 100) if rul_before != 0 else 0.0, 4)

    # ── Save metadata JSON ────────────────────────────────────────────────────
    os.makedirs(FBF_CUSTOM_META, exist_ok=True)
    meta_name = f"{gasifier}_{bucket}_custom_{ts_str}.json"
    meta_path = os.path.join(FBF_CUSTOM_META, meta_name)
    metadata  = {
        "run_id":             meta_name.replace(".json", ""),
        "timestamp":          datetime.now().isoformat(),
        "gasifier_id":        gasifier,
        "file_processed":     target_path.name,
        "bucket":             bucket,
        "bmat_file":          bmat_fname,
        "edge_modifications": applied_mods,
        "physics_passed":     True,
        "rul_before":         rul_before,
        "rul_after":          rul_after,
        "rul_delta":          rul_delta,
        "pct_improvement":    pct_improv,
        "output_csv_path":    out_path,
        "sensor_stats":       [s.model_dump() for s in sensor_stats],
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    return FBFCustomResponse(
        gasifier_id=gasifier,
        physics_passed=True,
        physics_detail=phys_detail,
        rul_before=rul_before,
        rul_after=rul_after,
        rul_delta=rul_delta,
        pct_improvement=pct_improv,
        sensor_stats=sensor_stats,
        output_csv_path=out_path,
        metadata_path=meta_path,
        message=f"Physics gate passed. RUL improved by {pct_improv:.2f}% ({rul_delta:+.2f} days).",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 12 – GET /{gasifier}/fbf/top-interventions
#
#  Returns the latest MC run's success-case CSV path (and summary metadata)
#  so the caller can directly open or download the ranked intervention list.
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/{gasifier}/fbf/top-interventions",
    response_model=TopInterventionsResponse,
    summary="Return the latest MC run's success-case CSV for a gasifier",
    tags=["FBF"],
)
def top_interventions(gasifier: str):
    """
    Looks up the most recent `/{gasifier}/fbf/run-montecarlo` result for the
    given gasifier and returns its success-case CSV path (sorted best→worst by
    % RUL improvement) along with key summary fields.

    The CSV columns are:
      trial_number | bucket | simulated_rul | baseline_rul |
      days_improvement | pct_improvement | cell_r{R}_c{C} …
    """
    gasifier = gasifier.lower()

    if not os.path.exists(FBF_MC_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"No MC results directory found at {FBF_MC_DIR}. Run /{gasifier}/fbf/run-montecarlo first.",
        )

    # Find the latest JSON summary for this gasifier
    json_files = sorted(
        glob.glob(os.path.join(FBF_MC_DIR, f"{gasifier}_fbf_mc_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not json_files:
        raise HTTPException(
            status_code=404,
            detail=f"No MC run found for gasifier '{gasifier}'. Run /{gasifier}/fbf/run-montecarlo first.",
        )

    latest_json = json_files[0]
    try:
        with open(latest_json, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read run summary: {e}")

    # Derive the matching CSV path (same timestamp stem)
    run_id   = summary.get("run_id", Path(latest_json).stem)
    csv_path = os.path.join(FBF_MC_CSV_DIR, f"{run_id}_success.csv")

    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"Success-case CSV not found at {csv_path}. The last run may have had zero successes.",
        )

    run_summary = summary.get("summary", {})

    return TopInterventionsResponse(
        gasifier_id=gasifier,
        run_id=run_id,
        timestamp=summary.get("timestamp", ""),
        baseline_rul=run_summary.get("baseline_rul", 0.0),
        success_cases=run_summary.get("success_cases", 0),
        results_csv_path=csv_path,
        results_json_path=latest_json,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT 13 – GET /{gasifier}/fbf/recommend-intervention
#
#  Reads the best MC trial from the latest run's success CSV / JSON, derives
#  the sensor before→after operating targets, and returns a full operator-facing
#  recommendation with plain-English guidance.
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/{gasifier}/fbf/recommend-intervention",
    response_model=RecommendedInterventionResponse,
    summary="Operator-ready intervention recommendation from the best MC trial",
    tags=["FBF"],
)
def recommend_intervention(
    gasifier: str,
    trial_rank: int = Query(
        1,
        ge=1,
        description=(
            "Which trial to recommend (1 = best overall by % improvement, "
            "2 = second best, etc.). Defaults to 1."
        ),
    ),
):
    """
    Derives a concrete, operator-facing recommendation from the latest
    `/{gasifier}/fbf/run-montecarlo` run.

    **What it returns**

    | Field | Description |
    |---|---|
    | `recommended_cells` | The exact B-matrix edge values to apply. Pass these directly into `/{gasifier}/fbf/custom` to validate before committing. |
    | `sensor_targets` | For **every** sensor: the current operating mean vs the target mean that results from the intervention. The `direction` field says whether the operator should increase or decrease that sensor's setpoint. |
    | `rul_improvement_days` / `_pct` | Expected RUL gain. |
    | `operator_summary` | Plain-English bullet list of actions. |

    **How sensor targets are computed**

    The sensor means are stored directly in the success-case CSV by
    `/{gasifier}/fbf/run-montecarlo` (columns `base_mean_*` and `sim_mean_*`),
    so **no re-simulation is needed** here – the values are read straight from
    the file, making this endpoint fast.

    **Example flow**
    ```
    POST /{gasifier}/fbf/run-montecarlo   →  runs MC search, saves results
    GET  /{gasifier}/fbf/recommend-intervention  →  get best recommendation
    POST /{gasifier}/fbf/custom           →  validate those cells + physics gate
    ```
    """
    gasifier = gasifier.lower()

    # ── 1. Find latest MC JSON summary ───────────────────────────────────────
    if not os.path.exists(FBF_MC_DIR):
        raise HTTPException(
            status_code=404,
            detail=f"No MC results directory at {FBF_MC_DIR}. Run /{gasifier}/fbf/run-montecarlo first.",
        )

    json_files = sorted(
        glob.glob(os.path.join(FBF_MC_DIR, f"{gasifier}_fbf_mc_*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not json_files:
        raise HTTPException(
            status_code=404,
            detail=f"No MC run found for '{gasifier}'. Run /{gasifier}/fbf/run-montecarlo first.",
        )

    with open(json_files[0], "r", encoding="utf-8") as fh:
        mc_json = json.load(fh)

    run_id      = mc_json.get("run_id", Path(json_files[0]).stem)
    baseline_rul = float(mc_json.get("summary", {}).get("baseline_rul", 0.0))
    perturb_cells_meta = mc_json.get("parameters", {}).get("perturb_cells", [])

    # ── 2. Load success-case CSV ──────────────────────────────────────────────
    csv_path = os.path.join(FBF_MC_CSV_DIR, f"{run_id}_success.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"Success-case CSV not found at {csv_path}. The last MC run may have had zero successes.",
        )

    df_success = pd.read_csv(csv_path)
    if df_success.empty:
        raise HTTPException(
            status_code=404,
            detail="The success-case CSV is empty – no physics-passing improvements were found in the last MC run.",
        )

    # Sort best → worst and pick the requested rank
    df_success = df_success.sort_values("pct_improvement", ascending=False).reset_index(drop=True)
    if trial_rank > len(df_success):
        raise HTTPException(
            status_code=404,
            detail=(
                f"trial_rank={trial_rank} requested but only {len(df_success)} success cases exist. "
                f"Use trial_rank ≤ {len(df_success)}."
            ),
        )

    best_row = df_success.iloc[trial_rank - 1]

    # ── 3. Extract cell columns ───────────────────────────────────────────────
    # Cell columns look like  cell_r1_c0, cell_r0_c1, …
    cell_cols = [c for c in df_success.columns if c.startswith("cell_r")]

    # Load the B-matrix to get original (unmodified) edge values
    bucket = mc_json.get("bucket", "late")
    try:
        B_original, _ = _load_bmat_array(gasifier, bucket)
    except HTTPException:
        B_original = None

    recommended_cells: list[InterventionCellDetail] = []
    for col_name in cell_cols:
        # Parse row / col from column name e.g. cell_r1_c0 → row=1, col=0
        parts = col_name.replace("cell_r", "").split("_c")
        if len(parts) != 2:
            continue
        try:
            r, c = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        orig_val = float(B_original[r, c]) if B_original is not None else float("nan")
        sensor_from = SENSORS[r] if r < len(SENSORS) else f"node_{r}"
        sensor_to   = SENSORS[c] if c < len(SENSORS) else f"node_{c}"

        recommended_cells.append(InterventionCellDetail(
            cell_key       = col_name,
            row            = r,
            col            = c,
            node_from      = f"N{r+1}",
            node_to        = f"N{c+1}",
            sensor_from    = sensor_from,
            sensor_to      = sensor_to,
            original_value = round(orig_val, 6),
            recommended    = round(float(best_row[col_name]), 6),
        ))

    # ── 4. Build sensor targets from stored means ─────────────────────────────
    # The MC loop stores base_mean_{sensor} and sim_mean_{sensor} in the CSV.
    # If those columns exist (new format), use them. Otherwise fall back to the
    # JSON best_overall block for backward compatibility.
    sensor_targets: list[SensorTarget] = []
    has_means_in_csv = any(c.startswith("sim_mean_") for c in df_success.columns)

    for i, sensor in enumerate(SENSORS):
        sim_col  = f"sim_mean_{sensor}"
        base_col = f"base_mean_{sensor}"

        if has_means_in_csv and sim_col in best_row.index and base_col in best_row.index:
            current_mean = float(best_row[base_col])
            target_mean  = float(best_row[sim_col])
        else:
            # Fallback: use JSON best_overall sensor means
            best_json    = mc_json.get("best_overall") or {}
            sim_means    = best_json.get("sensor_sim_means",  {})
            base_means   = best_json.get("sensor_base_means", {})
            current_mean = float(base_means.get(f"base_mean_{sensor}", 0.0))
            target_mean  = float(sim_means.get(f"sim_mean_{sensor}",   0.0))

        delta      = target_mean - current_mean
        pct_change = (delta / current_mean * 100) if current_mean != 0 else 0.0
        noise      = abs(current_mean) * 0.005 + 1e-9
        direction  = "increase" if delta > noise else "decrease" if delta < -noise else "no change"

        sensor_targets.append(SensorTarget(
            sensor       = sensor,
            node         = f"N{i+1}",
            current_mean = round(current_mean, 4),
            target_mean  = round(target_mean,  4),
            delta        = round(delta,         4),
            pct_change   = round(pct_change,    2),
            direction    = direction,
        ))

    # ── 5. Build plain-English operator summary ───────────────────────────────
    rul_days = float(best_row.get("days_improvement", 0.0))
    rul_pct  = float(best_row.get("pct_improvement",  0.0))
    bucket_label = str(best_row.get("bucket", "Unknown"))
    trial_num    = int(best_row.get("trial_number", 0))

    action_lines = []
    for st in sensor_targets:
        if st.direction == "no change":
            continue
        arrow = "▲ INCREASE" if st.direction == "increase" else "▼ DECREASE"
        action_lines.append(
            f"  • {arrow} {st.sensor}: {st.current_mean:.4g} → {st.target_mean:.4g}"
            f"  ({st.pct_change:+.1f}%)"
        )

    if action_lines:
        actions_str = "\n".join(action_lines)
    else:
        actions_str = "  • No significant sensor adjustments required."

    operator_summary = (
        f"RECOMMENDED INTERVENTION  |  Rank #{trial_rank}  |  Bucket: {bucket_label}\n"
        f"Expected RUL gain: +{rul_days:.1f} days  ({rul_pct:+.1f}%)\n"
        f"Baseline RUL: {baseline_rul:.1f} d  →  Simulated RUL: {float(best_row.get('simulated_rul', 0)):.1f} d\n\n"
        f"Sensor operating targets (adjust setpoints towards these values):\n"
        f"{actions_str}\n\n"
        f"Next step: pass `recommended_cells` into POST /{gasifier}/fbf/custom to\n"
        f"validate against the physics gate before committing to these changes."
    )

    return RecommendedInterventionResponse(
        gasifier_id          = gasifier,
        run_id               = run_id,
        trial_number         = trial_num,
        bucket               = bucket_label,
        baseline_rul         = round(baseline_rul, 2),
        simulated_rul        = round(float(best_row.get("simulated_rul", 0.0)), 2),
        rul_improvement_days = round(rul_days, 2),
        rul_improvement_pct  = round(rul_pct, 2),
        recommended_cells    = recommended_cells,
        sensor_targets       = sensor_targets,
        physics_detail       = {},   # stored in JSON but not critical here
        operator_summary     = operator_summary,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ROOT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Fix Before Fail – Gasifier Analytics API",
        "version": "2.1.0",
        "docs":    "/docs",
        "redoc":   "/redoc",
    }