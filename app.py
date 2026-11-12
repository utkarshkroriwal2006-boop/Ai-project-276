"""
POLAR STATION OPS — Remote Research Station Operations Management Platform
Run: streamlit run app.py

Stack: Streamlit, Python, Pandas, NumPy, Scikit-learn, SQLite, Plotly.
All operational records, simulations, forecasts, maintenance flags and chatbot
messages are persisted in a local SQLite database (polar_ops.db).
This is a decision-support prototype using simulated data, not a certified
safety-critical control system.
"""
from __future__ import annotations

import json
import re
import sqlite3
import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, precision_score
from sklearn.model_selection import train_test_split

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "polar_ops.db"
RNG_SEED = 42

st.set_page_config(
    page_title="Polar Station Ops",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DATABASE -----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            region TEXT NOT NULL,
            capacity INTEGER DEFAULT 30,
            season_start TEXT,
            season_end TEXT,
            comms_status TEXT DEFAULT 'Intermittent',
            last_sync TEXT
        );
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            category TEXT NOT NULL,
            unit TEXT NOT NULL,
            current_stock REAL NOT NULL,
            minimum_stock REAL NOT NULL,
            lead_days INTEGER NOT NULL,
            daily_usage REAL NOT NULL,
            criticality TEXT DEFAULT 'Medium',
            updated_at TEXT,
            UNIQUE(station_id, item),
            FOREIGN KEY(station_id) REFERENCES stations(id)
        );
        CREATE TABLE IF NOT EXISTS usage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            usage_date TEXT NOT NULL,
            quantity REAL NOT NULL,
            weather_temp REAL,
            wind_speed REAL,
            occupancy INTEGER,
            is_simulated INTEGER DEFAULT 1,
            FOREIGN KEY(station_id) REFERENCES stations(id)
        );
        CREATE TABLE IF NOT EXISTS equipment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id INTEGER NOT NULL,
            asset_name TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            hours_used REAL NOT NULL,
            vibration REAL NOT NULL,
            temperature REAL NOT NULL,
            last_service TEXT,
            service_interval_days INTEGER DEFAULT 90,
            condition_status TEXT DEFAULT 'Normal',
            FOREIGN KEY(station_id) REFERENCES stations(id)
        );
        CREATE TABLE IF NOT EXISTS maintenance_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id INTEGER NOT NULL,
            equipment_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            risk_score REAL NOT NULL,
            severity TEXT NOT NULL,
            reasons TEXT NOT NULL,
            predicted_failure INTEGER NOT NULL,
            acknowledged INTEGER DEFAULT 0,
            FOREIGN KEY(station_id) REFERENCES stations(id),
            FOREIGN KEY(equipment_id) REFERENCES equipment(id)
        );
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id INTEGER NOT NULL,
            item TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            model_name TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            predicted_daily_usage REAL NOT NULL,
            days_to_empty REAL,
            depletion_date TEXT,
            stockout_risk TEXT,
            recommended_order REAL,
            FOREIGN KEY(station_id) REFERENCES stations(id)
        );
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            station_id INTEGER NOT NULL,
            season_days INTEGER NOT NULL,
            model_mae REAL,
            baseline_mae REAL,
            stockouts_baseline INTEGER,
            stockouts_model INTEGER,
            stockouts_avoided INTEGER,
            flag_precision REAL,
            details_json TEXT,
            FOREIGN KEY(station_id) REFERENCES stations(id)
        );
        CREATE TABLE IF NOT EXISTS chatbot_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            intent TEXT
        );
        """)
        count = con.execute("SELECT COUNT(*) FROM stations").fetchone()[0]
        if count == 0:
            seed_database(con)

def seed_database(con):
    today = date.today()
    stations = [
        ("Aurora Base", "Antarctica · Coastal", 42, "Intermittent"),
        ("Frost Ridge", "Antarctica · Inland", 26, "Low bandwidth"),
    ]
    for name, region, cap, comms in stations:
        con.execute(
            "INSERT INTO stations(name,region,capacity,season_start,season_end,comms_status,last_sync) VALUES(?,?,?,?,?,?,?)",
            (name, region, cap, (today-timedelta(days=120)).isoformat(),
             (today+timedelta(days=80)).isoformat(), comms, datetime.now().isoformat(timespec="minutes"))
        )
    station_ids = [r["id"] for r in con.execute("SELECT id FROM stations").fetchall()]
    inventory_templates = [
        ("Diesel fuel", "Energy", "litres", 26000, 6000, 35, 390, "Critical"),
        ("Fresh water", "Water", "litres", 48000, 9000, 28, 620, "Critical"),
        ("Food boxes", "Food", "boxes", 9200, 1800, 42, 115, "Critical"),
        ("Medical kits", "Medical", "kits", 180, 35, 50, 0.7, "Critical"),
        ("Heating oil", "Energy", "litres", 11500, 2500, 35, 170, "High"),
        ("Battery packs", "Power", "units", 420, 90, 60, 2.1, "High"),
    ]
    rng = np.random.default_rng(RNG_SEED)
    for sid in station_ids:
        for item, cat, unit, stock, minimum, lead, usage, crit in inventory_templates:
            scale = 1.0 if sid == station_ids[0] else 0.62
            con.execute("""INSERT INTO inventory
                (station_id,item,category,unit,current_stock,minimum_stock,lead_days,daily_usage,criticality,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (sid,item,cat,unit,stock*scale,minimum*scale,lead,usage*scale,crit,datetime.now().isoformat(timespec="minutes")))
            # 120-day simulated season history with weather/occupancy effects.
            for d in range(120):
                dt = today - timedelta(days=119-d)
                temp = -28 - 13*np.sin(2*np.pi*d/120) + rng.normal(0,2.5)
                wind = max(2, 22 + 10*abs(np.cos(2*np.pi*d/37)) + rng.normal(0,3))
                occupancy = max(8, int((42 if sid == station_ids[0] else 26) + rng.normal(0,2)))
                weather_factor = 1.0 + max(0, (-temp-25))/100
                wind_factor = 1.0 + max(0, wind-25)/180
                item_factor = weather_factor if cat in ("Energy","Power") else (0.9+occupancy/100)
                q = max(0.05, usage*item_factor*wind_factor*rng.normal(1,0.09))
                con.execute("""INSERT INTO usage_history
                    (station_id,item,usage_date,quantity,weather_temp,wind_speed,occupancy,is_simulated)
                    VALUES(?,?,?,?,?,?,?,1)""",
                    (sid,item,dt.isoformat(),round(q,2),round(temp,1),round(wind,1),occupancy))
        equipment_templates = [
            ("Main diesel generator","Power",4420,4.7,94, today-timedelta(days=105),90),
            ("Backup generator","Power",1280,1.3,69, today-timedelta(days=35),90),
            ("Water desalination unit","Water",3890,3.8,47, today-timedelta(days=98),90),
            ("Heating circulation pump","Heating",4610,4.1,82, today-timedelta(days=112),90),
            ("Satellite communications dish","Communications",2900,0.9,-18, today-timedelta(days=25),180),
            ("Cold storage compressor","Refrigeration",3220,2.9,76, today-timedelta(days=70),90),
        ]
        for name, typ, hours, vib, temp, serviced, interval in equipment_templates:
            con.execute("""INSERT INTO equipment
                (station_id,asset_name,asset_type,hours_used,vibration,temperature,last_service,service_interval_days,condition_status)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (sid,name,typ,hours*(1 if sid == station_ids[0] else .65),
                 vib, temp, serviced.isoformat(), interval, "Needs inspection" if vib>3.5 else "Normal"))

# ----------------------------- DATA HELPERS -----------------------------
def read_df(query, params=()):
    with get_conn() as con:
        return pd.read_sql_query(query, con, params=params)

def get_stations():
    return read_df("SELECT * FROM stations ORDER BY name")

def selected_station_id(name):
    row = read_df("SELECT id FROM stations WHERE name=?", (name,))
    return int(row.iloc[0]["id"])

def log_chat(role, message, intent="general"):
    with get_conn() as con:
        con.execute("INSERT INTO chatbot_messages(created_at,role,message,intent) VALUES(?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), role, message, intent))

def fmt_num(x):
    return f"{x:,.1f}" if pd.notna(x) else "—"

# ----------------------------- FORECASTING -----------------------------
FEATURES = ["day_index", "temp", "wind", "occupancy", "lag_1", "lag_3", "lag_7", "rolling_7"]

def build_usage_features(hist):
    h = hist.sort_values("usage_date").copy()
    h["usage_date"] = pd.to_datetime(h["usage_date"])
    h["day_index"] = np.arange(len(h))
    h["lag_1"] = h["quantity"].shift(1)
    h["lag_3"] = h["quantity"].shift(3)
    h["lag_7"] = h["quantity"].shift(7)
    h["rolling_7"] = h["quantity"].shift(1).rolling(7, min_periods=1).mean()
    return h.dropna(subset=["lag_7"]).copy()

def forecast_item(station_id, item, horizon=30):
    hist = read_df("""SELECT usage_date,quantity,weather_temp AS temp,wind_speed AS wind,occupancy
                      FROM usage_history WHERE station_id=? AND item=? ORDER BY usage_date""",
                   (station_id,item))
    if len(hist) < 15:
        return None
    feat = build_usage_features(hist)
    if len(feat) < 8:
        return None
    X = feat[FEATURES].astype(float)
    y = feat["quantity"].astype(float)
    model = RandomForestRegressor(n_estimators=160, max_depth=7, min_samples_leaf=2,
                                  random_state=RNG_SEED)
    model.fit(X, y)
    baseline = LinearRegression()
    # Baseline uses a simple linear trend on the training window.
    baseline.fit(feat[["day_index"]], y)
    last = hist.iloc[-1]
    recent = hist["quantity"].tail(7).mean()
    last_day = int(feat["day_index"].max())
    future_rows = []
    rolling_values = list(hist["quantity"].astype(float).tail(7))
    rng = np.random.default_rng(RNG_SEED + station_id + len(item))
    for step in range(1, horizon+1):
        day = last_day + step
        # Weather/occupancy assumptions continue from recent conditions with mild seasonal variation.
        temp = float(last["temp"] if pd.notna(last["temp"]) else -35) + 1.5*np.sin(step/8)
        wind = float(last["wind"] if pd.notna(last["wind"]) else 25) + 2*np.cos(step/6)
        occ = float(last["occupancy"] if pd.notna(last["occupancy"]) else 30)
        row = pd.DataFrame([{
            "day_index":day, "temp":temp, "wind":wind, "occupancy":occ,
            "lag_1":rolling_values[-1], "lag_3":np.mean(rolling_values[-3:]),
            "lag_7":np.mean(rolling_values[-7:]), "rolling_7":np.mean(rolling_values[-7:])
        }])
        pred = max(0.01, float(model.predict(row[FEATURES])[0]))
        future_rows.append({"day":step,"predicted_usage":pred,
                            "baseline_usage":max(0.01,float(baseline.predict(pd.DataFrame({"day_index":[day]}))[0]))})
        rolling_values.append(pred)
    fdf = pd.DataFrame(future_rows)
    return {"model":model, "baseline":baseline, "history":hist, "future":fdf,
            "predicted_daily":float(fdf["predicted_usage"].mean()),
            "baseline_daily":float(fdf["baseline_usage"].mean())}

def create_forecasts(station_id, horizon=45):
    inv = read_df("SELECT * FROM inventory WHERE station_id=?", (station_id,))
    results = []
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as con:
        con.execute("DELETE FROM forecasts WHERE station_id=?", (station_id,))
        for _, itemrow in inv.iterrows():
            out = forecast_item(station_id, itemrow["item"], horizon)
            if not out:
                continue
            daily = out["predicted_daily"]
            stock = float(itemrow["current_stock"])
            days_empty = stock / daily if daily > 0 else None
            depletion = (date.today()+timedelta(days=int(days_empty))).isoformat() if days_empty is not None else None
            lead = int(itemrow["lead_days"])
            if days_empty is not None and days_empty <= lead:
                risk = "Critical"
            elif days_empty is not None and days_empty <= lead+10:
                risk = "High"
            elif days_empty is not None and days_empty <= lead+25:
                risk = "Medium"
            else:
                risk = "Low"
            target = float(itemrow["minimum_stock"]) + daily*lead*1.25
            order = max(0, target-stock)
            con.execute("""INSERT INTO forecasts
                (station_id,item,generated_at,model_name,horizon_days,predicted_daily_usage,days_to_empty,depletion_date,stockout_risk,recommended_order)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (station_id,itemrow["item"],now,"Random Forest",horizon,daily,days_empty,depletion,risk,order))
            results.append({"Item":itemrow["item"],"Predicted daily use":daily,
                            "Days to empty":days_empty,"Estimated empty date":depletion,
                            "Risk":risk,"Suggested order":order,"Unit":itemrow["unit"]})
    return pd.DataFrame(results)

# ----------------------------- MAINTENANCE MODEL -----------------------------
def equipment_features(df):
    out = df.copy()
    out["days_since_service"] = (pd.Timestamp(date.today()) - pd.to_datetime(out["last_service"])).dt.days
    out["hours_per_day_proxy"] = out["hours_used"] / np.maximum(out["days_since_service"], 1)
    return out

def maintenance_training_data():
    """Generate transparent synthetic labelled examples for a demo classifier."""
    rng = np.random.default_rng(81)
    n = 700
    vib = rng.uniform(.2,6.5,n)
    temp = rng.normal(65,22,n)
    days = rng.integers(1,220,n)
    hours = rng.uniform(200,9000,n)
    # Simulated failure label: threshold-based with noise; clearly synthetic.
    probability = 1/(1+np.exp(-(-5 + .75*vib + .035*np.maximum(temp-65,0)
                                + .018*np.maximum(days-75,0) + .00012*hours)))
    labels = (rng.random(n) < probability).astype(int)
    X = pd.DataFrame({"vibration":vib,"temperature":temp,"days_since_service":days,"hours_used":hours})
    return X, labels

def get_maintenance_flags(station_id):
    assets = read_df("SELECT * FROM equipment WHERE station_id=?", (station_id,))
    if assets.empty:
        return pd.DataFrame()
    Xtrain, ytrain = maintenance_training_data()
    clf = RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=3,
                                 class_weight="balanced", random_state=RNG_SEED)
    clf.fit(Xtrain, ytrain)
    rows = []
    with get_conn() as con:
        for _, a in assets.iterrows():
            days = max(0,(date.today()-date.fromisoformat(a["last_service"])).days)
            x = pd.DataFrame([{"vibration":a["vibration"],"temperature":a["temperature"],
                               "days_since_service":days,"hours_used":a["hours_used"]}])
            risk = float(clf.predict_proba(x)[0][1])
            reasons = []
            if a["vibration"] >= 3.5: reasons.append("High vibration")
            if a["temperature"] >= 85: reasons.append("High operating temperature")
            if days >= int(a["service_interval_days"]): reasons.append("Service interval exceeded")
            if a["hours_used"] >= 4000: reasons.append("High operating hours")
            if not reasons: reasons.append("Routine monitoring recommended")
            severity = "Critical" if risk >= .72 else ("High" if risk >= .48 else ("Medium" if risk >= .25 else "Low"))
            predicted = int(risk >= .48)
            rows.append({"id":int(a["id"]),"Equipment":a["asset_name"],"Type":a["asset_type"],
                         "Vibration":float(a["vibration"]),"Temperature °C":float(a["temperature"]),
                         "Days since service":days,"Failure risk":risk,"Severity":severity,
                         "Reasons":", ".join(reasons),"Predicted failure":predicted})
            con.execute("""INSERT INTO maintenance_flags
                (station_id,equipment_id,created_at,risk_score,severity,reasons,predicted_failure,acknowledged)
                VALUES(?,?,?,?,?,?,?,0)""",
                (station_id,int(a["id"]),datetime.now().isoformat(timespec="seconds"),
                 risk,severity,", ".join(reasons),predicted))
    return pd.DataFrame(rows)

# ----------------------------- SIMULATION / EVALUATION -----------------------------
def run_season_simulation(station_id, season_days=60):
    inv = read_df("SELECT * FROM inventory WHERE station_id=?", (station_id,))
    all_errors, baseline_errors = [], []
    baseline_stockouts, model_stockouts, avoided = 0, 0, 0
    detail = []
    rng = np.random.default_rng(900+station_id)
    for _, it in inv.iterrows():
        hist = read_df("""SELECT usage_date,quantity,weather_temp AS temp,wind_speed AS wind,occupancy
                          FROM usage_history WHERE station_id=? AND item=? ORDER BY usage_date""",
                       (station_id,it["item"]))
        if len(hist) < 30: continue
        # Hold out the latest 20% as a simulated unseen season segment.
        cut = max(20, int(len(hist)*.8))
        train = hist.iloc[:cut].copy()
        test = hist.iloc[cut:].copy()
        if len(test) < 5: test = hist.tail(20).copy(); train = hist.iloc[:-20].copy()
        train_feat = build_usage_features(train)
        if len(train_feat) < 8: continue
        rf = RandomForestRegressor(n_estimators=120,max_depth=7,min_samples_leaf=2,random_state=RNG_SEED)
        rf.fit(train_feat[FEATURES],train_feat["quantity"])
        lr = LinearRegression().fit(train_feat[["day_index"]],train_feat["quantity"])
        # Construct test lag features from actual preceding observations to evaluate one-step predictions.
        combined = pd.concat([train,test]).reset_index(drop=True)
        cfeat = build_usage_features(combined)
        test_dates = set(pd.to_datetime(test["usage_date"]).dt.date)
        test_feat = cfeat[cfeat["usage_date"].dt.date.isin(test_dates)]
        if test_feat.empty: continue
        pred = np.maximum(.01,rf.predict(test_feat[FEATURES]))
        base = np.maximum(.01,lr.predict(test_feat[["day_index"]]))
        actual = test_feat["quantity"].to_numpy()
        all_errors.extend(np.abs(actual-pred).tolist())
        baseline_errors.extend(np.abs(actual-base).tolist())
        # Operational policy comparison over the same future test usage.
        starting = float(it["current_stock"])
        min_stock = float(it["minimum_stock"])
        lead = int(it["lead_days"])
        stock_b = starting
        stock_m = starting
        for j, (actual_use,pred_use) in enumerate(zip(actual,pred)):
            # Baseline only orders after minimum-stock threshold is crossed.
            if stock_b <= min_stock and j + lead < len(actual):
                stock_b += float(np.mean(actual[max(0,j-lead):j+1]))*lead*1.1
            # Model orders when forecast depletion is within lead time plus buffer.
            if stock_m/max(pred_use,.01) <= lead+10 and j + lead < len(actual):
                stock_m += float(pred_use)*lead*1.25
            stock_b -= actual_use
            stock_m -= actual_use
            if stock_b < 0: baseline_stockouts += 1
            if stock_m < 0: model_stockouts += 1
        detail.append({"item":it["item"],"test_days":len(actual),
                       "model_MAE":float(np.mean(np.abs(actual-pred))),
                       "baseline_MAE":float(np.mean(np.abs(actual-base)))})
    # Synthetic ground truth for maintenance flag evaluation; flags are assessed against a
    # transparent operational threshold, not real-world failure records.
    assets = read_df("SELECT * FROM equipment WHERE station_id=?",(station_id,))
    flags = get_maintenance_flags(station_id) if not assets.empty else pd.DataFrame()
    if not flags.empty:
        actual_positive = ((flags["Vibration"]>=3.5) |
                           (flags["Temperature °C"]>=85) |
                           (flags["Days since service"]>=90)).astype(int)
        flag_precision = float(precision_score(actual_positive,flags["Predicted failure"],zero_division=0))
    else:
        flag_precision = 0.0
    avoided = max(0,baseline_stockouts-model_stockouts)
    model_mae = float(np.mean(all_errors)) if all_errors else 0.0
    baseline_mae = float(np.mean(baseline_errors)) if baseline_errors else 0.0
    details = {"items":detail,"note":"Season evaluation uses simulated historical observations and synthetic maintenance labels."}
    with get_conn() as con:
        cur = con.execute("""INSERT INTO simulation_runs
            (run_at,station_id,season_days,model_mae,baseline_mae,stockouts_baseline,stockouts_model,stockouts_avoided,flag_precision,details_json)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(timespec="seconds"),station_id,season_days,model_mae,baseline_mae,
             baseline_stockouts,model_stockouts,avoided,flag_precision,json.dumps(details)))
        run_id = cur.lastrowid
    return {"run_id":run_id,"model_mae":model_mae,"baseline_mae":baseline_mae,
            "stockouts_baseline":baseline_stockouts,"stockouts_model":model_stockouts,
            "stockouts_avoided":avoided,"flag_precision":flag_precision,"details":details}

# ----------------------------- NLP CHATBOT -----------------------------
# Offline NLP: TF-IDF intent classification + deterministic entity and
# quantity extraction + SQL-grounded calculations. Item-specific questions
# are handled before global dashboard intents, so one item never leaks into
# an answer about another item.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NLP_INTENT_EXAMPLES = {
    "duration": [
        "can diesel last 15 days", "will battery packs last ten days",
        "do we have enough water for 30 days", "can food boxes cover two weeks",
        "will heating oil be sufficient for 45 days",
        "can medical kits last for 60 days", "how many days can this supply last",
        "will we have enough stock for one month"
    ],
    "item_status": [
        "how much diesel do we have", "show me battery stock",
        "what is the current water quantity", "tell me the food inventory",
        "how many medical kits are available", "check heating oil levels"
    ],
    "depletion": [
        "which supply will run out first", "what item depletes soonest",
        "when will the inventory become empty", "show the earliest stockout"
    ],
    "maintenance": [
        "which equipment needs maintenance", "check generator condition",
        "which machine needs inspection", "show repair alerts",
        "is the pump operating safely"
    ],
    "reorder": [
        "what should we reorder", "which supplies need resupply",
        "how much should we purchase", "show the urgent order list"
    ],
    "evaluation": [
        "show simulation results", "what is forecast mae",
        "how many stockouts were avoided", "show maintenance precision"
    ],
    "station_summary": [
        "give me a station overview", "show overall station status",
        "is the station operationally ready", "summarise station operations"
    ],
    "help": [
        "help", "what can you do", "how do I ask questions",
        "show chatbot capabilities"
    ],
}

_NLP_TEXTS, _NLP_LABELS = [], []
for _intent_name, _examples in NLP_INTENT_EXAMPLES.items():
    for _example in _examples:
        _NLP_TEXTS.append(_example)
        _NLP_LABELS.append(_intent_name)

_NLP_VECTORIZER = TfidfVectorizer(
    ngram_range=(1, 2), stop_words="english", sublinear_tf=True
)
_NLP_MATRIX = _NLP_VECTORIZER.fit_transform(_NLP_TEXTS)

# Every inventory entity is represented here. The canonical SQL item name
# is still read from the database, so new inventory rows can be supported.
_INVENTORY_ALIASES = {
    "diesel fuel": ["diesel", "diesel fuel", "diesel oil"],
    "fresh water": ["water", "fresh water", "drinking water"],
    "food boxes": ["food", "food box", "food boxes", "rations", "food ration", "food rations"],
    "medical kits": ["medical", "medicine", "medicines", "medical kit", "medical kits", "first aid kits"],
    "heating oil": ["heating oil", "heater oil", "fuel oil"],
    "battery packs": ["battery", "batteries", "battery pack", "battery packs", "power pack", "power packs"],
}

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
}

def detect_nlp_intent(question):
    """Classify a question against example utterances using TF-IDF similarity."""
    vector = _NLP_VECTORIZER.transform([question.lower()])
    scores = cosine_similarity(vector, _NLP_MATRIX)[0]
    best_index = int(np.argmax(scores))
    if float(scores[best_index]) < 0.08:
        return "unknown", float(scores[best_index])
    return _NLP_LABELS[best_index], float(scores[best_index])

def extract_requested_days(question):
    """Read numeric or written durations, including weeks and months."""
    text = question.lower().replace(",", "")
    unit_pattern = r"(?:days?|d|weeks?|wks?|months?|mos?)"

    match = re.search(rf"\b(\d+(?:\.\d+)?)\s*({unit_pattern})\b", text)
    if match:
        amount = float(match.group(1))
        unit = match.group(2)
    else:
        match = re.search(
            rf"\b({'|'.join(sorted(_NUMBER_WORDS, key=len, reverse=True))})\s*({unit_pattern})\b",
            text
        )
        if not match:
            return None
        amount = float(_NUMBER_WORDS[match.group(1)])
        unit = match.group(2)

    if unit.startswith(("week", "wk")) or unit == "wks":
        amount *= 7
    elif unit.startswith(("month", "mo")):
        amount *= 30
    return amount

def extract_inventory_entity(question, inventory_df):
    """Return exactly one matching canonical inventory item, or None."""
    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    candidates = []

    for canonical in inventory_df["item"].dropna().astype(str).tolist():
        aliases = set(_INVENTORY_ALIASES.get(canonical.lower(), []))
        aliases.add(canonical.lower())
        for alias in aliases:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized):
                candidates.append((len(alias), canonical))

    # Prefer the longest exact phrase: "heating oil" beats "oil";
    # "battery packs" beats "battery".
    if candidates:
        candidates.sort(key=lambda pair: (pair[0], len(pair[1])), reverse=True)
        return candidates[0][1]

    # Conservative fuzzy fallback for common single-word typos.
    from difflib import get_close_matches
    words = re.findall(r"[a-z]+", normalized)
    alias_map = {}
    for canonical in inventory_df["item"].dropna().astype(str).tolist():
        for alias in _INVENTORY_ALIASES.get(canonical.lower(), [canonical.lower()]):
            if len(alias.split()) == 1:
                alias_map[alias] = canonical
    for word in words:
        matches = get_close_matches(word, list(alias_map), n=1, cutoff=0.86)
        if matches:
            return alias_map[matches[0]]
    return None

def _item_daily_usage(station_id, item, inventory_row):
    """Use the newest saved ML forecast, falling back to recent SQL usage."""
    saved = read_df(
        """SELECT predicted_daily_usage, generated_at
           FROM forecasts WHERE station_id=? AND item=?
           ORDER BY generated_at DESC, id DESC LIMIT 1""",
        (station_id, item)
    )
    if not saved.empty:
        daily = float(saved.iloc[0]["predicted_daily_usage"])
        if np.isfinite(daily) and daily > 0:
            return daily, "latest saved Random Forest forecast"

    recent = read_df(
        """SELECT quantity FROM usage_history
           WHERE station_id=? AND item=?
           ORDER BY usage_date DESC, id DESC LIMIT 30""",
        (station_id, item)
    )
    if not recent.empty:
        daily = float(recent["quantity"].mean())
        if np.isfinite(daily) and daily > 0:
            return daily, "average of the latest recorded usage observations"

    return max(float(inventory_row["daily_usage"]), 0.001), "configured daily-use estimate"

def _answer_for_inventory_item(question, station_id, item, inventory_row):
    """Answer only about the requested supply; never list unrelated supplies."""
    stock = float(inventory_row["current_stock"])
    unit = str(inventory_row["unit"])
    daily, usage_source = _item_daily_usage(station_id, item, inventory_row)
    coverage = stock / max(daily, 0.001)
    days = extract_requested_days(question)

    if days is not None:
        required = daily * days
        balance = stock - required
        if balance >= 0:
            verdict = f"**Yes. {item} is estimated to last {days:g} days.**"
            result_line = f"- Estimated remaining stock: **{balance:,.2f} {unit}**"
            status = "sufficient"
        else:
            verdict = f"**No. {item} is not estimated to last {days:g} days.**"
            result_line = f"- Estimated shortfall: **{abs(balance):,.2f} {unit}**"
            status = "insufficient"

        return (
            f"{verdict}\n\n"
            f"- Current stock: **{stock:,.2f} {unit}**\n"
            f"- Estimated daily consumption: **{daily:,.2f} {unit}/day**\n"
            f"- Requested duration: **{days:g} days**\n"
            f"- Quantity required: **{required:,.2f} {unit}**\n"
            f"{result_line}\n"
            f"- Estimated total coverage: **{coverage:.1f} days**\n\n"
            f"*Usage basis: {usage_source}. Assumes the consumption rate remains approximately constant.*"
        ), f"duration_{status}"

    return (
        f"**{item} — inventory status**\n\n"
        f"- Current stock: **{stock:,.2f} {unit}**\n"
        f"- Estimated daily consumption: **{daily:,.2f} {unit}/day**\n"
        f"- Estimated stock coverage: **{coverage:.1f} days**\n"
        f"- Minimum reserve: **{float(inventory_row['minimum_stock']):,.2f} {unit}**\n"
        f"- Resupply lead time: **{int(inventory_row['lead_days'])} days**\n\n"
        f"*Usage basis: {usage_source}. This response is limited to {item}.*"
    ), "item_status"

def chatbot_answer(user_text, station_id, station_name):
    """Offline NLP assistant grounded in the selected station's SQLite records."""
    question = user_text.strip()
    low = question.lower()
    if not question:
        return "Please ask a question about a supply, duration, equipment or station operations.", "empty"

    if re.search(r"\b(hi|hello|hey|good morning|good afternoon|good evening)\b", low):
        return (
            f"Hello! I'm PolarOps AI for **{station_name}**. Ask, for example, "
            "**Can diesel last 15 days?**, **Will water last two weeks?**, "
            "or **How many battery packs are available?**"
        ), "greeting"

    intent, confidence = detect_nlp_intent(question)

    # Fetch the selected station's inventory and resolve a named item first.
    # This ordering prevents generic keywords such as "stock" or "diesel"
    # from accidentally triggering an all-inventory summary.
    inventory = read_df(
        "SELECT * FROM inventory WHERE station_id=? ORDER BY item",
        (station_id,)
    )
    item = extract_inventory_entity(question, inventory) if not inventory.empty else None

    if item is not None:
        row = inventory[inventory["item"] == item].iloc[0]
        return _answer_for_inventory_item(question, station_id, item, row)

    # A duration without a named supply is ambiguous; ask which supply.
    if extract_requested_days(question) is not None:
        names = ", ".join(inventory["item"].astype(str).tolist())
        return f"Which supply should I check for that duration? Available items: {names}.", "clarify_item"

    assets = read_df("SELECT * FROM equipment WHERE station_id=?", (station_id,))
    forecasts = read_df(
        "SELECT * FROM forecasts WHERE station_id=? ORDER BY days_to_empty",
        (station_id,)
    )

    if intent == "help" or any(x in low for x in ["what can you do", "capabilities"]):
        return (
            "I can answer natural-language questions using station data saved in SQLite:\n\n"
            "- **Individual supplies:** diesel, heating oil, fresh water, food boxes, battery packs and medical kits\n"
            "- **Duration checks:** ask whether any supply can last 10, 15, 45 or any number of days\n"
            "- **Operations:** depletion, resupply, equipment maintenance and simulation metrics\n\n"
            "Examples: *Can diesel last 25 days?* · *Will medical kits last two months?* · "
            "*How much fresh water is available?*"
        ), "help"

    if intent == "depletion" or any(x in low for x in ["which supply runs out", "which item runs out", "run out first"]):
        if forecasts.empty:
            return "No saved forecasts yet. Open **Forecast & Resupply** and generate forecasts first.", "forecast_missing"
        r = forecasts.sort_values("days_to_empty").iloc[0]
        return (
            f"The earliest predicted depletion is **{r['item']}**, in approximately "
            f"**{float(r['days_to_empty']):.1f} days**. Risk: **{r['stockout_risk']}**."
        ), "depletion"

    if intent == "maintenance" or any(x in low for x in ["maintenance", "repair", "equipment", "generator", "pump", "machine", "vibration"]):
        if assets.empty:
            return "No equipment is registered for this station.", "maintenance"
        a = assets.copy()
        a["days_since_service"] = (pd.Timestamp(date.today()) - pd.to_datetime(a["last_service"])).dt.days
        a["attention"] = (
            (a["vibration"] >= 3.5) |
            (a["temperature"] >= 85) |
            (a["days_since_service"] >= a["service_interval_days"])
        )
        a = a.sort_values(["attention", "vibration", "temperature"], ascending=False)
        r = a.iloc[0]
        reasons = []
        if r["vibration"] >= 3.5: reasons.append("elevated vibration")
        if r["temperature"] >= 85: reasons.append("high operating temperature")
        if r["days_since_service"] >= r["service_interval_days"]: reasons.append("service interval exceeded")
        if not reasons: reasons.append("no basic alert threshold exceeded")
        return (
            f"Equipment needing the closest attention: **{r['asset_name']}**.\n\n"
            f"- Vibration: **{r['vibration']:.1f}**\n"
            f"- Temperature: **{r['temperature']:.1f} °C**\n"
            f"- Days since service: **{int(r['days_since_service'])}**\n"
            f"- Reason: {', '.join(reasons)}.\n\n"
            "This is a prototype decision-support flag, not proof of equipment failure."
        ), "maintenance"

    if intent == "reorder" or any(x in low for x in ["reorder", "resupply", "purchase", "buy supplies"]):
        if forecasts.empty:
            return "Generate forecasts first so I can calculate supply-specific reorder recommendations.", "forecast_missing"
        urgent = forecasts[forecasts["stockout_risk"].isin(["Critical", "High"])].sort_values("days_to_empty")
        if urgent.empty:
            return "No supply is currently classified as Critical or High risk by the saved forecasts.", "reorder"
        lines = [
            f"- **{r.item}:** {float(r.recommended_order):,.1f} recommended; "
            f"{float(r.days_to_empty):.1f} days of cover ({r.stockout_risk} risk)"
            for r in urgent.itertuples()
        ]
        return "Priority resupply list:\n\n" + "\n".join(lines), "reorder"

    if intent == "evaluation" or any(x in low for x in ["simulation", "evaluation", "precision", "stockouts avoided", "forecast mae"]):
        runs = read_df(
            "SELECT * FROM simulation_runs WHERE station_id=? ORDER BY id DESC LIMIT 1",
            (station_id,)
        )
        if runs.empty:
            return "No season simulation has been run yet. Open **Season Simulation** and run it first.", "evaluation"
        r = runs.iloc[0]
        return (
            f"Latest simulated evaluation:\n\n"
            f"- Forecast MAE: **{float(r['model_mae']):.2f}**\n"
            f"- Linear baseline MAE: **{float(r['baseline_mae']):.2f}**\n"
            f"- Stock-out events avoided: **{int(r['stockouts_avoided'])}**\n"
            f"- Maintenance flag precision: **{float(r['flag_precision']):.1%}**\n\n"
            "These results are based on simulated data."
        ), "evaluation"

    if intent == "station_summary" or any(x in low for x in ["overall station", "station summary", "station overview", "station status"]):
        total = len(inventory)
        critical = int(inventory["criticality"].isin(["Critical", "High"]).sum()) if total else 0
        return (
            f"**{station_name} operational summary**\n\n"
            f"- Tracked supply types: **{total}**\n"
            f"- Critical/high-priority supply types: **{critical}**\n"
            f"- Registered equipment assets: **{len(assets)}**\n"
            "Ask about an individual supply to receive only its stock and duration calculation."
        ), "station_summary"

    return (
        "I couldn't identify the requested supply or operation. Try a specific question such as "
        "**Can diesel last 15 days?**, **How much fresh water is available?**, "
        "or **Which equipment needs maintenance?**"
    ), "fallback"

# ----------------------------- UI -----------------------------
init_db()
stations = get_stations()
if stations.empty:
    st.error("No station records found. Restart the app to initialise the database.")
    st.stop()

st.sidebar.title("❄️ Polar Station Ops")
st.sidebar.caption("Remote Research Station Management")
station_names = stations["name"].tolist()
station_name = st.sidebar.selectbox("Active station", station_names)
sid = selected_station_id(station_name)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Operations Dashboard", "Forecast & Resupply", "Equipment Maintenance",
    "Season Simulation", "AI Operations Assistant", "Data & Settings"
])
st.sidebar.markdown("---")
st.sidebar.caption(f"Database: `{DB_PATH.name}`")
st.sidebar.caption("Offline-first prototype • SQLite persistence")

st.title("❄️ Polar Station Operations")
st.caption(f"{station_name} · Remote station decision-support platform · {date.today().strftime('%d %b %Y')}")

if page == "Operations Dashboard":
    inv = read_df("SELECT * FROM inventory WHERE station_id=?",(sid,))
    assets = read_df("SELECT * FROM equipment WHERE station_id=?",(sid,))
    fc = read_df("SELECT * FROM forecasts WHERE station_id=?",(sid,))
    total_stock_value = float(inv["current_stock"].sum()) if not inv.empty else 0
    urgent = int(fc["stockout_risk"].isin(["Critical","High"]).sum()) if not fc.empty else 0
    overdue = 0
    if not assets.empty:
        overdue = int(((pd.Timestamp(date.today())-pd.to_datetime(assets["last_service"])).dt.days >= assets["service_interval_days"]).sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Tracked supply types",len(inv))
    c2.metric("Forecast alerts",urgent,help="Critical or High saved depletion forecasts")
    c3.metric("Equipment overdue",overdue)
    c4.metric("Total stock quantity",f"{total_stock_value:,.0f}")
    left,right = st.columns([1.35,1])
    with left:
        st.subheader("Inventory position")
        if not inv.empty:
            plot = inv[["item","current_stock","minimum_stock"]].melt("item",var_name="Measure",value_name="Quantity")
            fig = px.bar(plot,x="item",y="Quantity",color="Measure",barmode="group",title="Current stock vs minimum reserve")
            fig.update_layout(xaxis_title="",legend_title="")
            st.plotly_chart(fig,use_container_width=True)
        else: st.info("No inventory records.")
    with right:
        st.subheader("Equipment condition")
        if not assets.empty:
            condition = assets.assign(Alert=np.where((assets.vibration>=3.5)|(assets.temperature>=85),"Inspect","Normal"))
            counts = condition["Alert"].value_counts().rename_axis("Condition").reset_index(name="Assets")
            st.plotly_chart(px.pie(counts,names="Condition",values="Assets",hole=.48),use_container_width=True)
    st.subheader("Latest supply forecast")
    if fc.empty:
        st.info("No forecast saved yet. Open Forecast & Resupply and generate one.")
    else:
        show=fc.sort_values("days_to_empty")[["item","predicted_daily_usage","days_to_empty","depletion_date","stockout_risk","recommended_order"]]
        st.dataframe(show,use_container_width=True,hide_index=True)
    st.subheader("Recent usage pattern")
    usage=read_df("SELECT usage_date,item,quantity FROM usage_history WHERE station_id=? ORDER BY usage_date DESC LIMIT 500",(sid,))
    if not usage.empty:
        usage["usage_date"]=pd.to_datetime(usage["usage_date"])
        chosen=st.selectbox("Supply to chart",sorted(usage["item"].unique()))
        chart=usage[usage.item==chosen].sort_values("usage_date")
        st.plotly_chart(px.line(chart,x="usage_date",y="quantity",markers=True,title=f"{chosen} daily usage"),use_container_width=True)

elif page == "Forecast & Resupply":
    st.subheader("Consumable depletion forecaster")
    st.write("Random Forest uses historical consumption, recent demand, temperature, wind and occupancy. Linear extrapolation is retained as a baseline for evaluation.")
    horizon=st.slider("Forecast horizon (days)",14,90,45,7)
    if st.button("🔮 Generate forecasts",type="primary"):
        with st.spinner("Training demand models and saving forecasts to SQLite..."):
            result=create_forecasts(sid,horizon)
        st.success(f"Saved {len(result)} forecast records to the SQL database.")
    fc=read_df("SELECT * FROM forecasts WHERE station_id=? ORDER BY days_to_empty",(sid,))
    if not fc.empty:
        a,b=st.columns(2)
        a.metric("Most urgent item",fc.iloc[0]["item"])
        b.metric("Days to depletion",f"{fc.iloc[0]['days_to_empty']:.1f}")
        st.dataframe(fc[["item","predicted_daily_usage","days_to_empty","depletion_date","stockout_risk","recommended_order"]],use_container_width=True,hide_index=True)
        st.subheader("Projected remaining stock")
        inv=read_df("SELECT * FROM inventory WHERE station_id=?",(sid,))
        item=st.selectbox("Choose supply",fc["item"].tolist())
        row=inv[inv.item==item].iloc[0]
        out=forecast_item(sid,item,horizon)
        if out:
            curve=float(row.current_stock)-out["future"]["predicted_usage"].cumsum()
            base=float(row.current_stock)-out["future"]["baseline_usage"].cumsum()
            dfp=pd.DataFrame({"Day":out["future"]["day"],"Random Forest":curve,"Linear baseline":base})
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=dfp.Day,y=dfp["Random Forest"],mode="lines",name="Random Forest"))
            fig.add_trace(go.Scatter(x=dfp.Day,y=dfp["Linear baseline"],mode="lines",name="Linear baseline",line=dict(dash="dash")))
            fig.add_hline(y=float(row.minimum_stock),line_dash="dot",annotation_text="Minimum reserve")
            fig.add_hline(y=0,line_dash="dash",annotation_text="Stock-out")
            fig.update_layout(xaxis_title="Days from today",yaxis_title=f"Remaining stock ({row.unit})")
            st.plotly_chart(fig,use_container_width=True)
        st.warning("Forecasts are estimates. Validate unusual weather, occupancy changes, cargo delays and manual stock counts before ordering.")
    else:
        st.info("Generate a forecast to populate this page. Forecast rows are stored in SQLite and remain available after restart.")

elif page == "Equipment Maintenance":
    st.subheader("Predictive maintenance watchlist")
    st.write("A demo Random Forest classifier estimates maintenance risk from vibration, temperature, operating hours and time since service.")
    if st.button("⚙️ Analyse equipment and save flags",type="primary"):
        with st.spinner("Scoring equipment..."):
            flags=get_maintenance_flags(sid)
        st.success(f"Saved {len(flags)} maintenance assessments to SQLite.")
    flags=read_df("""SELECT mf.id,mf.created_at,e.asset_name,mf.risk_score,mf.severity,mf.reasons,mf.acknowledged
                     FROM maintenance_flags mf JOIN equipment e ON e.id=mf.equipment_id
                     WHERE mf.station_id=? ORDER BY mf.id DESC LIMIT 30""",(sid,))
    if not flags.empty:
        latest=flags.drop_duplicates("asset_name",keep="first")
        st.dataframe(latest[["asset_name","risk_score","severity","reasons","acknowledged"]],use_container_width=True,hide_index=True)
        st.subheader("Acknowledge a maintenance flag")
        choices={f"{r.asset_name} · {r.severity} · #{r.id}":int(r.id) for r in latest.itertuples()}
        chosen=st.selectbox("Flag",list(choices))
        if st.button("Mark selected flag as acknowledged"):
            with get_conn() as con:
                con.execute("UPDATE maintenance_flags SET acknowledged=1 WHERE id=?",(choices[chosen],))
            st.success("Flag acknowledged and saved.")
            st.rerun()
    assets=read_df("SELECT * FROM equipment WHERE station_id=?",(sid,))
    st.subheader("Registered equipment")
    st.dataframe(assets,use_container_width=True,hide_index=True)
    with st.expander("Add equipment"):
        with st.form("add_equipment"):
            name=st.text_input("Equipment name")
            typ=st.selectbox("Equipment type",["Power","Heating","Water","Communications","Refrigeration","Other"])
            hours=st.number_input("Operating hours",min_value=0.0,value=1000.0)
            vib=st.number_input("Vibration reading",min_value=0.0,max_value=10.0,value=1.0)
            temp=st.number_input("Temperature °C",value=50.0)
            serviced=st.date_input("Last service date",value=date.today())
            interval=st.number_input("Service interval (days)",min_value=1,value=90)
            if st.form_submit_button("Save equipment"):
                if name.strip():
                    with get_conn() as con:
                        con.execute("""INSERT INTO equipment(station_id,asset_name,asset_type,hours_used,vibration,temperature,last_service,service_interval_days)
                                       VALUES(?,?,?,?,?,?,?,?)""",(sid,name.strip(),typ,hours,vib,temp,serviced.isoformat(),interval))
                    st.success("Equipment saved to SQL.")
                    st.rerun()
                else: st.error("Enter an equipment name.")

elif page == "Season Simulation":
    st.subheader("Simulated season evaluation")
    st.write("Evaluates the demand model against held-out simulated observations and compares it with a linear-extrapolation baseline. Maintenance precision uses synthetic labels and should be reported as prototype-only.")
    days=st.slider("Season simulation window (days)",30,120,60,10)
    if st.button("▶ Run season evaluation",type="primary"):
        with st.spinner("Running simulated season and calculating metrics..."):
            result=run_season_simulation(sid,days)
        st.success(f"Simulation #{result['run_id']} completed and saved to SQLite.")
        st.session_state["last_simulation_result"]=result
    runs=read_df("SELECT * FROM simulation_runs WHERE station_id=? ORDER BY id DESC",(sid,))
    if not runs.empty:
        r=runs.iloc[0]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Forecast MAE",f"{r['model_mae']:.2f}")
        c2.metric("Baseline MAE",f"{r['baseline_mae']:.2f}")
        c3.metric("Stock-outs avoided",int(r["stockouts_avoided"]))
        c4.metric("Flag precision",f"{r['flag_precision']:.1%}")
        compare=pd.DataFrame({"Method":["Random Forest","Linear extrapolation"],"MAE":[r["model_mae"],r["baseline_mae"]]})
        st.plotly_chart(px.bar(compare,x="Method",y="MAE",title="Forecast error comparison (lower is better)",text_auto=".2f"),use_container_width=True)
        st.write(f"Baseline stock-out events: **{int(r['stockouts_baseline'])}** · Model-policy stock-out events: **{int(r['stockouts_model'])}**")
        st.dataframe(runs[["run_at","season_days","model_mae","baseline_mae","stockouts_baseline","stockouts_model","stockouts_avoided","flag_precision"]],use_container_width=True,hide_index=True)
        st.caption("MAE is measured in the item's usage units per day, pooled across tracked supplies. Synthetic data and thresholds are for demonstration only; field validation is required.")

elif page == "AI Operations Assistant":
    st.subheader("🤖 PolarOps AI — Station operations assistant")
    st.write("Ask questions in natural language. Offline NLP identifies intent, supply names and requested durations, then retrieves the relevant SQL records. Item-specific questions return only the selected supply.")
    if "chat_messages" not in st.session_state:
        old=read_df("SELECT role,message FROM chatbot_messages ORDER BY id DESC LIMIT 30")
        st.session_state.chat_messages=[{"role":r["role"],"content":r["message"]} for _,r in old.iloc[::-1].iterrows()] if not old.empty else []
    prompts=["Which supply will run out first?","Do we have enough diesel for 30 days?","Which equipment needs attention?","What should we reorder?","Explain the latest simulation results.","Give me a station status summary."]
    cols=st.columns(3)
    for i,p in enumerate(prompts):
        if cols[i%3].button(p,key=f"prompt_{i}"):
            st.session_state.pending_prompt=p
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input=st.chat_input("Ask PolarOps about supplies, equipment, forecasts or station readiness...")
    if st.session_state.get("pending_prompt"):
        user_input=st.session_state.pop("pending_prompt")
    if user_input:
        st.session_state.chat_messages.append({"role":"user","content":user_input})
        log_chat("user",user_input,"user_query")
        with st.chat_message("user"): st.markdown(user_input)
        answer,intent=chatbot_answer(user_input,sid,station_name)
        st.session_state.chat_messages.append({"role":"assistant","content":answer})
        log_chat("assistant",answer,intent)
        with st.chat_message("assistant"): st.markdown(answer)
    if st.button("Clear visible chat"):
        st.session_state.chat_messages=[]
        st.rerun()
    st.caption("Offline NLP prototype: TF-IDF intent matching, supply entity extraction, duration parsing and SQLite-grounded calculations. It does not require an internet connection or hosted LLM.")

elif page == "Data & Settings":
    st.subheader("Station configuration")
    st.dataframe(stations,use_container_width=True,hide_index=True)
    st.subheader("Update inventory")
    inv=read_df("SELECT * FROM inventory WHERE station_id=?",(sid,))
    if not inv.empty:
        selected=st.selectbox("Supply item",inv["item"].tolist())
        current=inv[inv.item==selected].iloc[0]
        with st.form("update_inventory"):
            stock=st.number_input(f"Physical stock ({current.unit})",min_value=0.0,value=float(current.current_stock))
            minimum=st.number_input("Minimum reserve",min_value=0.0,value=float(current.minimum_stock))
            lead=st.number_input("Resupply lead time (days)",min_value=1,value=int(current.lead_days))
            usage=st.number_input("Expected daily usage",min_value=0.0,value=float(current.daily_usage))
            if st.form_submit_button("Save inventory update"):
                with get_conn() as con:
                    con.execute("""UPDATE inventory SET current_stock=?,minimum_stock=?,lead_days=?,daily_usage=?,updated_at=?
                                   WHERE station_id=? AND item=?""",
                                (stock,minimum,lead,usage,datetime.now().isoformat(timespec="minutes"),sid,selected))
                st.success("Inventory update saved to SQLite.")
                st.rerun()
    st.subheader("Add a usage observation")
    with st.form("add_usage"):
        item=st.selectbox("Supply",inv["item"].tolist(),key="usage_item")
        day=st.date_input("Usage date",value=date.today())
        qty=st.number_input("Quantity consumed",min_value=0.0,value=1.0)
        temp=st.number_input("Outside temperature °C",value=-30.0)
        wind=st.number_input("Wind speed (knots)",min_value=0.0,value=20.0)
        occ=st.number_input("Station occupancy",min_value=0,value=25)
        if st.form_submit_button("Save usage observation"):
            with get_conn() as con:
                con.execute("""INSERT INTO usage_history(station_id,item,usage_date,quantity,weather_temp,wind_speed,occupancy,is_simulated)
                               VALUES(?,?,?,?,?,?,?,0)""",(sid,item,day.isoformat(),qty,temp,wind,occ))
            st.success("Usage observation saved.")
            st.rerun()
    st.subheader("Database records")
    counts={}
    for table in ["inventory","usage_history","equipment","maintenance_flags","forecasts","simulation_runs","chatbot_messages"]:
        counts[table]=int(read_df(f"SELECT COUNT(*) AS n FROM {table}").iloc[0]["n"])
    st.json(counts)
    with st.expander("Download SQL database"):
        with open(DB_PATH,"rb") as f:
            st.download_button("Download polar_ops.db",f.read(),file_name="polar_ops.db",mime="application/x-sqlite3")
    st.warning("Reset demo data permanently deletes the local database and recreates sample records.")
    confirm=st.checkbox("I understand that resetting deletes locally saved records.")
    if st.button("Reset demo database",disabled=not confirm):
        if DB_PATH.exists(): DB_PATH.unlink()
        init_db()
        st.success("Demo database recreated.")
        st.rerun()

st.markdown("---")
st.caption("Polar Station Ops Prototype · Offline-first decision support · Validate all recommendations with station personnel before operational use.")
