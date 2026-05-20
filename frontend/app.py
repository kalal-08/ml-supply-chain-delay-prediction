from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "customer_cleaned.csv"
TARGET_COL = "lead_time_days"

app = Flask(__name__)


def risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


def action_plan(level: str) -> dict:
    if level == "High":
        return {
            "Action": "Expedite shipment and escalate supplier",
            "Priority": "P1",
            "Owner": "Supply Chain Manager",
            "SLA_Hours": 12,
        }
    if level == "Medium":
        return {
            "Action": "Monitor order and confirm dispatch timeline",
            "Priority": "P2",
            "Owner": "Operations Analyst",
            "SLA_Hours": 24,
        }
    return {
        "Action": "No immediate action; keep routine monitoring",
        "Priority": "P3",
        "Owner": "Auto-monitoring",
        "SLA_Hours": 72,
    }


def train_recommendation_model() -> dict:
    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} column not found in {DATA_PATH.name}")

    delay_threshold = df[TARGET_COL].quantile(0.75)
    df["delay"] = (df[TARGET_COL] > delay_threshold).astype(int)

    x_raw = df.drop(columns=[TARGET_COL, "delay"])
    x = pd.get_dummies(x_raw, drop_first=True)
    y = df["delay"]

    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(x_train, y_train)

    return {
        "df": df,
        "x_raw": x_raw,
        "x_cols": x.columns,
        "clf": clf,
        "delay_threshold": float(delay_threshold),
    }


def build_feature_schema(x_raw: pd.DataFrame) -> list:
    schema = []
    for col in x_raw.columns:
        series = x_raw[col]
        is_number = pd.api.types.is_numeric_dtype(series)
        if is_number:
            schema.append(
                {
                    "name": col,
                    "kind": "number",
                    "default": "" if pd.isna(series.iloc[0]) else str(series.iloc[0]),
                }
            )
        else:
            unique_vals = [str(v) for v in series.dropna().astype(str).unique().tolist()[:20]]
            schema.append(
                {
                    "name": col,
                    "kind": "category",
                    "default": "" if pd.isna(series.iloc[0]) else str(series.iloc[0]),
                    "options": unique_vals,
                }
            )
    return schema


def parse_input_to_row(form_data, schema: list) -> pd.DataFrame:
    row = {}
    for feature in schema:
        name = feature["name"]
        raw_val = form_data.get(name, "").strip()

        if feature["kind"] == "number":
            row[name] = float(raw_val) if raw_val else 0.0
        else:
            row[name] = raw_val

    return pd.DataFrame([row])


def recommend_for_order(input_row: pd.DataFrame) -> dict:
    input_encoded = pd.get_dummies(input_row, drop_first=True)
    input_encoded = input_encoded.reindex(columns=STATE["x_cols"], fill_value=0)

    prob = float(STATE["clf"].predict_proba(input_encoded)[0][1])
    level = risk_level(prob)
    plan = action_plan(level)

    return {
        "Delay_Probability": round(prob, 4),
        "Risk_Level": level,
        **plan,
    }


STATE = train_recommendation_model()
FEATURE_SCHEMA = build_feature_schema(STATE["x_raw"])


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted = {}

    if request.method == "POST":
        submitted = {k: v for k, v in request.form.items()}
        try:
            input_row = parse_input_to_row(request.form, FEATURE_SCHEMA)
            result = recommend_for_order(input_row)
        except ValueError:
            error = "Please enter valid numeric values in number fields."

    return render_template(
        "index.html",
        feature_schema=FEATURE_SCHEMA,
        result=result,
        submitted=submitted,
        error=error,
        threshold=STATE["delay_threshold"],
    )


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    payload = request.get_json(silent=True) or {}

    try:
        input_row = parse_input_to_row(payload, FEATURE_SCHEMA)
        result = recommend_for_order(input_row)
        return jsonify({"ok": True, "result": result})
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid numeric values."}), 400


if __name__ == "__main__":
    app.run(debug=True)
