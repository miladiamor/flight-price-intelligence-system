# app.py
import re
from datetime import datetime

from flask import Flask, request, jsonify, render_template

from services.model1_service import predict_price, predict_price_curve
from services.model2_service import advise
from services.model3_service import recommend as recommend_model3
from services.recommender_db import upsert_feedback, get_searches, get_feedback_summary

app = Flask(__name__, template_folder="templates", static_folder="static")


# -------------------------
# Validation helpers
# -------------------------
IATA_RE = re.compile(r"^[A-Z]{3}$")


def clean_iata(x):
    if x is None:
        return None
    return str(x).strip().upper()


def validate_iata(x):
    return x is not None and IATA_RE.match(x) is not None


def validate_date(date_str):
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except Exception:
        return False


# -------------------------
# Health + Landing + UI
# -------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/")
def index():
    return (
        "Backend is running ✅<br>"
        "<a href='/ui'>Open UI</a><br><br>"
        "POST /model1, POST /model1_curve, POST /model2, POST /model3, POST /predict, POST /feedback<br>"
        "GET /health, GET /searches, GET /feedbacks"
    )


@app.get("/ui")
def ui():
    return render_template("index.html")


# -------------------------
# Model 1 (Price)
# -------------------------
@app.post("/model1")
def model1_endpoint():
    data = request.get_json(silent=True) or {}
    origin = clean_iata(data.get("origin"))
    destination = clean_iata(data.get("destination"))
    date_str = data.get("date")

    if not origin or not destination or not date_str:
        return jsonify({"error": "origin, destination, date (YYYY-MM-DD) required"}), 400

    if not validate_iata(origin) or not validate_iata(destination):
        return jsonify({"error": "origin and destination must be valid IATA codes (3 letters like BER, PAR)"}), 400

    if not validate_date(date_str):
        return jsonify({"error": "date must be YYYY-MM-DD (example: 2026-02-15)"}), 400

    try:
        out = predict_price(origin, destination, date_str)
        return jsonify(out)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Model 1 Curve (Trend)
# -------------------------
@app.post("/model1_curve")
def model1_curve_endpoint():
    data = request.get_json(silent=True) or {}
    origin = clean_iata(data.get("origin"))
    destination = clean_iata(data.get("destination"))
    date_str = data.get("date")
    max_wait_days = int(data.get("max_wait_days", 7))

    if not origin or not destination or not date_str:
        return jsonify({"error": "origin, destination, date (YYYY-MM-DD) required"}), 400

    if not validate_iata(origin) or not validate_iata(destination):
        return jsonify({"error": "origin and destination must be valid IATA codes (3 letters like BER, PAR)"}), 400

    if not validate_date(date_str):
        return jsonify({"error": "date must be YYYY-MM-DD (example: 2026-02-15)"}), 400

    try:
        out = predict_price_curve(origin, destination, date_str, max_wait_days=max_wait_days)
        return jsonify(out)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Model 2 (Advisor)  ✅ UPDATED: uses Model1 + Curve
# -------------------------
@app.post("/model2")
def model2_endpoint():
    data = request.get_json(silent=True) or {}
    origin = clean_iata(data.get("origin"))
    destination = clean_iata(data.get("destination"))
    date_str = data.get("date")
    max_wait_days = int(data.get("max_wait_days", 7))

    if not origin or not destination or not date_str:
        return jsonify({"error": "origin, destination, date (YYYY-MM-DD) required"}), 400

    if not validate_iata(origin) or not validate_iata(destination):
        return jsonify({"error": "origin and destination must be valid IATA codes (3 letters like BER, PAR)"}), 400

    if not validate_date(date_str):
        return jsonify({"error": "date must be YYYY-MM-DD (example: 2026-02-15)"}), 400

    try:
        # Model 1 (calibrated now price)
        m1 = predict_price(origin, destination, date_str)

        # Model 1 curve (calibrated curve)
        curve = predict_price_curve(origin, destination, date_str, max_wait_days=max_wait_days)

        # Advisor decision based on calibrated curve
        out = advise(
            price_now=float(m1["predicted_price"]),
            curve=curve,
            days_left=int(m1["days_left"]),
        )
        return jsonify(out)

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Model 3 (Recommender)
# -------------------------
@app.post("/model3")
def model3_endpoint():
    data = request.get_json(silent=True) or {}

    user_id = str(data.get("user_id", "1"))
    origin = clean_iata(data.get("origin"))
    date_str = data.get("date")
    price = data.get("price", 200)
    k = data.get("k", 5)
    exclude_destination = data.get("exclude_destination")
    if exclude_destination:
        exclude_destination = clean_iata(exclude_destination)

    if not origin or not date_str:
        return jsonify({"error": "origin and date (YYYY-MM-DD) required"}), 400

    if not validate_iata(origin):
        return jsonify({"error": "origin must be a valid IATA code (3 letters like BER)"}), 400

    if not validate_date(date_str):
        return jsonify({"error": "date must be YYYY-MM-DD (example: 2026-02-15)"}), 400

    try:
        out = recommend_model3(
            user_id=user_id,
            origin=origin,
            date_str=date_str,
            price=float(price),
            k=int(k),
            exclude_destination=exclude_destination,
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Feedback (for Model 3)
# -------------------------
@app.post("/feedback")
def feedback_endpoint():
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "1"))
    destination = clean_iata(data.get("destination"))
    value = data.get("value")  # +1 or -1

    if not destination or value is None:
        return jsonify({"error": "destination and value (+1 or -1) required"}), 400

    if not validate_iata(destination):
        return jsonify({"error": "destination must be a valid IATA code (3 letters like PAR)"}), 400

    try:
        upsert_feedback(user_id=user_id, destination=destination, value=int(value))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/searches")
def searches_endpoint():
    try:
        return jsonify(get_searches(limit=int(request.args.get("limit", 50))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/feedbacks")
def feedbacks_endpoint():
    try:
        user_id = str(request.args.get("user_id", "1"))
        return jsonify(get_feedback_summary(user_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------
# Combined endpoint: 3 models together ✅ UPDATED Model2 wiring
# -------------------------
@app.post("/predict")
def predict_endpoint():
    data = request.get_json(silent=True) or {}

    user_id = str(data.get("user_id", "1"))
    origin = clean_iata(data.get("origin"))
    destination = clean_iata(data.get("destination"))
    date_str = data.get("date")
    k = int(data.get("k", 5))
    max_wait_days = int(data.get("max_wait_days", 7))

    if not origin or not destination or not date_str:
        return jsonify({"error": "user_id(optional), origin, destination, date required"}), 400

    if not validate_iata(origin) or not validate_iata(destination):
        return jsonify({"error": "origin and destination must be valid IATA codes (3 letters like BER, PAR)"}), 400

    if not validate_date(date_str):
        return jsonify({"error": "date must be YYYY-MM-DD (example: 2026-02-15)"}), 400

    try:
        # Model 1 (calibrated now price)
        m1 = predict_price(origin, destination, date_str)

        # Model 1 curve (calibrated curve)
        curve = predict_price_curve(origin, destination, date_str, max_wait_days=max_wait_days)

        # Model 2 advisor uses curve
        m2 = advise(
            price_now=float(m1["predicted_price"]),
            curve=curve,
            days_left=int(m1["days_left"]),
        )

        # Model 3 uses price from Model 1
        m3 = recommend_model3(
            user_id=user_id,
            origin=origin,
            date_str=date_str,
            price=float(m1["predicted_price"]),
            k=k,
            exclude_destination=destination,
        )

        return jsonify({
            "input": {
                "user_id": user_id,
                "origin": origin,
                "destination": destination,
                "date": date_str,
                "k": k,
                "max_wait_days": max_wait_days,
            },
            "model1": m1,
            "model1_curve": curve,
            "model2": m2,
            "model3": m3,
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
