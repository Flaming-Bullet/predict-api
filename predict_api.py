from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import os
import pickle
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models
MODEL_DIR = "/opt/render/project/src/"
buyers_model = pickle.load(open(os.path.join(MODEL_DIR, "buyers_model.pkl"), "rb"))
sellers_model = pickle.load(open(os.path.join(MODEL_DIR, "sellers_model.pkl"), "rb"))

# Polygon API key
POLYGON_API_KEY = "HpsG1iEIOwJFJ_1UcgAZUrAdwwIj0smp"

# Range map
RANGE_MAP = {
    "1W": 7,
    "1M": 30,
    "3M": 90,
    "1Y": 365
}

def fetch_data(ticker, days):
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start.date()}/{end.date()}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    res = requests.get(url)
    data = res.json().get("results", [])
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    return df

def add_features(df):
    df["price_change"] = df["c"].pct_change()
    df["volume_change"] = df["v"].pct_change()
    df["volume_rroc"] = df["volume_change"].pct_change()
    df["previous_price_change"] = df["price_change"].shift(1)
    df["previous_volume_change"] = df["volume_change"].shift(1)
    df["previous_volume_rroc"] = df["volume_rroc"].shift(1)
    df["close_position_in_range"] = (df["c"] - df["l"]) / (df["h"] - df["l"] + 1e-6)
    df["30d_volume_avg"] = abs(df['volume_change'].rolling(window=30, min_periods=1).mean())
    df["volume_ratio"] = ((df["volume_change"] / (df["30d_volume_avg"] + 1e-9)) - 1) * 100

    return df.dropna()

@app.route("/predict")
def predict():
    ticker = request.args.get("ticker", "").upper()
    range_str = request.args.get("range", "1M").upper()
    days_for_prediction = RANGE_MAP.get(range_str, 30)
    days_total = days_for_prediction + 3  # 3 extra days for feature calculation

    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400

    try:
        df = fetch_data(ticker, days_total)
        df = add_features(df)
        df = df.tail(days_for_prediction)  # trim to exact output length

        predicted_changes = []

        for _, row in df.iterrows():
            x = row[[
                'volume_change',            # 1
                'volume_rroc',              # 2
                'previous_price_change',    # 3
                'previous_volume_change',   # 4
                'previous_volume_rroc',     # 5
                'close_position_in_range',  # 6
                'volume_ratio',             # 7
                'RSI',                      # 8
                'price_to_volume_corr'      # 9
            ]].values.reshape(1, -1)
            model = buyers_model if row["price_change"] >= 0 else sellers_model
            pred = float(model.predict(x)[0]) / 100  # convert % to decimal
            predicted_changes.append(pred)

        return jsonify({"predictedChanges": predicted_changes})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
