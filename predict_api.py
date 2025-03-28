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
    "1W": 5,
    "1M": 22,
    "3M": 66,
    "1Y": 252
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

    # RSI calculation
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Price-to-Volume Correlation
    df['price_to_volume_corr'] = df['price_change'].rolling(window=7).corr(df['volume_change'])

    return df.dropna()

@app.route("/predict")
def predict():
    ticker = request.args.get("ticker", "").upper()
    range_str = request.args.get("range", "1M").upper()
    days_for_prediction = RANGE_MAP.get(range_str, 30)

    # Add buffer days for feature generation
    days_total = days_for_prediction + 60

    try:
        df = fetch_data(ticker, days_total)
        df = add_features(df)

        # Trim to prediction range only (after dropna)
        df = df.tail(days_for_prediction)

        rows = []
        last_predicted_price = None

        for i, (timestamp, row) in enumerate(df.iterrows()):
            # Extract input features
            x = row[[
                'volume_change', 'volume_rroc', 'previous_price_change',
                'previous_volume_change', 'previous_volume_rroc', 'close_position_in_range',
                'volume_ratio', 'RSI', 'price_to_volume_corr'
            ]].values.reshape(1, -1)

            # Choose model based on current price_change direction
            model = buyers_model if row["price_change"] >= 0 else sellers_model
            predicted_change = float(model.predict(x)[0]) * 100

            rows.append({
                "time": timestamp.strftime("%Y-%m-%d"),
                "actualChange": row["price_change"],
                "predictedChange": round(predicted_change, 2) if predicted_change is not None else None,
                "volume": row["v"]
            })

        return jsonify({ "data": rows })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({ "error": str(e) }), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
