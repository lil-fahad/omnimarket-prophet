
import requests
import pandas as pd
import os

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "YOUR_POLYGON_KEY")

def fetch_polygon_data(symbol="AAPL", multiplier=1, timespan="day", from_date="2020-01-01", to_date="2024-12-31", limit=5000):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": POLYGON_API_KEY
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Polygon API Error: {response.status_code} | {response.text}")
    
    data = response.json().get("results", [])
    if not data:
        raise ValueError("No data returned from Polygon.")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}, inplace=True)
    df.set_index("timestamp", inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]
