
from data.polygon_api import fetch_polygon_data

def load_data(symbol="AAPL", from_date="2020-01-01", to_date="2024-12-31"):
    df = fetch_polygon_data(symbol=symbol, from_date=from_date, to_date=to_date)
    return df
