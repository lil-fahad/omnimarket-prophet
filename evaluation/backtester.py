
def backtest_strategy(prices, predictions):
    signal = (predictions > prices.shift(1)).astype(int)
    returns = prices.pct_change().shift(-1)
    strategy_returns = signal * returns
    return strategy_returns.cumsum()
