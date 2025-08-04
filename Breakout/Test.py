import yfinance as yf

# Ticker for Nifty 50
ticker = yf.Ticker("^NSEI")

# Target date
date = "2025-07-31"

# Get historical data for a single day
data = ticker.history(start=date, end="2025-08-01")  # end is exclusive

# Print OHLC
if not data.empty:
    ohlc = data[['Open', 'High', 'Low', 'Close']]
    print(f"OHLC for {date}:\n", ohlc)
else:
    print("No data available for that date.")