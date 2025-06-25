import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime as ddt
import datetime as dt
import pandas_ta as ta



YFdateform = "%Y-%m-%d"
C_Date = ddt.now()

def Get_stock_price(scrip, dattim):
    try:
        # Fetch 1-minute data for the day

        data = yf.Ticker(scrip).history(
                start=(dattim + dt.timedelta(-1)).strftime(YFdateform),
                end=(dattim + dt.timedelta(1)).strftime(YFdateform),
                interval="1m"
            )

        # print(data)
        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize(None)

        exact_match = data[data.index == dattim]

        if not exact_match.empty:
            return round(exact_match["Close"].iloc[0], 3)
        else:
            earlier_data = data[data.index <= dattim]
            if not earlier_data.empty:
                return round(earlier_data["Close"].iloc[-1], 3)
            else:
                raise ValueError("No data available before the given time.")
    except Exception as e:
        print(f"Error: {e}")
        return None


def Get_RSI(scrip, datval):
    period = 14
    # print(scrip,datval)
    try:
        start_date = (datval - dt.timedelta(days=300)).strftime(YFdateform)
        end_date = (datval + dt.timedelta(days=1)).strftime(YFdateform)

        data = yf.Ticker(scrip).history(start=start_date, end=end_date, interval="1d")
    except:
        start_date = (datval - dt.timedelta(days=300)).strftime(YFdateform)
        end_date = (datval + dt.timedelta(days=0)).strftime(YFdateform)

        data = yf.Ticker(scrip).history(start=start_date, end=end_date, interval="1d")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    # print(data['Close'])
    if data.empty or len(data) < period:
        return None

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = round(100 - (100 / (1 + rs)),1)

    return rsi.iloc[-1]

def Max_Price(Scrip, start, end, intvl):

    data = yf.Ticker(Scrip).history(start=start, end=end, interval=intvl)
    return round(data['Close'].max(),2)

