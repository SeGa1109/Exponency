import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime as ddt
import datetime as dt
import pandas_ta as ta
import numpy as np
import time
import pywhatkit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

YFdateform = "%Y-%m-%d"
C_Date = ddt.now()

def Get_stock_price(scrip, dattim):
    print(scrip,dattim)
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
    # print(rsi)
    return rsi.iloc[-1]

def Get_RSI_Data(scrip, datval):
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
    # print(rsi)
    return rsi.values.tolist()

def Max_Price(Scrip, start, end, intvl):

    data = yf.Ticker(Scrip).history(start=start, end=end, interval=intvl)
    return round(data['Close'].max(),2)

# Get_RSI("TCS.NS",ddt.now())

def Fetch_Graph_Data(Scrip,intvl,Chart_Sel):
    print(type(Scrip))
    if str(type(Scrip))!= "<class 'list'>":
        if intvl == '1d':
            start_date = (ddt.now() - dt.timedelta(days=300)).strftime(YFdateform)
            end_date = (ddt.now() - dt.timedelta(days=-1)).strftime(YFdateform)
            data = yf.Ticker(Scrip).history(start=start_date, end=end_date, interval=intvl)

        if intvl == '1h':
            start_date = (ddt.now() - dt.timedelta(days=30)).strftime(YFdateform)
            end_date = (ddt.now() - dt.timedelta(days=-1)).strftime(YFdateform)
            data = yf.Ticker(Scrip).history(start=start_date, end=end_date, interval=intvl)

        data.index = pd.to_datetime(data.index).tz_localize(None)
        period = 14
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        data['RSI'] = round(100 - (100 / (1 + rs)),1)

        # print(data)
        return data

    else:
        Datas = []
        for step in Scrip:
            if intvl == '1d':
                start_date = (ddt.now() - dt.timedelta(days=300)).strftime(YFdateform)
                end_date = (ddt.now() - dt.timedelta(days=-1)).strftime(YFdateform)
                data = yf.Ticker(step).history(start=start_date, end=end_date, interval=intvl)

            if intvl == '1h':
                start_date = (ddt.now() - dt.timedelta(days=10)).strftime(YFdateform)
                end_date = (ddt.now() - dt.timedelta(days=-1)).strftime(YFdateform)
                data = yf.Ticker(step).history(start=start_date, end=end_date, interval=intvl)

            data.index = pd.to_datetime(data.index).tz_localize(None)
            period = 14
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)

            avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

            rs = avg_gain / avg_loss
            data['RSI'] = round(100 - (100 / (1 + rs)), 1)
            Datas.append(data)
        return Datas

def RSI_Filter(Scriplist,Dateval):
    master = []
    for step in Scriplist:
        try:
            Data = Get_RSI_Data(step,Dateval)
            Data = Data[-2:]
        except:
            continue
        on_date_rsi = Data[1]
        if on_date_rsi != None and on_date_rsi < 35:
            bf_date_rsi = Data[0]
            if on_date_rsi > bf_date_rsi:
                master.append(step)
    # print(master)
    return master

def Create_RSI_Chart(stock_data,Stock_Sel):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=(f"{Stock_Sel} Candlestick", "RSI"),
                        row_heights=[0.7, 0.3])
    stock_data.index = stock_data.index.strftime('%d-%b %H:%M')
    # Add Candlestick
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name=Stock_Sel),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'),
                  row=2, col=1)

    fig.update_layout(
        title=f"{Stock_Sel} - RSI Breakout",
        # xaxis_title="Date",
        yaxis_title="Price (INR)",
        xaxis2_title="Date",
        yaxis2_title="RSI",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        width=900,
        showlegend=False
    )

    fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)
    fig.update_yaxes(title_text="Price (INR)", side="right", row=1, col=1)
    fig.update_yaxes(title_text="RSI", side="right", row=2, col=1)
    fig.update_xaxes(type='category')

    return fig


# RSI_Filter(['JBMA.NS',"WIPRO.NS"],ddt.today())


