import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime as ddt
import datetime as dt
import numpy as np
import time
# import pywhatkit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import requests
import pandas as pd
from io import StringIO
import os

owner = "SeGa1109"
repo = "Exponency"
headers = {"Authorization": "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",
           "accept": "application/vnd.github.v3.raw"}
YFdateform = "%Y-%m-%d"
C_Date = ddt.now()
ldir = fr'D:\Exponency\Git'
def Get_stock_price(scrip, dattim):
    print(scrip,dattim)
    print(type(dattim))
    if not dattim:
        return 0
    try:
        # Fetch 1-minute data for the day

        data = yf.Ticker(scrip).history(
                start=(dattim + dt.timedelta(-2)).strftime(YFdateform),
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
    if len(data.values.tolist()) ==0:
        data = yf.Ticker(Scrip).history(start=start, end=end, interval='1h')

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

        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

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
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
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
    print(Stock_Sel)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f"{Stock_Sel[0]} Candlestick", "RSI", "Normalized MACD Histogram"),
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )
    stock_data.index = stock_data.index.strftime('%d-%b')
    # Add Candlestick
    fig.add_trace(go.Candlestick(x=stock_data.index,
                                 open=stock_data['Open'],
                                 high=stock_data['High'],
                                 low=stock_data['Low'],
                                 close=stock_data['Close'],
                                 name=Stock_Sel[0]),secondary_y=False,
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'),
                  row=2, col=1)
    stock_data['Rel_Per'] = (stock_data['Close']-Stock_Sel[2])*100/Stock_Sel[2]

    fig.update_layout(
        title=f"{Stock_Sel[0]} - RSI Breakout",
        # xaxis_title="Date",
        yaxis_title="Price (INR)",
        yaxis2_title="RSI",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=700,
        width=1200,
        showlegend=False
    )

    fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)
    fig.update_yaxes(title_text="Price (INR)", side="right", row=1, col=1)
    fig.update_yaxes(title_text="RSI", side="right", row=2, col=1)
    fig.update_xaxes(type='category')

    fig.add_hline(y=Stock_Sel[2], line_dash="dot", line_color="darkgoldenrod", row=1, col=1, line_width=1)  # Price on chart
    fig.add_hline(y=Stock_Sel[3], line_dash="dot", line_color="darkgoldenrod", row=2, col=1,line_width=1)  # RSI on chart
    fig.add_vline(x=Stock_Sel[1].strftime('%d-%b'), line_dash="dot", line_color="darkgoldenrod", row=1, col=1,line_width=1)  # Datetime on candlestick
    fig.add_vline(x=Stock_Sel[1].strftime('%d-%b'), line_dash="dot", line_color="darkgoldenrod", row=2, col=1,line_width=1)  # Datetime on RSI
    # Add annotations for % change next to price axis
    buy_price = Stock_Sel[2]

    # Generate ticks for +/- N% from buy_price
    percent_range = 10    # For example, -10% to +10%
    step = 2
    pct_changes = np.arange(-percent_range, percent_range + 1, step)
    # Compute corresponding prices
    yticks = list(buy_price * (1 + pct_changes / 100))
    yticks = [round(x,1) for x in yticks]
    print(yticks)
    # Add % annotations aligned with y-ticks
    for price in yticks:
        pct = ((price - buy_price) / buy_price) * 100
        fig.add_annotation(
            x=Stock_Sel[1].strftime('%d-%b'),  # X-axis at buy date
            xref="x",
            y=price,
            text=f"{pct:+.0f}%",
            showarrow=False,
            font=dict(size=10, color="#D3D3D3"),
            align="left",
            row=1, col=1
        )
    fig.update_yaxes(
        tickvals=yticks,  # Explicitly set ticks at your % prices
        showgrid=True,  # Enable gridlines
        row=1, col=1  # If subplot used, match your row/col
    )
    fig.update_xaxes(type='category')

    stock_data['MACD_Norm'] = stock_data['MACD'] *100/ stock_data['EMA26']
    stock_data['Signal_Norm'] = stock_data['Signal'] *100/ stock_data['EMA26']
    stock_data['MACD_Hist'] = stock_data['MACD_Norm'] - stock_data['Signal_Norm']

    import matplotlib.pyplot as plt  # For colormap
    from plotly.graph_objs import Bar

    focus_threshold = 1
    # Avoid division by zero
    epsilon = 1e-6
    colors = []
    for val in stock_data['MACD_Hist']:
        if val >= 0:
            # Use greens, focus intensity scaling to ±1.5 range
            intensity = min(val / (focus_threshold + epsilon), 1.0)
            rgba = plt.cm.Greens(intensity)
        else:
            intensity = min(abs(val) / (focus_threshold + epsilon), 1.0)
            rgba = plt.cm.Reds(intensity)

        hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in rgba[:3])
        colors.append(hex_color)

    # Add MACD histogram bars with varying color
    fig.add_trace(
        Bar(
            x=stock_data.index,
            y=stock_data['MACD_Hist'],
            marker_color=colors,
            name='MACD Histogram'
        ),
        row=3, col=1
    )

    # Optional: Add Signal and MACD lines (normalized)
    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['MACD_Norm'],
        mode='lines', name='MACD (Norm)',
        line=dict(color='blue', width=1)
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=stock_data.index, y=stock_data['Signal_Norm'],
        mode='lines', name='Signal (Norm)',
        line=dict(color='red', width=1)
    ), row=3, col=1)
    fig.add_vline(x=Stock_Sel[1].strftime('%d-%b'), line_dash="dot", line_color="darkgoldenrod", row=3  , col=1,
                  line_width=1)  # Datetime on RSI

    fig.update_yaxes(title_text="MACD (Norm)", side="right", row=3, col=1)
    fig.update_yaxes(range=[-3, 3],
                     tickvals = [-2,-0.5,0,1.5,3],
                     row=3, col=1)



    return fig

def Backtest_RSI(Scrip):
    data = yf.Ticker(fr"{Scrip}").history(start='2024-01-01', end='2025-06-30')

    period = 14
    data.index = pd.to_datetime(data.index).tz_localize(None)

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    data['rsi'] = round(100 - (100 / (1 + rs)), 1)
    data = data[['Close', 'rsi']]
    data['Screened'] = (data['rsi'].shift(1) < data['rsi']) & (data['rsi'].shift(1) < 35) \
        &(data['rsi']<40)
    data['Screened'] = data['Screened'].astype(int)
    data['SignalCount'] = (data['Screened'].shift(1, fill_value=0) == 0) & (data['Screened'] == 1)
    data['MaxRSI_Rec'] = False

    signals = data.index[data['SignalCount'] == True].tolist()
    Perfect_Signal_Count = 0
    merged_blocks = []

    cleaned_signals = []
    i = 0

    while i < len(signals):
        current = signals[i]
        next_idx = i + 1
        paired = False

        while next_idx < len(signals):
            next_signal = signals[next_idx]
            workday_diff = np.busday_count(current.date(), next_signal.date())
            if workday_diff <= 10:
                rsi_current = data.loc[current, 'rsi']
                rsi_next = data.loc[next_signal, 'rsi']
                rsi_diff = rsi_next - rsi_current
                if rsi_diff >= 5:
                    # ✅ Valid pair within 10 business days
                    cleaned_signals.append((current, next_signal))
                    i = next_idx
                    paired = True
                    break
                else:
                    i = next_idx
                    next_idx += 1
            else:
                break  # stop checking once past 10-day window

        if not paired:
            # ❌ No pair found within 10 days — force pair with next signal
            try:
                next_signal = signals[i + 1]
            except:
                next_signal = data.index[-1]
            cleaned_signals.append((current, next_signal))
            i += 1  # move to next signal

    # quit()
    S1 = [(start, end - timedelta(days=1)) for start, end in cleaned_signals]
    data['SignalCount'] = False
    merged_signals = []
    for start, end in S1:
        if merged_signals and 0 <= (start - merged_signals[-1][0]).days <= 10:
            merged_signals[-1] = (merged_signals[-1][0], max(merged_signals[-1][1], end))
        else:
            merged_signals.append((start, end))

    for start, end in merged_signals:
        try:
            data.loc[data.index == start, 'SignalCount'] = True
            # print("X")
            iterblock = data.loc[start:end]
            Good_Signal = iterblock[iterblock['rsi'] >= 55]

            if not Good_Signal.empty:
                data.loc[Good_Signal.index[0], 'MaxRSI_Rec'] = True
                Perfect_Signal_Count += 1

            else:
                max_rsi = iterblock['rsi'].max()

                first_max_index = iterblock[iterblock['rsi'] == max_rsi].index[0]
                # print( iterblock[iterblock['rsi'] == max_rsi].index[0])
                data.loc[first_max_index, 'MaxRSI_Rec'] = True
        except:
            pass

    Get_in_data = data[data['SignalCount'] == True][['Close', 'rsi']]
    Get_out_data = data[data['MaxRSI_Rec'] == True][['Close', 'rsi']]
    Get_in_data = Get_in_data.reset_index()
    Get_out_data = Get_out_data.reset_index()
    Get_in_data.columns = ['Entry Date', "Entry Close", 'Entry RSI']
    Get_out_data.columns = ['Exit Date', "Exit Close", 'Exit RSI']
    # print(Get_in_data)
    # print(Get_out_data)
    Backtestdata = pd.concat([Get_in_data, Get_out_data], axis=1)
    Backtestdata['Entry Close'] = Backtestdata['Entry Close'].round(2)
    Backtestdata['Exit Close'] = Backtestdata['Exit Close'].round(2)
    Backtestdata['Swingdays'] = Backtestdata['Exit Date'] - Backtestdata['Entry Date']
    Backtestdata['Swing Percent'] = (Backtestdata['Exit Close'] - Backtestdata['Entry Close']) * 100 / Backtestdata[
        'Entry Close']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    print(f"Backtest Data :{Scrip}")
    # print(Backtestdata)
    False_Signals = (Backtestdata['Swing Percent'] < 0).sum()

    print("-----")
    stats = {
        "Total Signals": data['SignalCount'].sum(),
        "Perfect Signal Count": Perfect_Signal_Count,
        "False Signals": (Backtestdata['Swing Percent'] <= 0).sum(),
        "Minimum Entry RSI": round(Backtestdata['Entry RSI'].min(), 2),
        "Median Entry RSI": round(Backtestdata['Entry RSI'].median(), 2),
        "Maximum Exit RSI": round(Backtestdata['Exit RSI'].max(), 2),
        "Median Exit RSI": round(Backtestdata['Exit RSI'].median(), 2),
        "Median Swing Days": (Backtestdata['Swingdays']/ pd.Timedelta(days=1)).median(),
        "Median Swing Percent": round(Backtestdata['Swing Percent'].median(), 2)
    }

    stats = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])

    stats['Value'] = stats['Value'].astype(str)
    data.to_clipboard()
    Backtestdata['Entry Date'] = Backtestdata['Entry Date'].dt.strftime('%d-%m-%Y')
    Backtestdata['Exit Date'] = Backtestdata['Exit Date'].dt.strftime('%d-%m-%Y')
    Backtestdata['Swingdays'] = Backtestdata['Swingdays'].astype(str).str.extract(r'(\d+)').astype(float)
    return Backtestdata,stats
# RSI_Filter(['JBMA.NS',"WIPRO.NS"],ddt.today())

def Get_Specific_Stock_Price(scrip,dateval):

    data = yf.Ticker(scrip).history(start=(dateval + dt.timedelta(-4)).strftime(YFdateform),
                                    end=(dateval + dt.timedelta(1)).strftime(YFdateform))
    data = data.values.tolist()[-1]
    return round(data[0],2),round(data[1],2),round(data[2],2),round(data[3],2)

def Get_Specific_Stock_Close(scrip, dateval):
    print(scrip)
    data = yf.Ticker(scrip).history(start=(dateval + dt.timedelta(-4)).strftime(YFdateform),
                                    end=(dateval + dt.timedelta(1)).strftime(YFdateform))

    return data.values.tolist()[-1][3]


raw_url = f"https://raw.githubusercontent.com/SeGa1109/Exponency/main/FINPRRO/Scriplist.csv"
index_list = pd.read_csv(raw_url)
# print(index_list)












