import yfinance as yf
import pandas as pd
import numpy as np
import numpy as np
from datetime import timedelta

Scrip = 'SAREGAMA.NS'
data = yf.Ticker(fr"{Scrip}").history(start = '2024-01-01',end = '2025-06-30')

period = 14
data.index = pd.to_datetime(data.index).tz_localize(None)

delta = data['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)

avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

rs = avg_gain / avg_loss
data['rsi'] = round(100 - (100 / (1 + rs)), 1)
data = data[['Close','rsi']]
data['Screened'] = (data['rsi'].shift(1)<data['rsi']) & (data['rsi'].shift(1)<35) \
    # (data['rsi']<40)
data['Screened'] = data['Screened'].astype(int)
data['SignalCount'] = (data['Screened'].shift(1,fill_value=0)==0) & (data['Screened']==1)
data['MaxRSI_Rec'] = False

signals= data.index[data['SignalCount']==True].tolist()
Perfect_Signal_Count = 0
merged_blocks=[]

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

for start,end in merged_signals:
    try:
        data.loc[data.index==start,'SignalCount'] = True
        # print("X")
        iterblock = data.loc[start:end]
        Good_Signal = iterblock[iterblock['rsi'] >= 55]

        if not Good_Signal.empty:
            data.loc[Good_Signal.index[0],'MaxRSI_Rec'] = True
            Perfect_Signal_Count+=1

        else:
            max_rsi = iterblock['rsi'].max()

            first_max_index = iterblock[iterblock['rsi'] == max_rsi].index[0]
            # print( iterblock[iterblock['rsi'] == max_rsi].index[0])
            data.loc[first_max_index, 'MaxRSI_Rec'] = True
    except:
        pass

Get_in_data = data[data['SignalCount']==True][['Close','rsi']]
Get_out_data = data[data['MaxRSI_Rec']==True][['Close','rsi']]
Get_in_data = Get_in_data.reset_index()
Get_out_data = Get_out_data.reset_index()
Get_in_data.columns = ['Entry Date',"Entry Close",'Entry RSI']
Get_out_data.columns = ['Exit Date',"Exit Close",'Exit RSI']
# print(Get_in_data)
# print(Get_out_data)
Backtestdata = pd.concat([Get_in_data,Get_out_data],axis=1)
Backtestdata['Swingdays'] = Backtestdata['Exit Date'] - Backtestdata['Entry Date']
Backtestdata['Swing Percent'] = (Backtestdata['Exit Close']-Backtestdata['Entry Close'])*100/Backtestdata['Entry Close']
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(f"Backtest Data :{Scrip}")
print(Backtestdata)
False_Signals  = (Backtestdata['Swing Percent']<0).sum()

print("-----")
stats = {
    "Total Signals": data['SignalCount'].sum(),
    "Perfect Signal Count": Perfect_Signal_Count,
    "False Signals": (Backtestdata['Swing Percent'] <= 0).sum(),
    "Minimum Entry RSI": round(Backtestdata['Entry RSI'].min(), 2),
    "Median Entry RSI": round(Backtestdata['Entry RSI'].median(), 2),
    "Maximum Exit RSI": round(Backtestdata['Exit RSI'].max(), 2),
    "Median Exit RSI": round(Backtestdata['Exit RSI'].median(), 2),
    "Median Swing Days": Backtestdata['Swingdays'].median(),
    "Median Swing Percent": round(Backtestdata['Swing Percent'].median(), 2)
}
print(stats)
data.to_clipboard()