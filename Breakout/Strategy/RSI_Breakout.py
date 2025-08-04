import pandas as pd

from BENV import *



def format_CRSI(val):
    if pd.isnull(val):
        return ""
    if val > 60:
        return f"🚩{val:.2f}"

    if val > 55:
        return f"🟢{val:.2f}"

    if val > 45:
        return f"🟡{val:.2f}"

    if val > 40:
        return f"⏳{val:.2f}"

    if val < 40:
        return f"⏸️{val:.2f}"

def format_Incr_Prc(row):
    val1 = row['% Increase']
    val2 = row['Target %']


    ratio = val1 / val2
    # print(row['Scrip'],ratio)
    if ratio > 1.1:
        return f"🚀{val1}%"
    elif ratio > 0.8:
        return f"✅{val1}%"
    elif ratio > 0.5:
        return f"✔️{val1}%"
    elif ratio > -0.1:
        return f"🚧{val1}%"
    else:

        return f"❌{val1}%"

def format_days(val):
    if val <=6:
        return f"🌱 {val}"
    elif val <= 12:
        return f"🌿 {val}"
    elif val > 12 :
        return f"🌳 {val}"

def Fetch_Data():
    df = pd.read_csv(fr'{ldir}\Breakout\CSV\RSI_Watchlist.csv')
    df['Buy_Date'] = pd.to_datetime(df['Buy_Date'],dayfirst=True)
    df['Buy_Price'] = df.apply(lambda row : Get_stock_price(row['Scrip'],row['Buy_Date'])
    if pd.isna(row['Buy_Price']) else row['Buy_Price'],axis=1)
    df['Buy_Rsi'] = df.apply(lambda row : Get_RSI(row['Scrip'],row['Buy_Date'])
    if pd.isna(row['Buy_Rsi']) else row['Buy_Rsi'],axis=1)
    df['C.price'] = df.apply(lambda row : Get_stock_price(row['Scrip'],ddt.now()),axis=1)
    df['C.RSI'] = df.apply(lambda row : Get_RSI(row['Scrip'],ddt.now()),axis=1)
    df["C.RSI"] = df["C.RSI"].apply(format_CRSI)
    df['Max Swing'] = df.apply(lambda row : Max_Price(row['Scrip'],row['Buy_Date'],ddt.now(),"1m")-row['Buy_Price'],axis=1)
    df['Swing Days'] = df.apply(
        lambda row: int(np.busday_count(row['Buy_Date'].date(), dt.date.today()))+1
        if pd.notnull(row['Buy_Date']) else None,
        axis=1
    )
    df["Swing Days"] = df["Swing Days"].apply(format_days)
    df['Swing price'] = df.apply(lambda row: row['C.price']-row['Buy_Price'],axis =1)
    df['% Increase'] = df.apply(lambda row: round(row['Swing price']*100/row['Buy_Price'],2),axis =1)
    df["% Increase"] = df.apply(format_Incr_Prc, axis=1)
    df['Max % Increase'] = df.apply(lambda row: round((row['Max Swing'])*100/row['Buy_Price'],2),axis =1)
    df["Status"] = df["Status"].fillna("Wait")
    df["Remarks"] = df["Remarks"].fillna("--")
    df["Target %"] = df["Target %"].fillna("2")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        # print(df)
        print("Good")
    column_config = {
        "Scrip": st.column_config.TextColumn("Scrip", disabled=True),
        "Buy_Date": st.column_config.DatetimeColumn("Buy Date", disabled=True),
        "Buy_Price": st.column_config.NumberColumn("Buy Price", disabled=True,format="₹ %0.2f"),
        "Buy_Rsi": st.column_config.NumberColumn("Buy RSI", disabled=True),
        "C.price": st.column_config.NumberColumn("Current Price", disabled=True,format="₹ %0.2f"),
        "C.RSI": st.column_config.TextColumn("Current RSI", disabled=True),
        "Target %": st.column_config.NumberColumn("Target %",format="%0.2f%%"),
        "Swing Days" : st.column_config.TextColumn("Swing Days",disabled=True),
        "Max Swing": st.column_config.NumberColumn("Max Swing", disabled=True,format="₹ %0.2f"),
        "Swing price": st.column_config.NumberColumn("Swing price", disabled=True,format="₹ %0.2f"),
        "% Increase": st.column_config.TextColumn("% Increase", disabled=True),
        "Max % Increase": st.column_config.NumberColumn("Max % Increase", disabled=True,format="%0.2f%%"),
        "Status": st.column_config.SelectboxColumn("Status",options=["Good","Hold","Uptrend","DownTrend - ve"])
    }
    return df,column_config

# Fetch_Data()

def Order_Log():
    df = pd.read_csv(fr'{ldir}\Breakout\CSV\RSI_Orderlog.csv')
    df['Buy_Time'] = pd.to_datetime(df['Buy_Time'],format = '%d-%m-%Y %H:%M')
    df['Buy_Price'] = df.apply(lambda row:  Get_stock_price(row['Scrip'], row['Buy_Time'] )if pd.isna(row['Buy_Price']) else row['Buy_Price'], axis=1)

    df['Sell_Time'] = pd.to_datetime(df['Sell_Time'],format = '%d-%m-%Y %H:%M')
    df['Sell_Price'] = df.apply(
        lambda row: Get_stock_price(row['Scrip'], row['Sell_Time']) if pd.isna(row['Sell_Price']) else row['Sell_Price'],axis=1)

    df['Per-Cent'] = df.apply(lambda row : (row['Sell_Price'] - row['Buy_Price'])*100/row['Buy_Price'],axis=1)

    df['Working_Days'] = np.busday_count(df['Buy_Time'].values.astype('datetime64[D]'),
                                         df['Sell_Time'].values.astype('datetime64[D]'))


    return df

