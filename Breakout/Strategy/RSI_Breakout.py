from BENV import *

def format_goal(val):
    if pd.isnull(val):
        return ""
    if val > 1:
        return f"🚀{val:.0%}"
    elif val > 0.75:
        return f"✅✅{val:.0%}"
    elif val > 0:
        return f"✅{val:.0%}"
    else:
        return f"⚠️{val:.0%}"


def Fetch_Data():
    df = pd.read_csv(fr'D:\Exponency\RSI Breakout\RSI_Watchlist.csv')

    df['Buy_Date'] = pd.to_datetime(df['Buy_Date'])
    df['Buy_Price'] = df.apply(lambda row : Get_stock_price(row['Scrip'],row['Buy_Date']),axis=1)
    df['Buy_Rsi'] = df.apply(lambda row : Get_RSI(row['Scrip'],row['Buy_Date']),axis=1)

    df['C.price'] = df.apply(lambda row : Get_stock_price(row['Scrip'],ddt.now()),axis=1)
    df['C.RSI'] = df.apply(lambda row : Get_RSI(row['Scrip'],ddt.now()),axis=1)
    df['Max Price'] = df.apply(lambda row : Max_Price(row['Scrip'],row['Buy_Date'],ddt.now(),"1m"),axis=1)
    df['Swing Days'] = df.apply(lambda row: (ddt.today() - row['Buy_Date']).days, axis=1)
    df['Swing price'] = df.apply(lambda row: row['C.price']-row['Buy_Price'],axis =1)
    df['% Increase'] = df.apply(lambda row: round(row['Swing price']*100/row['Buy_Price'],2),axis =1)
    df['Max % Increase'] = df.apply(lambda row: round((row['Max Price']-row['Buy_Price'])*100/row['Buy_Price'],2),axis =1)
    df["Status"] = df["Status"].fillna("Wait")
    df["Remarks"] = df["Remarks"].fillna("--")
    df['Goal %'] = df.apply(lambda row : row['% Increase']/row['Target %'],axis =1)
    df["Goal %"] = df["Goal %"].apply(format_goal)
    df["Target %"] = df["Target %"].fillna("2")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


    column_config = {
        "Scrip": st.column_config.TextColumn("Scrip", disabled=True),
        "Buy_Date": st.column_config.DatetimeColumn("Buy Date", disabled=True),
        "Buy_Price": st.column_config.NumberColumn("Buy Price", disabled=True,format="₹ %0.2f"),
        "Buy_Rsi": st.column_config.NumberColumn("Buy RSI", disabled=True),
        "C.price": st.column_config.NumberColumn("Current Price", disabled=True,format="₹ %0.2f"),
        "C.RSI": st.column_config.NumberColumn("Current RSI", disabled=True),
        "Target %": st.column_config.NumberColumn("Target %",format="%0.2f%%"),
        "Swing Days" : st.column_config.NumberColumn("Swing Days",disabled=True),
        "Max Price": st.column_config.NumberColumn("Max Price", disabled=True,format="₹ %0.2f"),
        "Swing price": st.column_config.NumberColumn("Swing price", disabled=True,format="₹ %0.2f"),
        "% Increase": st.column_config.NumberColumn("% Increase", disabled=True,format="%0.2f%%"),
        "Max % Increase": st.column_config.NumberColumn("Max % Increase", disabled=True,format="%0.2f%%"),
        "Goal %": st.column_config.TextColumn("Goal %", disabled=True),
        "Status": st.column_config.SelectboxColumn("Status",options=["Good","Bad","Wait",])
    }
    return df,column_config

Fetch_Data()