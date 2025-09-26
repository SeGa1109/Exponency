from BENV import *
from zoneinfo import ZoneInfo
# st.set_page_config(layout="wide")
st.title('FINPRRO DashBoard')

st.write(fr"Data Extracted @{ddt.now(ZoneInfo("Asia/Kolkata"))}")
st.set_page_config(layout="wide")
pd.set_option('display.max_columns', True)
# st.session_state.auto_refresh = False

def DataPull(df):
    # print("S1")
    df['Prev_Close'] = df.apply(lambda row : Get_Specific_Stock_Close(row['YF_Ticker'],ddt.today()+ dt.timedelta(-1)),axis = 1)
    df[["Open","High","Low","Current_LTP"]] = df.apply(lambda row : Get_Specific_Stock_Price(row['YF_Ticker'], ddt.today()),axis=1, result_type="expand")
    df['Gap'] = df["Open"]-df["Prev_Close"]
    df["High_Avg"] = (df['Prev_Close']+df['High'])/2
    df["Low_Avg"] = (df['Prev_Close']+df['Low'])/2
    return df[['Index Name','Prev_Close','Gap','Open','Current_LTP','Low','Low_Avg','High', 'High_Avg',]]

st.session_state.data = DataPull(index_list)

def Adv_Dec_Count():
    data = st.session_state.data
    data = data.drop(data.index[0])
    # print(data)
    op=[]
    count = len(data)
    op.append(count)#index list addition
    adv = (data['Current_LTP']>data['Prev_Close']).sum()
    op.append(adv)
    op.append(count-adv)
    Avg_Adv = (data['Current_LTP']>data['High_Avg']).sum()
    Avg_Dec = (data['Current_LTP']<data['Low_Avg']).sum()
    Nuetral = count - Avg_Adv - Avg_Dec
    op+=[Avg_Adv,Avg_Dec,Nuetral]
    return  op

Adv_Dec = Adv_Dec_Count()

st.code(fr'Index Count = {Adv_Dec[0]}; 🚀🟢={Adv_Dec[1]}; ❗🔴={Adv_Dec[2]}')

st.code(fr'Average :: 🟡={Adv_Dec[5]}; 🚀🟢={Adv_Dec[3]}; ❗🔴={Adv_Dec[4]};  ')
if st.toggle("Auto Refresh"):
    st.session_state.auto_refresh = True
else:
    st.session_state.auto_refresh = False



st.write("-------------")
def style_gap(val):
    if val >=0:
        return 'background-color: lightgreen; color: black'
    elif val < 0:
        return 'background-color: lightcoral; color: black'
    return ''

def style_index_name(row):
    current = row['Current_LTP']
    high_avg = row['High_Avg']
    low_avg = row['Low_Avg']

    cap = 0.01  # 2% cap

    if current > high_avg:
        # % above high_avg, capped at 2%
        pct = min((current - high_avg) / high_avg, cap) / cap
        # Light green → Dark green
        r1, g1, b1 = (200, 230, 201)  # light green
        r2, g2, b2 = (46, 125, 50)    # dark green
    elif current < low_avg:
        # % below low_avg, capped at 2%
        pct = min((low_avg - current) / low_avg, cap) / cap
        # Light red → Dark red
        r1, g1, b1 = (255, 205, 210)  # light red
        r2, g2, b2 = (198, 40, 40)    # dark red
    else:
        return 'background-color: #FFFACD; color: black;'  # neutral yellow

    # Linear interpolation
    r = int(r1 + (r2 - r1) * pct)
    g = int(g1 + (g2 - g1) * pct)
    b = int(b1 + (b2 - b1) * pct)

    return f'background-color: rgb({r},{g},{b}); color: black;'


styled_df = st.session_state.data
styled_df = styled_df.style.format(precision=2) \
    .applymap(style_gap, subset=['Gap']) \
    .apply(
        lambda row: [style_index_name(row) if col in ['Current_LTP', 'Open', 'High', 'Low', 'Prev_Close','High_Avg','Low_Avg'] else ''
                     for col in styled_df.columns],
        axis=1
    )
st.dataframe(styled_df, height=800)

if st.session_state.auto_refresh:
    time.sleep(1)
    st.rerun()
