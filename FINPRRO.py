from BENV import *
from zoneinfo import ZoneInfo
# st.set_page_config(layout="wide")
st.title('FINPRO DashBoard')
ist = ZoneInfo("Asia/Kolkata")
if "selected_datetime" not in st.session_state:
    st.session_state.selected_datetime = ddt.now(ist).replace(tzinfo=None)
with st.form("datetime_form"):

    selected_datetime = st.datetime_input(
        "Select Date & Time",value=st.session_state.selected_datetime,
        format="DD/MM/YYYY"
    )

    ok = st.form_submit_button("OK")
st.write(fr"Data Extracted @{selected_datetime}")
st.set_page_config(layout="wide")
pd.set_option('display.max_columns', True)
# st.session_state.auto_refresh = False

def DataPull(df,timeframe):
    #print(fr"{timeframe+ dt.timedelta(-1),"-",timeframe}")
    df['Prev_Close'] = df.apply(lambda row : Get_Specific_Stock_Close(row['YF_Ticker'],timeframe - dt.timedelta(1)),axis = 1)
    df[["Open","High","Low","Current_LTP"]] = df.apply(lambda row : Get_Specific_Stock_Price(row['YF_Ticker'], timeframe),axis=1, result_type="expand")
    df['Gap'] = df["Open"]-df["Prev_Close"]
    df["High_Avg"] = (df['Prev_Close']+df['High'])/2
    df["Low_Avg"] = (df['Prev_Close']+df['Low'])/2
    return df[['Index Name','Prev_Close','Gap','Open','Current_LTP','Low','Low_Avg','High', 'High_Avg',]]

st.session_state.data = DataPull(index_list,selected_datetime)

def Adv_Dec_Count():
    data = st.session_state.data
    data = data.drop(data.index[0])
    data = data.drop(data.index[1])
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

st.code(fr'Index Count = {Adv_Dec[0]}; 🚀🟢={Adv_Dec[1]}; ❗🔴={Adv_Dec[2]}   ')

st.code(fr'Average :: 🟡={Adv_Dec[5]}; 🚀🟢={Adv_Dec[3]}; ❗🔴={Adv_Dec[4]};  ')



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


df = st.session_state.data

styled_df = (
    df.style
      .format(precision=2)
      .map(style_gap, subset=['Gap'])
      .apply(
          lambda row: [
              style_index_name(row)
              if col in ['Current_LTP', 'Open', 'High', 'Low',
                         'Prev_Close', 'High_Avg', 'Low_Avg']
              else ''
              for col in row.index          # or df.columns
          ],
          axis=1
      )
)

st.dataframe(styled_df, height=800)


