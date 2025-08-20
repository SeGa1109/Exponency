from BENV import *

# st.set_page_config(layout="wide")
st.title('FINPRRO DashBoard')

st.write("Data Extracted @",ddt.now())
st.set_page_config(layout="wide")
pd.set_option('display.max_columns', None)
st.session_state.auto_refresh = True

def DataPull(df):
    # print("S1")
    df['N-1'] = df.apply(lambda row : Get_Specific_Stock_Close(row['YF_Ticker'],ddt.today()+ dt.timedelta(-1)),axis = 1)
    df[["N_Open","N_High","N_Low","Current"]] = df.apply(lambda row : Get_Specific_Stock_Price(row['YF_Ticker'], ddt.today()),axis=1, result_type="expand")
    df['N_Gap'] = df["N_Open"]-df["N-1"]
    df["High_Avg"] = (df['N-1']+df['N_High'])/2
    df["Low_Avg"] = (df['N-1']+df['N_Low'])/2
    return df[['N_Gap','Index Name', 'Current', 'N_Open', 'N_High', 'N_Low', 'N-1','High_Avg','Low_Avg']]

st.session_state.data = DataPull(index_list)

def Adv_Dec_Count():
    data = st.session_state.data
    data = data.drop(data.index[0])
    # print(data)
    op=[]
    count = len(data)
    op.append(count)#index list addition
    adv = (data['Current']>data['N-1']).sum()
    op.append(adv)
    op.append(count-adv)
    Avg_Adv = (data['Current']>data['High_Avg']).sum()
    Avg_Dec = (data['Current']<data['Low_Avg']).sum()
    Nuetral = count - Avg_Adv - Avg_Dec
    op+=[Avg_Adv,Avg_Dec,Nuetral]
    return  op

Adv_Dec = Adv_Dec_Count()

st.code(fr'Index Count = {Adv_Dec[0]}       Advances🚀🟢={Adv_Dec[1]}           Declines❗🔴={Adv_Dec[2]}')

st.code(fr'Avg-Nuetral🟡 = {Adv_Dec[5]}      Avg-Advance🚀🟢={Adv_Dec[3]};       Avg-Declines❗🔴={Adv_Dec[4]};  ')

st.write("-------------")
def style_n_gap(val):
    if val >=0:
        return 'background-color: lightgreen; color: black'
    elif val < 0:
        return 'background-color: lightcoral; color: black'
    return ''

def style_index_name(row):
    current = row['Current']
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
    .applymap(style_n_gap, subset=['N_Gap']) \
    .apply(
        lambda row: [style_index_name(row) if col in ['Current', 'N_Open', 'N_High', 'N_Low', 'N-1','High_Avg','Low_Avg'] else ''
                     for col in styled_df.columns],
        axis=1
    )
st.write(styled_df,use_container_width=True)

if st.session_state.auto_refresh:
    time.sleep(3)
    st.rerun()

