import streamlit
import streamlit.components.v1 as components
from BENV import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

from Strategy import RSI_Breakout

st.title('Exponency Build..!')

tab_widget=st.tabs(["RSI Breakout"])

Stock_list=["ETF","Nifty50", "NSE500", "NSEALL" ]


def Notif_Check():
    df = st.session_state.data
    df1 = df.copy()
    df1['C.RSI'] = df1['C.RSI'].str.extract(r"(\d+\.?\d*)").astype(float)
    df1 = df1[df1['Notif']==True]
    scriplist = df1[df1['C.RSI']>52]['Scrip'].values.tolist()
    print(scriplist)
    if scriplist:
        df.loc[df["Scrip"].isin(scriplist), "Notif"] = False
        df.to_csv('D:\Exponency\Git\Breakout\CSV\RSI_Watchlist.csv', index=False)
        st.session_state.data = df
        scriplist = '\n'.join(scriplist)
        MSG = f"Stocks Crossed RSI -55 \n {scriplist}"
        pywhatkit.sendwhatmsg_instantly("+919952509481", MSG, wait_time=10, tab_close=True)

with tab_widget[0]:

    tab_widget_RSI = st.tabs(["Watchlist", "Order Log" ,"Filter",])
    with tab_widget_RSI[0]:
        if "auto_refresh" not in st.session_state:
            st.session_state.auto_refresh = False
        st.session_state.auto_refresh = st.toggle("Auto Refresh", value=st.session_state.auto_refresh)

        if st.button("Refresh", use_container_width=True):
            Inp = RSI_Breakout.Fetch_Data()
            st.session_state.data = Inp[0]
            Notif_Check()
            st.rerun()

        Inp = RSI_Breakout.Fetch_Data()
        if "data" not in st.session_state:
            st.session_state.data = Inp[0]

        RSI = st.data_editor(st.session_state.data,column_config=Inp[1],use_container_width=True)
        if st.button("Save",use_container_width=True):
            # print(RSI)
            st.success("Changes saved!")
            RSI.to_csv('D:\Exponency\Git\Breakout\CSV\RSI_Watchlist.csv',index=False)

        if st.session_state.auto_refresh:
            time.sleep(30)
            Inp = RSI_Breakout.Fetch_Data()
            Notif_Check()
            st.session_state.data = Inp[0]
            st.rerun()

        with st.form("Graphy"):
            Stock, Interval, Chart, Candles = st.columns(4)
            with Stock :
                Stock_Sel = st.selectbox("Scrip",options = ['All']+st.session_state.data['Scrip'].values.tolist())

            with Interval :
                Interval_Sel = st.selectbox("Interval", options=['1d','1h','1m'])

            with Chart :
                Chart_Sel = st.selectbox("Chart Type", options=['Candle','Line'])

            with Candles :
                Candles_Sel = st.selectbox("Candle Stretch", options=['100','250','500',])

            if st.form_submit_button("Generate", use_container_width=True):
                print(Stock_Sel,Interval_Sel,Chart_Sel)
                if Stock_Sel != "All":
                    stock_data = Fetch_Graph_Data(Stock_Sel, Interval_Sel, Chart_Sel)
                    stock_data = stock_data.tail(int(Candles_Sel))

                    if Chart_Sel =="Candle":
                        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios
                        with col2:
                            st.plotly_chart(Create_RSI_Chart(stock_data,Stock_Sel))
                else:
                    Scriplist = st.session_state.data['Scrip'].values.tolist()
                    stock_datum = Fetch_Graph_Data(Scriplist, Interval_Sel, Chart_Sel)
                    figlist=[]
                    for i,stock_data in enumerate(stock_datum):
                        stock_data = stock_data.tail(int(Candles_Sel))
                        figlist.append(Create_RSI_Chart(stock_data,Scriplist[i]))
                    print("len",len(figlist))
                    i = 0
                    while i < len(figlist):
                        cols = st.columns(2)
                        with cols[0]:
                            if isinstance(figlist[i], go.Figure):
                                st.plotly_chart(figlist[i])

                        if i + 1 < len(figlist):
                            with cols[1]:
                                if isinstance(figlist[i + 1], go.Figure):
                                    st.plotly_chart(figlist[i + 1])

                        i += 2  # Move to the next pair

    with tab_widget_RSI[1]:
        streamlit.dataframe(RSI_Breakout.Order_Log())

    with tab_widget_RSI[2]:
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            filter_index = st.selectbox("Index List", ["ETF", "NSE_50", "NSE_500"])
        with fcol2:
            filter_date = st.date_input("Data")

        data = pd.read_csv(fr"D:\Exponency_Build\Streamlit_Screener\Directory\{filter_index}.csv").values.tolist()
        data = [x[0] for x in data]

        if st.button("Filter"):
            OP = RSI_Filter(data, filter_date)
            OP = pd.DataFrame(OP, columns=['Scrip'])
            # OP['C.price'] = ""
            OP['C.price'] = OP.apply(lambda row: Get_stock_price(row['Scrip'], pd.to_datetime(filter_date)),
                                     axis=1)
            st.dataframe(OP)





