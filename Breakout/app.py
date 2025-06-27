import streamlit
import streamlit.components.v1 as components
from BENV import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")

from Strategy import RSI_Breakout

st.title('Exponency Build..!')

tab_widget=st.tabs(["RSi Breakout",'Chart'])

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
        df.to_csv('D:\Exponency\RSI Breakout\RSI_Watchlist.csv', index=False)
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
            RSI.to_csv('D:\Exponency\RSI Breakout\RSI_Watchlist.csv',index=False)

        if st.session_state.auto_refresh:
            time.sleep(30)
            Inp = RSI_Breakout.Fetch_Data()
            Notif_Check()
            st.session_state.data = Inp[0]
            st.rerun()

        Stock, Interval, Chart = st.columns(3)
        with Stock :
            Stock_Sel = st.selectbox("Scrip",options = st.session_state.data['Scrip'].values.tolist())

        with Interval :
            Interval_Sel = st.selectbox("Interval", options=['1d','1h','1m'])

        with Chart :
            Chart_Sel = st.selectbox("Chart Type", options=['Candle','Line'])

        if st.button("Generate", use_container_width=True):
            print(Stock_Sel,Interval_Sel,Chart_Sel)
            stock_data = Fetch_Graph_Data(Stock_Sel,Interval_Sel,Chart_Sel)
            stock_data = stock_data.tail(30)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                vertical_spacing=0.1, subplot_titles=(f"{Stock_Sel} Candlestick", "RSI"),
                                row_heights=[0.7, 0.3])

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
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                xaxis2_title="Date",
                yaxis2_title="RSI",
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                height=800,
                width=1000,
            showlegend = False
            )

            fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)
            fig.update_yaxes(title_text="Price (INR)", side="right", row=1, col=1)
            fig.update_yaxes(title_text="RSI", side="right", row=2, col=1)
            col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios
            with col2:
                st.plotly_chart(fig)

    with tab_widget_RSI[1]:
        streamlit.dataframe(RSI_Breakout.Order_Log())




