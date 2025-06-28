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

        with st.form("Graphy"):
            Stock, Interval, Chart, Candles = st.columns(4)
            with Stock :
                Stock_Sel = st.selectbox("Scrip",options = ['All']+st.session_state.data['Scrip'].values.tolist())

            with Interval :
                Interval_Sel = st.selectbox("Interval", options=['1d','1h','1m'])

            with Chart :
                Chart_Sel = st.selectbox("Chart Type", options=['Candle','Line'])

            if st.form_submit_button("Generate", use_container_width=True):
                print(Stock_Sel,Interval_Sel,Chart_Sel)
                if Stock_Sel != "All":
                    stock_data = Fetch_Graph_Data(Stock_Sel, Interval_Sel, Chart_Sel)
                    stock_data = stock_data.tail(100)
                    if Chart_Sel =="Candle":
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
                            # xaxis_title="Date",
                            yaxis_title="Price (INR)",
                            xaxis2_title="Date",
                            yaxis2_title="RSI",
                            xaxis_rangeslider_visible=False,
                            template="plotly_white",
                            height=600,
                            width=900,
                        showlegend = False
                        )

                        fig.add_hline(y=60, line_dash="dash", line_color="red", row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)
                        fig.update_yaxes(title_text="Price (INR)", side="right", row=1, col=1)
                        fig.update_yaxes(title_text="RSI", side="right", row=2, col=1)
                        col1, col2, col3 = st.columns([1, 2, 1])  # Adjust ratios
                        with col2:
                            st.plotly_chart(fig)
                else:
                    Scriplist = st.session_state.data['Scrip'].values.tolist()
                    stock_datum = Fetch_Graph_Data(Scriplist, Interval_Sel, Chart_Sel)
                    figlist=[]
                    for i,stock_data in enumerate(stock_datum):
                        stock_data = stock_data.tail(100)
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            vertical_spacing=0.1, subplot_titles=(f"{Scriplist[i]} Candlestick", "RSI"),
                                            row_heights=[0.7, 0.3])

                        # Add Candlestick
                        fig.add_trace(go.Candlestick(x=stock_data.index,
                                                     open=stock_data['Open'],
                                                     high=stock_data['High'],
                                                     low=stock_data['Low'],
                                                     close=stock_data['Close'],
                                                     name=Scriplist[i]),
                                      row=1, col=1)

                        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI'),
                                      row=2, col=1)

                        fig.update_layout(
                            title=f"{Scriplist[i]} - RSI Breakout",
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
                        fig.update_yaxes(fixedrange=False)
                        fig.update_xaxes(fixedrange=False)
                        figlist.append(fig)


                    i = 0
                    j = 1
                    for item2 in figlist:
                        i += 1
                        if i == 1:
                            locals()['colchart' + str(j)], locals()['colchart' + str(j + 1)] = st.columns([1, 1])
                            with locals()['colchart' + str(j)]:
                                st.plotly_chart(item2)
                        if i == 2:
                            with locals()['colchart' + str(j + 1)]:
                                st.plotly_chart(item2)
                                i = 0
                                j += 2

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
            OP['C.price'] = ""
            OP['C.price'] = OP.apply(lambda row: Get_stock_price(row['Scrip'], pd.to_datetime(filter_date)),
                                     axis=1)
            st.dataframe(OP)





