from BENV import *

st.set_page_config(layout="wide")

from Strategy import RSI_Breakout

st.title('Exponency Build..!')

tab_widget=st.tabs(["RSi Breakout",])

Stock_list=["ETF","Nifty50", "NSE500", "NSEALL" ]

with tab_widget[0]:
    tab_widget_RSI = st.tabs(["Filter", "Watchlist", "Order Log" ])
    with tab_widget_RSI[1]:
        Inp = RSI_Breakout.Fetch_Data()
        RSI = st.data_editor(data =Inp[0],column_config=Inp[1])
        if st.button("Save",use_container_width=True):
            print(RSI)
            st.success("Changes saved!")
            # Cdf = st.dataframe(RSI)
            RSI.to_csv('D:\Exponency\RSI Breakout\RSI_Watchlist.csv',index=False)
            # print(Cdf)



