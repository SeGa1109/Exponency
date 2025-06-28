import pandas as pd
import streamlit as st
import BENV

fcol1, fcol2 = st.columns(2)
with fcol1:
    filter_index = st.selectbox("Index List", ["ETF", "NSE_50", "NSE_500"])
with fcol2:
    filter_date = st.date_input("Data")

data = pd.read_csv(fr"D:\Exponency_Build\Streamlit_Screener\Directory\{filter_index}.csv").values.tolist()
data = [x[0] for x in data]

if st.button("Filter"):
    OP = BENV.RSI_Filter(data,filter_date)
    OP = pd.DataFrame(OP,columns=['Scrip'])
    OP['C.price'] = ""
    OP['C.price'] = OP.apply(lambda row : BENV.Get_stock_price(row['Scrip'],pd.to_datetime(filter_date)),axis=1)
    st.dataframe(OP)



