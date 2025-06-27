import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
col1, col2 = st.columns(2)

with col1:
    st.header("Left Column")
    st.button("Click Left")

with col2:
    st.header("Right Column")
    st.button("Click Right")
