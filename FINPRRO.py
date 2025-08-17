import os
from BENV import *


# st.set_page_config(layout="wide")
st.title('FINPRRO DashBoard')

st.write("Data Extracted @",ddt.now())
st.set_page_config(layout="wide")
pd.set_option('display.max_columns', None)
print(os.getcwd())

path = "FINPRRO/Scriplist.csv"
url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
r = requests.get(url, headers={'accept': 'application/vnd.github.v3.raw'})
# convert string to StringIO
string_io_obj = StringIO(r.text)

df = pd.read_csv(string_io_obj)
print("XX",df.columns)
df['N-1'] = df.apply(lambda row : round(Get_Specific_Stock_Price(row['YF_Ticker'],ddt.today()+ dt.timedelta(-1),False)[3],2),axis = 1)
df[["N_Open","N_High","N_Low","Current"]] = df.apply(
    lambda row : Get_Specific_Stock_Price(row['YF_Ticker'], ddt.today(), True),
    axis=1, result_type="expand")
df['N_Gap'] = df["N_Open"]-df["N-1"]
df["High_Avg"] = (df['N-1']+df['N_High'])/2
df["Low_Avg"] = (df['N-1']+df['N_Low'])/2
print(df)

def style_n_gap(val):
    if val > 0:
        return 'background-color: lightgreen; color: black'
    elif val < 0:
        return 'background-color: lightcoral; color: black'
    return ''

# Function to style 'Index Name' based on Current vs High_Avg/Low_Avg
def style_index_name(row):
    current = row['Current']
    high_avg = row['High_Avg']
    low_avg = row['Low_Avg']

    if current > high_avg:
        return 'background-color: lightgreen; color: black'
    elif current < low_avg:
        return 'background-color: lightcoral; color: black'
    else:
        return 'background-color: khaki; color: black'

# Apply styles
df = df[['Index Name','N_Gap', 'Current', 'N_Open', 'N_High', 'N_Low', 'N-1','High_Avg','Low_Avg']]

styled_df = df.style.format(precision=2)\
    .applymap(style_n_gap, subset=['N_Gap']) \
    .apply(lambda row: ['' if col != 'Current' else style_index_name(row) for col in df.columns], axis=1)

# Display in Streamlit
st.write(styled_df,use_container_width=True)
