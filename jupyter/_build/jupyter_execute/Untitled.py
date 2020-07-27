import pandas as pd

url = 'https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf/\
1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund'

df = pd.read_csv(url, skiprows=9)  

df

