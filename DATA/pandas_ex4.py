import pandas as pd
df=pd.read_csv("Income_data.csv")
df1=df.set_index("State")
df2=df1.loc["Alabama":"Arizona","2005":"2007"]
df3=df1.iloc[:3,1:4]
df4=df1.loc[:,["2005","2012"]]
df5=df4.sum(axis=0)
df6=df1.loc["California","2005":].values
df7=df1.loc[["California","Alaska"],:]
