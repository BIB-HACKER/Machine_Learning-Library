import pandas as pd
d=pd.read_csv("Machine Learning\Income_data.csv")
d1=d.set_index('State')
print(d1)
d2=d1.loc["Alabama":"Arizona","2005":"2007"]
print(d2)
d3=d1.iloc[:3,1:4]
print(d3)
d4=d1.loc[:,["2005","2012"]]
print(d4)
d5=d4.sum(axis=0)
print(d5)
print('################')
d6=d1.loc["California","2005":]
print(d6)
d7=d1.loc[["Alaska","California"],:]
print(d7)