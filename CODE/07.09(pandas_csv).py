import pandas as pd
data=pd.read_csv("Machine Learning\Income_data.csv")
print(data)
d1=(data.isnull().sum())  # empty value show on row data
print(d1)
d2=(data.fillna(1000)) # value fill on empty row data