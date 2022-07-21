import pandas as pd

d=pd.read_csv("Machine Learning\student_data.csv")
print(d)
print("################################")
df=pd.read_table('Machine Learning\dist.txt',delim_whitespace=True,names=('roll','name','age'))
print(df)
print("################################")
data=pd.read_csv('Machine Learning\dist.txt',delimiter='\s+',header=None,index_col=False)
data.columns=["a","b","c"]
print(data)