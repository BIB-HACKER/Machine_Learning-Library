import pandas as pd
s=pd.Series()
print(s)
s1=pd.Series([11,22,33,44])
print(s1)
s2=pd.Series([2,3,4,5],index=['a','b','c','d'])
print("################")
print(s2)
print("###################")
s3=pd.Series(7,index=['a','b','c','d'])
print(s3)
print("#################")
d={"name":"Mahendra","email":"dattamahendra@gmail.com"}
s4=pd.Series(d)
print(s4)
