import pandas as pd
students={
    "names":["Baban","Raj","Bibhakar"],
    "physic":[88,99,87],
    "chem":[78,95,89]
    }
s1=pd.Series(students)
print(s1)
print("################################")
d1=pd.DataFrame(students)
print(d1)
d2=d1.set_index("names")
print(d2)
print("################################")
d3=d2.sum(axis=1)
print(d3)
print("#####subject wise average######")
d4=d2.mean(axis=0)
print(d4)

print("####################################")
s=pd.Series()
print(s)
s1=pd.Series([11,22,33,44])
print(s1)
print("#################################")
s2=pd.Series([1,3,5,7],index=['122','b','c','d'])
print(s2)
print("#################################")
s3=pd.Series(8,index=['l','o','p'])
print(s3)
print("#################################")
d={"name":"bibhakar paul","gmail":"bibhakar660@gmail.com"}
s4=pd.Series(d)
print(s4)
print("#################################")
