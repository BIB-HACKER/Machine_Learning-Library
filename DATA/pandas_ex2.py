import pandas as pd
students={
    "names":["Raj","Avik","kamal"],
    "phys":[88,99,87],
    "chem":[78,88,77]
    }
s1=pd.Series(students)
print(s1)
print("##########################")
d1=pd.DataFrame(students)
print(d1)
d2=d1.set_index("names")
print(d2)
print("###################")
d3=d2.sum(axis=1)
print(d3)
print("###Subject wise average########")
d4=d2.mean(axis=0)
print(d4)






