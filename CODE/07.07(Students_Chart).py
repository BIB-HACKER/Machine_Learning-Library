import pandas as pd
s=pd.read_csv("Machine Learning\student_data.csv")
print(s)
s1=s.set_index("names")
print(s1)
marks=s1.iloc[:,2:]
print(marks)
total_marks=marks.sum(axis=1)
print(total_marks)

import matplotlib.pyplot as plt
names=s.iloc[:,1]
print(names)
plt.bar(names,total_marks)
plt.show()
