import numpy as np
a=np.array([[1,2],[5,6]])
print(a)
print('##########################')
b=np.array([[4,5],[7,8]])
print(b)
print('##########################')
c=a+b
print(c)
print('##########################')
d=a.dot(b)  # [first row(a) X first column(b)] and [f]irst row(a) X second column(b)]
            # [second row(a) X first column(b)] and [second row(a) X second coumn(b)]
print(d)
print('##########################')
print("squre root of 289:",np.sqrt(49))
print('##########################')
n1=np.arange(12).reshape(3,4)
print(n1)
print('##########################')
print("sum of ALL element: ",n1.sum())
print("row sum: ",n1.sum(axis=1))
print("column sum: ",n1.sum(axis=0))
