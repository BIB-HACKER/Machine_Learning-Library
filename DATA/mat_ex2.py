import matplotlib.pyplot as plt  
import numpy as np  
x = np.linspace(-10, 9, 20)

y=x**2
y1=x**3
y2=x**4
y3=x**5

plt.subplot(2,2,1)  
plt.plot(x, y, 'b*-')  
plt.subplot(2,2,2)  
plt.plot(x, y1, 'y--')  
plt.subplot(2,2,3)  
plt.plot(x, y2, 'b*-')  
plt.subplot(2,2,4)  
plt.plot(x, y3, 'y--') 
plt.show()