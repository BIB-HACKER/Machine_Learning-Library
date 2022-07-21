import matplotlib.pyplot as plt  
import numpy as np

x = np.linspace(-10, 9, 20)

y = x ** 2

plt.plot(x, y, 'b')  
plt.xlabel('X axis')  
plt.ylabel('Y axis')  
plt.title('Cube Function')  
plt.show()  