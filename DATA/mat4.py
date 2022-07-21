import matplotlib.pyplot as plt

import numpy as np
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'p-', color='gray',
         markersize=19, linewidth=9,
         markerfacecolor='red',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);
plt.show()