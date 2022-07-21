
import matplotlib.pyplot as plt

import numpy as np
x = np.linspace(0, 10, 30)
y = np.sin(x)

plt.plot(x, y, 'o', color='black');
plt.show()
#####################################
plt.plot(x, y, '-p', color='gray',
         markersize=25, linewidth=4,
         markerfacecolor='red',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2, 1.2);
plt.show()
################################
rng = np.random.RandomState(42)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.9,
            cmap='autumn')
plt.colorbar();  # show color scale
plt.show()
############################