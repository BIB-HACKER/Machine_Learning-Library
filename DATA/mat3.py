import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(42)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)

plt.scatter(x, y, c=colors, s=sizes, 
            alpha=1,
            cmap='Spectral')
plt.colorbar();  # show color scale
plt.show()