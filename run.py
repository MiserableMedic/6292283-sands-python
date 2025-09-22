import matplotlib.pyplot as plt
import numpy as np
from signals import create_sine_wave

t,sine = create_sine_wave(5,2,1000)

plt.plot(t,sine)
plt.grid(True)
plt.show()