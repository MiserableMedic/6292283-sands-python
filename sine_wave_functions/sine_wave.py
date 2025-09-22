import numpy as np
import matplotlib.pyplot as plt

def create_sine_wave(frequency, duration):
    t = np.linspace(0, duration)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    plt.plot(sine_wave, label=f'Frequency: {frequency}')