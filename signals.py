import numpy as np

def create_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, sample_rate)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return sine_wave