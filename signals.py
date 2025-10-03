import numpy as np

def create_sine_wave(frequency, duration, sample_rate):

    t = np.linspace(0, duration, sample_rate * duration)
    sine_wave = np.sin(2 * np.pi * frequency * t)

    return t, sine_wave

def create_cosine_wave(frequency, duration, sample_rate):

    t = np.linspace(0, duration, sample_rate * duration)
    cosine_wave = np.cos(2 * np.pi * frequency * t)

    return t, cosine_wave