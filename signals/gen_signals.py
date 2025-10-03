import numpy as np

def create_sine_wave(frequency, duration, sample_rate):

    t = np.linspace(0, duration, sample_rate * duration)
    sine_wave = np.sin(2 * np.pi * frequency * t)

    return t, sine_wave

def create_cosine_wave(frequency, duration, sample_rate):

    t, sine_wave = create_sine_wave(frequency,duration,sample_rate)

    cosine_wave = sine_wave - (np.pi/2)

    return t, cosine_wave