import numpy as np


def create_sine_wave(frequency, duration, sample_rate, phase=0, amp=1):

    t = np.linspace(0, duration, sample_rate * duration)
    sine_wave = amp*np.sin(2 * np.pi * frequency * t + phase)

    return t, sine_wave


def create_cosine_wave(frequency, duration, sample_rate, phase=0, amp=1):

    t, cosine_wave = create_sine_wave(frequency,duration,sample_rate,phase=(np.pi/2+phase),amp=amp)

    return t, cosine_wave


def create_sinc_wave(duration, sample_rate):

    t= np.linspace((-1)*duration/2, duration/2, int(sample_rate*duration))
    sinc_wave = np.sinc(t)

    return t, sinc_wave