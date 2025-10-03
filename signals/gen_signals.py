import numpy as np


def create_sine_wave(frequency, duration, sample_rate, phase=0, amp=1):

    t = np.linspace(0, duration, sample_rate * duration)
    sine_wave = amp*np.sin(2 * np.pi * frequency * t + phase)

    return t, sine_wave


def create_cosine_wave(frequency, duration, sample_rate, phase=0, amp=1):

    t, cosine_wave = create_sine_wave(frequency,duration,sample_rate,phase=(np.pi/2+phase),amp=amp)

    return t, cosine_wave


def create_sinc_wave(duration, sample_rate, phase=0,amp=1):

    t = np.linspace((-1)*duration, duration, sample_rate*duration)
    sinc_wave = amp*np.sinc(t+phase)

    return t, sinc_wave


def create_unit_step(strt_duration, end_duration, displace=0, amp=1):
    t = np.linspace(strt_duration, end_duration, 2000)
    unit_step = np.where(t < 0-displace, 0, amp)

    return  t, unit_step