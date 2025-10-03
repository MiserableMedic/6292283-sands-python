import numpy as np

time_spacing = lambda strt_duration, end_duration, sample_rate : np.linspace(strt_duration, end_duration, sample_rate * abs((end_duration-strt_duration)))

def create_sine_wave(strt_duration, end_duration, frequency, sample_rate=1000, phase=0, amp=1):

    t = time_spacing(strt_duration, end_duration, sample_rate)
    sine_wave = amp*np.sin(2 * np.pi * frequency * t + phase)

    return t, sine_wave


def create_cosine_wave(strt_duration, end_duration, frequency, sample_rate=1000, phase=0, amp=1):

    t, cosine_wave = create_sine_wave(frequency,strt_duration, end_duration, sample_rate,phase=(np.pi/2+phase),amp=amp)

    return t, cosine_wave


def create_sinc_wave(strt_duration, end_duration, sample_rate=1000, phase=0,amp=1):

    t = t = time_spacing(strt_duration, end_duration, sample_rate)
    sinc_wave = amp*np.sinc(t+phase)

    return t, sinc_wave


def create_unit_step(strt_duration, end_duration, sample_rate=1000, displace=0, amp=1):

    t = time_spacing(strt_duration, end_duration, sample_rate)
    unit_step = np.where(t < 0-displace, 0, amp)

    return  t, unit_step


def create_pulse(strt_duration, end_duration, sample_rate=1000, displace=0, amp=1):

    t, step_up = create_unit_step(strt_duration, end_duration, sample_rate=sample_rate, displace=0.5-displace, amp=amp)
    t, step_down = create_unit_step(strt_duration, end_duration, sample_rate=sample_rate, displace=-0.5-displace, amp=-amp)

    pulse = amp * (step_up+step_down)

    return t, pulse