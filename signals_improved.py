import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, t: np.ndarray, samples: np.ndarray, sample_rate: float):
        self.t = t
        self.samples = samples
        self.sample_rate = sample_rate

    def check_comp(self, other):
        if not np.array_equal(self.t, other.t):
            print(f"Self time array: {self.t.shape}")
            print(f"Other time array: {other.t.shape}")
            raise ValueError("Signals must have the same time array")
        if self.sample_rate != other.sample_rate:
            print(f"Self sample rate: {self.sample_rate}")
            print(f"Other sample rate: {other.sample_rate}")
            raise ValueError("Signals must have the same sample rate")

    def add(self, other):
        self.check_comp(other)
        return Signal(self.t, self.samples + other.samples, self.sample_rate)

    def multiply(self, other):
        self.check_comp(other)
        return Signal(self.t, self.samples * other.samples, self.sample_rate)

    def shift(self, displace: float):
        return Signal(self.t + displace, self.samples, self.sample_rate)

    def scale(self, factor: float):
        self.t = self.t * factor
        start, end = self.t[0], self.t[-1]
        t = int(self.sample_rate * np.abs(end - start))

        self.t = np.linspace(start, end, t)
        self.samples = np.interp(self.t, self.t / factor, self.samples)

        return Signal(self.t, self.samples, self.sample_rate)

    def amplify(self, factor: float):
        return Signal(self.t, self.samples * factor, self.sample_rate)

    def reverse(self):
        return Signal(-self.t, self.samples, self.sample_rate)

    def convolution(self, other):
        self.check_comp(other)
        convolved_samples = np.convolve(self.samples, other.samples, mode='same')
        return Signal(self.t, convolved_samples, self.sample_rate)

    def plot(self):
        print(f"time array shape: {self.t.shape}")
        plt.plot(self.t, self.samples)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()


class GenSignal:
    def __init__(self, sample_rate: float = 1000.0):
        self.sample_rate = float(sample_rate)

    def duration_split(self, duration: float):
        return duration[0], duration[1]
    
    def _time(self, duration: float):

        start, end = self.duration_split(duration)
        t = int(self.sample_rate * np.abs(end - start))

        return np.linspace(start, end, t)

    def sine(self, freq, duration, amp=1.0, phase=0.0):
        t = self._time(duration)
        samples = amp * np.sin(2*np.pi*freq*t + phase)

        return Signal(t, samples, self.sample_rate)
    
    def cosine(self, freq, duration, amp=1.0, phase=0.0):
        return self.sine(freq, duration, amp=amp, phase=(np.pi/2 + phase))
    
    def sinc(self, duration, amp=1.0, phase=0.0):
        t = self._time(duration)
        samples = amp * np.sinc(t + phase)

        return Signal(t, samples, self.sample_rate)
    
    def unit_step(self, duration, amp=1.0, displace=0.0):
        t = self._time(duration)
        samples = np.where(t < 0 - displace, 0, amp)

        return Signal(t, samples, self.sample_rate)
    
    def pulse(self, duration, amp=1.0, displace=0.0):
        step_up = self.unit_step(duration, amp=1.0, displace=(0.5 - displace))
        step_down = self.unit_step(duration, amp=-1.0, displace=(-0.5 - displace))

        samples = amp * (step_up.samples + step_down.samples)

        return Signal(step_up.t, samples, self.sample_rate)
    
    def triangle(self, duration, amp=1.0, displace=0.0):
        t = self._time(duration)
        samples = amp * (1 - np.abs(t - displace))

        samples = np.where(samples < 0, 0, samples)

        return Signal(t, samples, self.sample_rate)
    

if __name__ == "__main__":
    gen = GenSignal(sample_rate=1000)

    duration1 = [-2, 2]
    duration2 = [-4, 4]

    triangle_signal = gen.triangle(duration=duration2, amp=4, displace=0)
    pulse_signal = gen.pulse(duration=duration1, amp=4).scale(2)

    #convolved_signal = pulse_signal.convolution(triangle_signal)

    triangle_signal.plot()
    pulse_signal.plot()
    #convolved_signal.plot()
