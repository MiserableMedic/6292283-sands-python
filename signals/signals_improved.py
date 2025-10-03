import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, t: np.ndarray, samples: np.ndarray, sample_rate: float):
        self.t = t
        self.samples = samples
        self.sample_rate = sample_rate

    def plot(self):
        plt.plot(self.t, self.samples)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Signal')
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