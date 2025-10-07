import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, t: np.ndarray, samples: np.ndarray, sample_rate: float):
        self.t = t
        self.samples = samples
        self.sample_rate = sample_rate

    def _time_scale(self, start, end, sample_rate):
        if end < start:
            start, end = end, start
        dt = 1.0 / sample_rate

        n_intervals = int(round((end - start) * sample_rate))

        n_samples = n_intervals + 1 # Include endpoint

        print(f"Creating time scale from {start} to {end} with {n_samples} samples.")

        return start + dt * np.arange(n_samples)

    def check_comp(self, other):
        rtol, atol = 1e-10, 1e-12

        if self.t.shape != other.t.shape:
            print(f"Self time array: {self.t.shape}. Length: {len(self.t)}")
            print(f"Other time array: {other.t.shape}. Length: {len(other.t)}")
            print(f"Self time array values: {self.t}")
            print(f"Other time array values: {other.t}")
            raise ValueError("Signals must have the same time array")
        
        if self.sample_rate != other.sample_rate:
            print(f"Self sample rate: {self.sample_rate}")
            print(f"Other sample rate: {other.sample_rate}")
            raise ValueError("Signals must have the same sample rate")
        
        if not np.allclose(self.t, other.t, rtol=rtol, atol=atol):
            max_dt = float(np.max(np.abs(self.t - other.t)))
            raise ValueError(f"Signals must share the same time grid; max |Δt|={max_dt}. Resample one onto the other's grid.")

    def add(self, other):
        self.check_comp(other)
        return Signal(self.t, self.samples + other.samples, self.sample_rate)

    def multiply(self, other):
        self.check_comp(other)
        return Signal(self.t, self.samples * other.samples, self.sample_rate)

    def shift(self, displace):
        return Signal(self.t + displace, self.samples, self.sample_rate)

    def scale(self, factor):
        t_old, x_old = self.t, self.samples

        start_new, end_new = t_old[0] * factor, t_old[-1] * factor
        t_new = self._time_scale(start_new, end_new, self.sample_rate)
        x_new = np.interp(t_new / factor, t_old, x_old)

        return Signal(t_new, x_new, self.sample_rate)
        
    def amplify(self, factor):
        return Signal(self.t, self.samples * factor, self.sample_rate)

    def reverse(self):
        return Signal(-self.t, self.samples, self.sample_rate)

    def convolution(self, other):
        self.check_comp(other)
        convolved_samples = np.convolve(self.samples, other.samples, mode='same') * (1 / self.sample_rate)
        return Signal(self.t, convolved_samples, self.sample_rate)

    def extend(self, duration, fill_value=0):
        start_req, end_req = duration[0], duration[1]
        if end_req <= start_req:
            raise ValueError("end must be greater than start")

        sample_rate = self.sample_rate
        start_cur, end_cur = float(self.t[0]), float(self.t[-1])

        t_new = self._time_scale(start_req, end_req, sample_rate)

        inside = (t_new >= start_cur) & (t_new <= end_cur)

        samples_new = np.full_like(t_new, fill_value, dtype=float)
        if np.any(inside):
            x_inside = np.interp(t_new[inside], self.t, self.samples)
            samples_new[inside] = x_inside

        return Signal(t_new, samples_new, sample_rate)

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

    def _duration_split(self, duration: float):
        return duration[0], duration[1]
    
    def _time(self, duration: float):

        start, end = self._duration_split(duration)
        t = int(self.sample_rate * np.abs(end - start)) + 1

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

    def ramp(self, duration, amp=1.0, displace=0.0):
        t = self._time(duration)
        samples = amp * (t - displace)

        samples = np.where(samples < 0, 0, samples)

        return Signal(t, samples, self.sample_rate)


if __name__ == "__main__":
    gen = GenSignal(sample_rate=1051)

    duration1 = [-3, 4]
    duration2 = [-3, 1]

    triangle_signal = gen.ramp(duration=duration2).extend(duration=duration1, fill_value=0.0)
    pulse_signal = gen.pulse(duration=duration1,amp=2.0)
    

    convolved_signal1 = pulse_signal.convolution(triangle_signal)

    triangle_signal.plot()
    pulse_signal.plot()
    convolved_signal1.plot()
