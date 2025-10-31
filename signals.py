import numpy as np
import matplotlib.pyplot as plt

class Signal:
    def __init__(self, t: np.ndarray, samples: np.ndarray, sample_rate: float):
        '''
        Initialize a Signal object.

        Args:
            t (np.ndarray): time array
            samples (np.ndarray): sample array
            sample_rate (float): sample rate in Hz
        '''
        self.t = t
        self.samples = samples
        self.sample_rate = sample_rate

    def _time_scale(self, start, end, sample_rate):
        '''
        Create time scale from start to end with given sample rate 
        Endpoint is included

        Args:
            start (float): start time
            end (float): end time
            sample_rate (float): sample rate in Hz

        Returns:
            np.ndarray: time array
        '''
        if end < start:
            start, end = end, start
        dt = 1.0 / sample_rate

        n_intervals = int(round((end - start) * sample_rate))

        n_samples = n_intervals + 1 # Include endpoint

        print(f"Creating time scale from {start} to {end} with {n_samples} samples.")

        return start + dt * np.arange(n_samples)

    def check_comp(self, other):
        '''
        Checks compatibility of two signals
        They must have the same time array and sample rate

        Args:
            other (Signal): Signal object to compare with

        Raises:
            ValueError: if the signals are not compatible
        '''

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
        '''
        Adds two signals
        They must have the same time array and sample rate

        Args:
            other (Signal): Signal object to add
            
        Returns:
            Signal: new Signal object with added samples
        '''

        self.check_comp(other)
        return Signal(self.t, self.samples + other.samples, self.sample_rate)

    def multiply(self, other):
        '''
        Multiplies two signals
        They must have the same time array and sample rate

        Args:
            other (Signal): Signal object to multiply   

        Returns:
            Signal: new Signal object with multiplied samples
        '''

        self.check_comp(other)
        return Signal(self.t, self.samples * other.samples, self.sample_rate)

    def shift(self, displace):
        '''
        Shifts the signal in time

        Args:
            displace (float): time displacement in seconds

        Returns:
            Signal: new Signal object with shifted time array
        '''
        return Signal(self.t + displace, self.samples, self.sample_rate)

    def scale(self, factor):
        '''
        Scales the signal in time
        
        Args:
            factor (float): time scaling factor

        Returns:
            Signal: new Signal object with scaled time array
        '''

        t_old, x_old = self.t, self.samples

        start_new, end_new = t_old[0] * factor, t_old[-1] * factor
        t_new = self._time_scale(start_new, end_new, self.sample_rate)
        x_new = np.interp(t_new / factor, t_old, x_old)

        return Signal(t_new, x_new, self.sample_rate)
        
    def amplify(self, factor):
        '''
        Amplifies the signal in amplitude

        Args:
            factor (float): amplitude scaling factor

        Returns:
            Signal: new Signal object with amplified samples
        '''

        return Signal(self.t, self.samples * factor, self.sample_rate)

    def reverse(self):
        '''
        Reverses the signal in time

        Returns:
            Signal: new Signal object with reversed time array
        '''

        return Signal(-self.t, self.samples, self.sample_rate)

    def convolution(self, other):
        '''
        Convolves two signals
        They must have the same time array and sample rate

        Args:
            other (Signal): Signal object to convolve with

        Returns:
            Signal: new Signal object with convolved samples
        '''

        self.check_comp(other)
        convolved_samples = np.convolve(self.samples, other.samples, mode='same') * (1 / self.sample_rate)
        return Signal(self.t, convolved_samples, self.sample_rate)
    

    def pad_signal(self, duration, fill_value=0):
        '''
        Pads the signal to a new duration with a fill value
        New duration must be larger than current duration

        Args:
            duration (list or tuple): new duration [start, end]
            fill_value (float): value to fill the padded areas

        Returns:
            Signal: new Signal object with padded samples
        '''

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

    def add_to_plot(self, fig_num, show=False):
        '''
        Adds the signal to a matplotlib plot

        Args:
            fig_num (int): figure number
            show (bool): whether to show the plot immediately

        Returns:
            None
        '''

        plt.subplot(3, 1, fig_num)
        plt.plot(self.t, self.samples)
        plt.grid()

        if show:
            fig = plt.gcf()
            fig.supxlabel("Time [s]")
            fig.supylabel("Amplitude")
            plt.show()


class GenSignal:
    def __init__(self, sample_rate: float = 1000.0):
        '''
        Initialize a GenSignal object.
        
        Args:
            sample_rate (float): sample rate in Hz
        '''

        self.sample_rate = float(sample_rate)

    def _duration_split(self, duration):
        '''
        Splits duration into start and end time

        Args:
            duration (list or tuple): duration [start, end]
        '''
       
        if isinstance(duration, (int, float)):
            return 0.0, float(duration)
        elif isinstance(duration, (list, tuple)) and len(duration) == 2:
            return float(duration[0]), float(duration[1])
        else:
            raise ValueError("Invalid duration format")
    
    def _time(self, duration):
        '''
        Generates time array for a given duration

        Args:
            duration (list or tuple): duration [start, end]
        '''

        start, end = self._duration_split(duration)
        t = int(self.sample_rate * np.abs(end - start)) + 1

        return np.linspace(start, end, t)

    def sine(self, freq, duration, amp=1.0, phase=0.0):
        '''
        Generates a sine wave signal
        
        Args:
            freq (float): frequency in Hz
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            phase (float): phase in radians
        Returns:
            Signal: Signal object with sine wave samples
        '''
        if freq <= 0:
            return Signal(np.zeros(len(self._time(duration))),  np.zeros(len(self._time(duration))), self.sample_rate)
        t = self._time(duration)
        samples = amp * np.sin(2*np.pi*freq*t + phase)

        return Signal(t, samples, self.sample_rate)
    
    def cosine(self, freq, duration, amp=1.0, phase=0.0):
        '''
        Generates a cosine wave signal

        Args:
            freq (float): frequency in Hz
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            phase (float): phase in radians

        Returns:
            Signal: Signal object with cosine wave samples
        '''

        return self.sine(freq, duration, amp=amp, phase=(np.pi/2 + phase))
    
    def sinc(self, duration, amp=1.0, phase=0.0):
        '''
        Generates a sinc wave signal
        
        Args:
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            phase (float): phase in radians

        Returns:
            Signal: Signal object with sinc wave samples
        '''
        t = self._time(duration)
        samples = amp * np.sinc(t + phase)

        return Signal(t, samples, self.sample_rate)
    
    def unit_step(self, duration, amp=1.0, displace=0.0):
        '''
        Generates a unit step signal

        Args:
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            displace (float): time displacement

        Returns:
            Signal: Signal object with unit step samples
        '''
        t = self._time(duration)
        samples = np.where(t < 0 - displace, 0, amp)

        return Signal(t, samples, self.sample_rate)
    
    def pulse(self, duration, amp=1.0, displace=0.0):
        '''
        Generates a unit step signal

        Args:
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            displace (float): time displacement
        
        Returns:
            Signal: Signal object with pulse samples
        '''

        step_up = self.unit_step(duration, amp=1.0, displace=(0.5 - displace))
        step_down = self.unit_step(duration, amp=-1.0, displace=(-0.5 - displace))

        samples = amp * (step_up.samples + step_down.samples)

        return Signal(step_up.t, samples, self.sample_rate)
    
    def triangle(self, duration, amp=1.0, displace=0.0):
        '''
        Generates a triangle wave signal

        Args:
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            displace (float): time displacement
        
        Returns:
            Signal: Signal object with triangle wave samples
        '''

        t = self._time(duration)
        samples = amp * (1 - np.abs(t - displace))

        samples = np.where(samples < 0, 0, samples)

        return Signal(t, samples, self.sample_rate)

    def ramp(self, duration, amp=1.0, displace=0.0):
        '''
        Generates a triangle wave signal

        Args:
            duration (list or tuple): duration [start, end]
            amp (float): amplitude
            displace (float): time displacement
        
        Returns:
            Signal: Signal object with ramp samples
        '''

        t = self._time(duration)
        samples = amp * (t - displace)

        samples = np.where(samples < 0, 0, samples)

        return Signal(t, samples, self.sample_rate)

if __name__ == "__main__":
    gen = GenSignal(sample_rate=1000)
    sig = gen.sine(freq=5.0, duration=[3,-2], amp=1.0, phase=0.0)
    sig.add_to_plot(fig_num=1, show=True)
    
