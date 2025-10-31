import numpy as np
import pytest
from signals import GenSignal

#----------------------------------------------------------------------
# Fixtures
#----------------------------------------------------------------------

@pytest.fixture(scope="module")
def fs():
    return 1000

@pytest.fixture(scope="module")
def tol():
    return 1e-6

#----------------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------------

def parse_duration(dur):
    if isinstance(dur, (int, float)):
        return 0.0, float(dur)
    elif isinstance(dur, (list, tuple)) and len(dur) == 2:
        return float(dur[0]), float(dur[1])
    else:
        raise ValueError("Invalid duration format")

def expected_len(fs, dur):
    start, end = parse_duration(dur)
    return int(fs * abs(end - start)) + 1

def assert_timegrid(t, fs, dur, tol):
    start, end = parse_duration(dur)
    n = expected_len(fs, dur)
    ref = np.linspace(start, end, n)
    assert len(t) == n
    assert np.allclose(t, ref, atol=tol, rtol=0)

#----------------------------------------------------------------------
# Tests for GenSignal.sine & GenSignal.cosine
#----------------------------------------------------------------------

@pytest.mark.parametrize("freq,dur,amp,phase", [
    (5.0,  1, 1.0, 0.0),
    (37.5, -1, 2.3, 100.7),
    (123.0, [0, 5], 0.5, -1.1),
    (123.0, [0, -5], 0.5, -0.5),
])
def test_sine(fs, tol, freq, dur, amp, phase):
    gen = GenSignal(sample_rate=fs)
    sig = gen.sine(freq=freq, duration=dur, amp=amp, phase=phase)
    t, x = sig.t, sig.samples

    assert_timegrid(t, fs, dur, tol)
    start, end = parse_duration(dur)
    print(start, end)

    assert np.isclose(x[0],  amp*np.sin(2*np.pi*freq*start + phase), atol=tol)
    assert np.isclose(x[-1], amp*np.sin(2*np.pi*freq*end   + phase), atol=tol)
    assert np.max(np.abs(x)) <= amp + tol

def test_sine_zero_freq_returns_empty(fs):
    gen = GenSignal(sample_rate=fs)
    sig = gen.sine(freq=0.0, duration=[0,1], amp=1.0, phase=0.0)
    assert abs(sig.samples).sum() == 0


@pytest.mark.parametrize("freq,dur,amp,phase", [
    (5.0,  1, 1.0, 0.0),
    (37.5, 5, 2.3, 100.7),
    (123.0, [0, 5], 0.5, -1.1),
    (123.0, [0, -5], 0.5, -0.5)
])
def test_cosine(fs, tol, freq, dur, amp, phase):
    gen = GenSignal(sample_rate=fs)
    sig = gen.cosine(freq=freq, duration=dur, amp=amp, phase=phase)
    t, x = sig.t, sig.samples

    assert_timegrid(t, fs, dur, tol)
    start, end = parse_duration(dur)

    assert np.isclose(x[0],  amp*np.cos(2*np.pi*freq*start + phase), atol=tol)
    assert np.isclose(x[-1], amp*np.cos(2*np.pi*freq*end   + phase), atol=tol)
    assert np.max(np.abs(x)) <= amp + tol

def test_cosine_zero_freq_returns_empty(fs):
    gen = GenSignal(sample_rate=fs)
    sig = gen.cosine(freq=0.0, duration=[0,1], amp=1.0, phase=0.0)
    assert abs(sig.samples).sum() == 0

#----------------------------------------------------------------------
# Tests for GenSignal.sinc
#----------------------------------------------------------------------

@pytest.mark.parametrize("freq,dur,amp,phase", [
    (5.0,  1, 1.0, 0.0),
    (37.5, -1, 2.3, 100.7),
    (123.0, [0, 5], 0.5, -1.1),
    (123.0, [0, -5], 0.5, -0.5)
])
def test_sinc(fs, tol, freq, dur, amp, phase):
    gen = GenSignal(sample_rate=fs)
    sig = gen.sinc(freq=freq, duration=dur, amp=amp, phase=phase)
    t, x = sig.t, sig.samples

    assert_timegrid(t, fs, dur, tol)
    start, end = parse_duration(dur)

    assert np.isclose(x[0],  amp*np.sinc(2*np.pi*freq*start + phase), atol=tol)
    assert np.isclose(x[-1], amp*np.sinc(2*np.pi*freq*end   + phase), atol=tol)
    assert np.max(np.abs(x)) <= amp + tol

def test_sinc_zero_freq_returns_empty(fs):
    gen = GenSignal(sample_rate=fs)
    sig = gen.sinc(freq=0.0, duration=[0,1], amp=1.0, phase=0.0)
    assert abs(sig.samples).sum() == 0
