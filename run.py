from signals import GenSignal

if __name__ == "__main__":
    gen = GenSignal(sample_rate=1000)
    duration = [-2, 2]

    sine = gen.sine(duration=duration)
    cosine = gen.cosine(duration=duration)
    sinc = gen.sinc(duration=duration)

    sine.add_to_plot(fig_num=1, show=False)
    cosine.add_to_plot(fig_num=2, show=False)
    sinc.add_to_plot(fig_num=3, show=True)

    unit_step = gen.unit_step(duration=duration)
    pulse = gen.pulse(duration=duration)
    ramp = gen.ramp(duration=duration)
    
    unit_step.add_to_plot(fig_num=1, show=False)
    pulse.add_to_plot(fig_num=2, show=False)
    ramp.add_to_plot(fig_num=3, show=True)

    triangle = gen.triangle(duration=duration, amp=2)
    triangle_shifted = triangle.shift(2.0)
    triangle_convolved_step = triangle.convolution(unit_step)

    triangle.add_to_plot(fig_num=1, show=False)
    triangle_shifted.add_to_plot(fig_num=2, show=False)
    triangle_convolved_step.add_to_plot(fig_num=3, show=True)



