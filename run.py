from signals.signals import create_cosine_wave, create_sinc_wave, create_sine_wave, create_unit_step, create_pulse
from signals.plot_signal import plot_sig

duration = [-2,2]

#t1,cosine = create_cosine_wave(5,2,1000,amp=2,phase=np.pi/2)
t,sine = create_sine_wave(duration, 2)
#t3, sinh = create_sinc_wave(-4,4, 1000)
#t4, unit_step1 = create_unit_step(duration, amp=1)
#t6, unit_step2 = create_unit_step(duration, amp=1)

#t5, pulse = create_pulse(duration, amp=4)

#print(unit_step)
#print(sinh)

#plot_sig(t1,cosine)
plot_sig(t,sine)
#plot_sig(t3,sinh)
#plot_sig(t4, unit_step)

#plot_sig(t5,pulse)
#plot_sig(t4,unit_step1)
#plot_sig(t4, unit_step2)

duration = [-4,4]

#t, pulse = create_pulse(duration, amp=4, displace=-1.5)
#t, step = create_unit_step(duration, displace=2)

#
