import numpy as np
import matplotlib.pyplot as plt

class Bursting():
    def __init__(self):
        self.dt = 0.05
        self.time = np.arange(0, 200, self.dt)
        self.tau_ref = 0.03
        self.R = 10
        self.tau_m = 8
        self.w = 100.0
        self.v_reset = -70.6
        self.v_rest = -70.0
        self.tau_w = 144.0
        self.VT = -50.4

    def I(self, time):
        return - 5 * (time < 10) + 5 * (time < 30) - 8 * (time < 50) + 8 * (time < 90) - 10 * (time < 140) + 10 * (time < 170)

    def simulate(self):
        k = int(self.tau_ref / self.dt)
        Vm = np.zeros(np.size(self.time))

        for i, t in enumerate(self.time):
            if t > 0:
                if Vm[i - 1] >= self.VT:
                    Vm[i] = self.v_reset
                    k -= 1
                elif Vm[i - 1] == self.v_reset and k >= 0:
                    Vm[i] = self.v_reset
                    k -= 1
                else:
                    Vm[i] = Vm[i - 1] + self.dt * (-(Vm[i - 1] - self.v_reset) + self.I(t) * self.R) / self.tau_m
                    k = int(self.tau_ref / self.dt)
            else:
                Vm[i] = self.v_reset

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(self.time, Vm, color='skyblue', linewidth = 1)
        plt.title('Leaky Integrate-and-Fire Model')
        plt.ylabel('${(mV)}$')
        plt.xlabel('${(ms)}$')

        plt.subplot(2, 1, 2)
        plt.plot(self.time, self.I(self.time), color='pink', linewidth = 1)
        plt.ylabel('${(nA)}$')
        plt.xlabel('${(ms)}$')
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        plt.show()

a = Bursting()
a.simulate()