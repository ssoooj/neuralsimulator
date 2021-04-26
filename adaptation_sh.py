import numpy as np
import matplotlib.pyplot as plt

class adaptation():
    def __init__(self):
        self.dt = 0.0001
        self.time = np.arange(0, 2, self.dt)
        self.A = 0.0
        self.alpha = 0.05
        self.tau_m = 0.01
        self.tau_a = 0.1
        self.v_reset = 0.0
        self.v_spike = 1.0
        self.tref = 0.003

    def simulate(self):
        stimulus = np.zeros(len(self.time))
        stimulus[(self.time > 0.2) & (self.time < 1.0)] = 4.0
        tn = self.time[0]
        V = np.random.rand() * (self.v_spike - self.v_reset) + self.v_reset

        vrec, arec = [], []
        count = 0
        for k in range(len(stimulus)):
            if self.time[k] < tn:
                continue
            count += 1
            V += (-V - self.A + stimulus[k]) * self.dt / self.tau_m
            self.A += (-self.A) * self.dt / self.tau_a
            vrec.append(V)
            arec.append(self.A)

            if V >= self.v_spike:
                V = self.v_reset
                self.A += self.alpha / self.tau_a
                tn = self.time[k] + self.tref

        # padding vrec because of the difference of the size between vrec and time
        if len(self.time) > len(vrec):
            sub_val = len(self.time) - len(vrec)
            matrix = np.zeros(sub_val)
            mV = np.append(vrec, matrix)
            pA = np.append(arec, matrix)

        plt.figure()

        plt.subplot(3, 1, 1)
        plt.plot(self.time, mV, color='yellowgreen', linewidth=1)
        plt.title('Leaky Integrate-and-Fire Model : Adaptation')
        plt.ylabel('${(mV)}$')
        plt.xlabel('${(ms)}$')

        plt.subplot(3, 1, 2)
        plt.plot(self.time, pA, color='hotpink', linewidth=1)
        plt.ylabel('${(pA)}$')
        plt.xlabel('${(ms)}$')

        plt.subplot(3, 1, 3)
        plt.plot(self.time, stimulus, color='orange', linewidth=1)
        plt.ylabel('Input Stimulus ${(mV)}$')
        plt.xlabel('Time ${(ms)}$')
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.6)
        plt.show()


a = adaptation()
a.simulate()