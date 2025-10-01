import numpy as np


class IdealGas:

    def __init__(self, gamma):
        self.gamma = gamma

    def pressure(self, density, energy):
        return (self.gamma - 1) * density * energy

    def soundSpeed(self, density, energy):
        return np.sqrt(self.gamma * (self.gamma - 1) * energy)

    def energy(self, density, pressure):
        return (pressure / ((self.gamma - 1) * density))


class TaitMurnaghan:
    def __init__(self, rho0, c0, n, gamma):
        self.rho0 = rho0
        self.c0 = c0
        self.n = n
        self.gamma = gamma

    def pressure(self, density, energy):
        return self.pref(density) + self.gamma * density * (energy - self.eref(density))

    def soundSpeed(self, density, energy):
        gamma = self.gamma
        rho0 = self.rho0
        c0 = self.c0
        B = rho0 * c0 ** 2
        return np.sqrt(self.gamma * (1 + self.gamma) * (energy - self.eref(density)) +
                       (B + n * self.pref(density)) / density)

    def energy(self, density, pressure):
        return (pressure - self.pref(density)) / (self.gamma * density) + self.eref(density)

    def pref(self, density):
        rho0 = self.rho0
        c0 = self.c0
        x = density / self.rho0
        n = self.n
        B = rho0 * c0 ** 2
        return B / n * (x ** n - 1)

    def eref(self, density):
        rho0 = self.rho0
        c0 = self.c0
        x = density / self.rho0
        n = self.n
        result = 1 / (n - 1) * (x ** (n - 1) - 1)
        result += 1 / x - 1
        result *= c0 ** 2 / n
        return result


class Hugoniot:
    def __init__(self, rho0, c0, s, gamma):
        self.rho0 = rho0
        self.c0 = c0
        self.s = s
        self.gamma = gamma

    def pressure(self, density, energy):
        return self.pref(density) + self.gamma * density * (energy - self.eref(density))

    def soundSpeed(self, density, energy):
        rho0 = self.rho0
        c0 = self.c0
        s = self.s
        return np.sqrt(c0 ** 2 + 4.0 * s * self.pressure(density, energy) / rho0)

    def energy(self, density, pressure):
        return (pressure - self.pref(density)) / (self.gamma * density) + self.eref(density)

    def pref(self, density):
        rho0 = self.rho0
        c0 = self.c0
        x = self.rho0 / density
        s = self.s
        B = rho0 * c0 ** 2
        tmp = 1.0 - (1.0 - x) * s;
        result = np.zeros_like(density)
        ixle1 = np.where(x < 1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = B * (1 - x[ixle1]) / (tmp[ixle1] ** 2)
        result[ixge1] = B * (1 - x[ixge1])
        return result

    def eref(self, density):
        rho0 = self.rho0
        c0 = self.c0
        x = self.rho0 / density
        s = self.s
        B = rho0 * c0 ** 2
        result = np.zeros_like(density)
        ixle1 = np.where(x < 1)
        ixge1 = np.where(x >= 1)
        result[ixle1] = 0.5 * self.pref(density[ixle1]) * (1.0 / rho0 - 1.0 / density[ixle1])
        result[ixge1] = 0.5 * B * (1.0 - x[ixge1]) ** 2 / rho0
        return result


class Hugoniot2:
    def __init__(self, gamma, rho10, c10, s1, rho20, c20, s2, rhoae_start, rhoae_end, rhoea_start, rhoea_end, curr_phase):
        self.rho10 = rho10
        self.c10 = c10
        self.s1 = s1
        self.gamma = gamma
        self.rho20 = rho20
        self.c20 = c20
        self.s2 = s2
        self.rhoae_start = rhoae_start
        self.rhoae_end = rhoae_end
        self.rhoea_start = rhoea_start
        self.rhoea_end = rhoea_end
        self.curr_phase = curr_phase.copy()
        self.prev_phases = curr_phase.copy()

    def pressure(self, density, energy):
        self.prev_phases = self.curr_phase.copy()
        for i in range(len(density)):
            dens = density[i]
            if dens > self.rhoae_end:
                self.curr_phase[i] = 'e'
            elif dens < self.rhoea_end:
                self.curr_phase[i] = 'a'
            elif self.rhoea_end < dens < self.rhoae_start:
                if self.prev_phases[i] == 'e':
                    density[i] = self.rhoea_start
            elif self.rhoea_start < dens < self.rhoae_end:
                if self.prev_phases[i] == 'a':
                    density[i] = self.rhoae_start

        return self.pref(density) + self.gamma * density * (energy - self.eref(density))

    def soundSpeed(self, density, energy):
        rho0 = np.zeros_like(density)
        c0 = np.zeros_like(density)
        s = np.zeros_like(density)
        ia = np.where(self.prev_phases == 'a')
        ie = np.where(self.prev_phases == 'e')
        rho0[ia] = self.rho10
        c0[ia] = self.c10
        s[ia] = self.s1
        rho0[ie] = self.rho20
        c0[ie] = self.c20
        s[ie] = self.s2
        return np.sqrt(c0 ** 2 + 4.0 * s * self.pressure(density, energy) / rho0)

    def energy(self, density, pressure):
        return (pressure - self.pref(density)) / (self.gamma * density) + self.eref(density)

    def pref(self, density):
        rho0 = np.zeros_like(density)
        print(rho0)
        c0 = np.zeros_like(density)
        s = np.zeros_like(density)
        ia = np.where(self.prev_phases == 'a')
        ie = np.where(self.prev_phases == 'e')
        print(ia)
        if ia:
            rho0[ia] = self.rho10
            c0[ia] = self.c10
            s[ia] = self.s1
        if ie:
            rho0[ie] = self.rho20
            c0[ie] = self.c20
            s[ie] = self.s2
        x = rho0 / density

        B = rho0 * c0 ** 2
        tmp = 1.0 - (1.0 - x) * s
        # if x < 1:
        #     return B * (1 - x) / (tmp ** 2)
        # else:
        #     return B * (1 - x)
        result = np.zeros_like(density)
        ixle1 = np.where(x < 1)
        ixge1 = np.where(x >= 1)

        result[ixle1] = B[ixle1] * (1 - x[ixle1]) / (tmp[ixle1] ** 2)
        result[ixge1] = B[ixge1] * (1 - x[ixge1])
        return result

    def eref(self, density):
        # if self.curr_phase == 'a':
        #     rho0 = self.rho10
        #     c0 = self.c10
        #     x = self.rho10 / density
        #     s = self.s1
        # elif self.curr_phase == 'e':
        #     rho0 = self.rho20
        #     c0 = self.c20
        #     x = self.rho20 / density
        #     s = self.s2

        rho0 = np.zeros_like(density)
        c0 = np.zeros_like(density)
        s = np.zeros_like(density)
        ia = np.where(self.prev_phases == 'a')
        ie = np.where(self.prev_phases == 'e')
        if ia:
            rho0[ia] = self.rho10
            c0[ia] = self.c10
            s[ia] = self.s1
        if ie:
            rho0[ie] = self.rho20
            c0[ie] = self.c20
            s[ie] = self.s2
        x = rho0 / density

        B = rho0 * c0 ** 2
        result = np.zeros_like(density)
        ixle1 = np.where(x < 1)
        ixge1 = np.where(x >= 1)
        # if x < 1:
        #     return 0.5 * self.pref(density) * (1.0 / rho0 - 1.0 / density)
        # else:
        #     return 0.5 * B * (1.0 - x) ** 2 / rho0
        result[ixle1] = 0.5 * self.pref(density[ixle1]) * (1.0 / rho0[ixle1] - 1.0 / density[ixle1])
        result[ixge1] = 0.5 * B[ixge1] * (1.0 - x[ixge1]) ** 2 / rho0[ixge1]
        return result
