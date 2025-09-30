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
